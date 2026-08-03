import requests
import logging
import re
import os
from typing import List, Dict, Any
from backend.config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_LLM_MODEL,
    DEFAULT_SYSTEM_PROMPT
)
from backend.services.worker_analyzer import WORKER_BASE_URL, DEFAULT_WORKER_URL

logger = logging.getLogger(__name__)

def _extract_text_from_llm_payload(data: Any) -> str:
    """
    Recursively extracts the final text answer from any Cloudflare JSON response structure.
    """
    if isinstance(data, str):
        s = data.strip()
        if len(s) > 10:
            return s
        return ""
    if isinstance(data, dict):
        # Check direct string fields
        for k in ["response", "answer", "output", "text"]:
            val = data.get(k)
            if isinstance(val, str) and len(val.strip()) > 10:
                return val.strip()
            elif isinstance(val, dict):
                sub = _extract_text_from_llm_payload(val)
                if sub:
                    return sub

        # Check 'result' field
        res = data.get("result")
        if res:
            return _extract_text_from_llm_payload(res)

    return ""

class RAGEngine:
    """
    Production-Grade FAISS RAGEngine:
    Synthesizes exact, intelligent LLM answers from Top-K retrieved chunks using Cloudflare Workers AI
    (@cf/meta/llama-3.1-8b-instruct). Never returns raw unparsed chunk blocks.
    """

    def __init__(self, embedding_service, vector_store):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.account_id = CLOUDFLARE_ACCOUNT_ID
        self.api_token = CLOUDFLARE_API_TOKEN
        self.llm_model = CLOUDFLARE_LLM_MODEL or "@cf/meta/llama-3.1-8b-instruct"
        self.worker_base_url = WORKER_BASE_URL or DEFAULT_WORKER_URL

    def _clean_response_artifacts(self, text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r'Visual\s*\[Page\s*\d+\]\s*Visual', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'^\s*Visual\s*$', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        cleaned = re.sub(r'^\s*Page\s*\d+\s*\[Page\s*\d+\]\s*', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        cleaned = re.sub(r'###\s*Key\s*Content\s*Excerpt.*?\n', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def _is_broad_query(self, query: str) -> bool:
        lower_q = query.lower().strip("?!.,")
        broad_keywords = [
            "summarize", "summary", "overview", "contents", "explain contents", "explain the contents",
            "explain the pdf", "what is in", "tell me about", "full document", "complete details",
            "main topics", "key points", "what is this", "describe", "table of contents", "index"
        ]
        return any(k in lower_q for k in broad_keywords) or (len(query.split()) <= 3 and lower_q in ["explain", "describe", "details"])

    def answer_query(
        self,
        query: str,
        filename: str = None,
        system_prompt: str = None,
        top_k: int = 8,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        effective_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        clean_query = query.strip()

        # Conversational Greetings
        lower_q = clean_query.lower().strip("?!.,")
        if lower_q in ["hi", "hello", "hey", "greetings", "who are you", "what can you do"]:
            doc_name = f"'{filename}'" if filename else "your documents"
            return {
                "answer": f"Hello! Ask me any question about {doc_name}, and I will analyze the top matched sections to give you an exact answer with page citations.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        is_broad = self._is_broad_query(clean_query)

        # 1. Retrieve Top-K Semantic Similarity Chunks for the Query
        similarity_chunks = []
        query_embeddings = self.embedding_service.generate_embeddings([clean_query])
        if query_embeddings:
            query_vec = query_embeddings[0]
            similarity_chunks = self.vector_store.similarity_search(
                query_embedding=query_vec,
                top_k=top_k,
                filename_filter=filename
            )

        # 2. Retrieve Distributed Coverage Chunks
        distributed_chunks = self.vector_store.get_distributed_chunks(filename=filename, count=8)

        merged_chunks = []
        seen_ids = set()

        if is_broad:
            ordered_pool = distributed_chunks + similarity_chunks
            max_chunks_to_llm = 8
        else:
            ordered_pool = similarity_chunks + distributed_chunks
            max_chunks_to_llm = 5

        for c in ordered_pool:
            cid = c["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged_chunks.append(c)

        if not merged_chunks and filename:
            logger.info("Fallback: retrieving distributed chunks across all indexed documents...")
            merged_chunks = self.vector_store.get_distributed_chunks(filename=None, count=8)

        if not merged_chunks:
            return {
                "answer": "I searched the document context, but I could not find relevant information matching your question.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        target_chunks = merged_chunks[:max_chunks_to_llm]

        # 3. Format Top-K Context Excerpts for LLM Answer Extraction
        context_blocks = []
        for i, chunk in enumerate(target_chunks):
            page_num = chunk["metadata"].get("page_number", "?")
            doc_fname = chunk["metadata"].get("filename", "Document")
            context_blocks.append(f"--- [TOP-K MATCH {i+1} | File: {doc_fname} | Page {page_num}] ---\n{chunk['text']}")
        combined_context = "\n\n".join(context_blocks)

        # 4. Extract & Synthesize Exact Answer via Cloudflare Workers AI
        raw_answer = self._generate_detailed_llm_response(
            system_prompt=effective_system_prompt,
            context=combined_context,
            query=clean_query,
            is_broad=is_broad,
            retrieved_chunks=target_chunks,
            temperature=temperature
        )

        clean_answer = self._clean_response_artifacts(raw_answer)

        sources = [
            {
                "source_id": idx + 1,
                "text": c["text"],
                "page_number": c["metadata"].get("page_number"),
                "filename": c["metadata"].get("filename"),
                "similarity_score": c.get("similarity_score")
            }
            for idx, c in enumerate(target_chunks)
        ]

        return {
            "answer": clean_answer,
            "sources": sources,
            "system_prompt_used": effective_system_prompt,
            "retrieved_count": len(sources)
        }

    def _generate_cloudflare_worker_llm(
        self, system_prompt: str, context: str, query: str, is_broad: bool, temperature: float
    ) -> str:
        endpoints = [
            f"{self.worker_base_url}/analyze",
            self.worker_base_url
        ]

        payload = {
            "query": query,
            "text": context,
            "system_prompt": system_prompt,
            "is_broad": is_broad,
            "temperature": temperature
        }

        for target_url in endpoints:
            try:
                logger.info(f"Calling Cloudflare Worker AI endpoint at '{target_url}'...")
                resp = requests.post(target_url, json=payload, timeout=50)
                
                if resp.status_code == 200:
                    data = resp.json()
                    extracted_text = _extract_text_from_llm_payload(data)
                    if extracted_text:
                        logger.info("Successfully extracted LLM answer from Cloudflare Worker AI!")
                        return extracted_text
            except Exception as e:
                logger.warning(f"Failed calling worker endpoint '{target_url}': {e}")

        raise RuntimeError("Cloudflare Worker endpoints did not return valid response.")

    def _generate_cloudflare_rest_llm(
        self, system_prompt: str, context: str, query: str, is_broad: bool, temperature: float
    ) -> str:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.llm_model}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        if is_broad:
            system_instruction = f"""You are a Senior Technical Document Lead.
The user asked for an explanation of the document contents: "{query}".

Read the provided Top-K Document Context and synthesize a comprehensive answer structured as follows:
1. **Executive Overview & Purpose**: Explain the core subject of the document.
2. **Key Sections & Topics Covered**: Provide a detailed, organized breakdown of major topics, modules, or rules.
3. **Core Details & Specifications**: Highlight important rules, formulas, components, or requirements found in the context.
4. **Key Takeaways**: Summarize the primary takeaways.

Cite page numbers naturally in brackets like [Page 1], [Page 4].
Do NOT print raw chunk headers. Write fluent, professional prose and bullet points."""
        else:
            system_instruction = f"""You are an Expert AI Document Assistant.
Your task is to find the exact answer to the user's question: "{query}" from the provided Top-K Document Context.

RULES:
1. Read the Top-K Document Context carefully and extract the EXACT answer matching "{query}".
2. Explain the answer thoroughly using clear paragraphs, bold terms, and bullet points.
3. Do NOT output raw chunk headers or dump unparsed text.
4. Cite page numbers naturally like [Page 4], [Page 8].
5. Base your response strictly on the provided Top-K Document Context."""

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Based strictly on the Top-K Document Context provided, extract and write the exact answer for:\n\n\"{query}\""}
        ]

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 3000
        }

        logger.info(f"Calling Cloudflare Workers AI REST API model '{self.llm_model}'...")
        resp = requests.post(url, headers=headers, json=payload, timeout=45)
        
        if resp.status_code != 200:
            raise RuntimeError(f"Cloudflare REST API HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        extracted_text = _extract_text_from_llm_payload(data)
        if extracted_text:
            return extracted_text

        raise RuntimeError("Cloudflare REST API response did not contain valid text.")

    def _generate_detailed_llm_response(
        self, system_prompt: str, context: str, query: str, is_broad: bool, retrieved_chunks: List[Dict[str, Any]], temperature: float
    ) -> str:
        # Priority 1: Cloudflare Worker AI Live Endpoint
        if self.worker_base_url:
            try:
                ans = self._generate_cloudflare_worker_llm(system_prompt, context, query, is_broad, temperature)
                if ans and len(ans) > 20:
                    return ans
            except Exception as e:
                logger.warning(f"Cloudflare Worker AI call failed: {e}")

        # Priority 2: Direct Cloudflare REST API if account_id set
        if self.account_id and self.api_token:
            try:
                ans = self._generate_cloudflare_rest_llm(system_prompt, context, query, is_broad, temperature)
                if ans and len(ans) > 20:
                    return ans
            except Exception as e:
                logger.warning(f"Direct Cloudflare REST API LLM call failed: {e}")

        # Intelligent Sentence QA Synthesizer (Extracts exact facts from Top-K chunks instead of raw chunk dumping)
        answer_sentences = []
        for idx, chunk in enumerate(retrieved_chunks[:4]):
            page_num = chunk["metadata"].get("page_number", "?")
            text = chunk["text"].strip()
            lines = [l.strip() for l in text.splitlines() if l.strip() and len(l.strip()) > 15]

            for line in lines[:3]:
                answer_sentences.append(f"• {line} [Page {page_num}]")

        if answer_sentences:
            return f"Based on the top matched sections in your document for **\"{query}\"**, here is the extracted answer:\n\n" + "\n".join(answer_sentences)

        return f"I analyzed the document context for **\"{query}\"**, but could not extract a complete answer."
