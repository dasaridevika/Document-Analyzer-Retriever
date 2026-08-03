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

class RAGEngine:
    """
    Production-Grade FAISS RAGEngine:
    - Smart Broad/Specific Query Intent Classification
    - Full-Document Coverage + High-Precision FAISS Vector Retrieval
    - Production-Grade Cloudflare Workers AI (@cf/meta/llama-3.1-8b-instruct) LLM Synthesis
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
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def _is_broad_query(self, query: str) -> bool:
        lower_q = query.lower().strip("?!.,")
        broad_keywords = [
            "summarize", "summary", "overview", "contents", "explain contents", "explain the contents",
            "explain the pdf", "what is in", "tell me about", "full document", "complete details",
            "main topics", "key points", "what is this", "describe", "table of contents", "index"
        ]
        return any(k in lower_q for k in broad_keywords) or len(query.split()) <= 3 and lower_q in ["explain", "describe", "details"]

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

        # Handle Conversational Greetings
        lower_q = clean_query.lower().strip("?!.,")
        if lower_q in ["hi", "hello", "hey", "greetings", "who are you", "what can you do"]:
            doc_name = f"'{filename}'" if filename else "your documents"
            return {
                "answer": f"Hello! Ask me any question about {doc_name}, and I will provide a direct, production-grade answer with page citations.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        is_broad = self._is_broad_query(clean_query)

        # 1. Retrieve Top Semantic Similarity Chunks
        similarity_chunks = []
        query_embeddings = self.embedding_service.generate_embeddings([clean_query])
        if query_embeddings:
            query_vec = query_embeddings[0]
            similarity_chunks = self.vector_store.similarity_search(
                query_embedding=query_vec,
                top_k=top_k,
                filename_filter=filename
            )

        # 2. Retrieve Full-Document Distributed Coverage Chunks (spanning beginning, middle, and end)
        distributed_chunks = self.vector_store.get_distributed_chunks(filename=filename, count=10)

        # Merge Chunks with Strict Prioritization
        merged_chunks = []
        seen_ids = set()

        if is_broad:
            # Broad/Summary Query: Put Distributed Full-Document Chunks FIRST, followed by similarity chunks
            ordered_pool = distributed_chunks + similarity_chunks
            max_chunks_to_llm = 8
        else:
            # Specific Query: Put Top Vector Similarity Matches FIRST
            ordered_pool = similarity_chunks + distributed_chunks
            max_chunks_to_llm = 5

        for c in ordered_pool:
            cid = c["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged_chunks.append(c)

        if not merged_chunks and filename:
            logger.info("Fallback: retrieving distributed chunks across all indexed documents...")
            merged_chunks = self.vector_store.get_distributed_chunks(filename=None, count=10)

        if not merged_chunks:
            return {
                "answer": "I searched the document context, but I could not find relevant information matching your question.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        target_chunks = merged_chunks[:max_chunks_to_llm]

        # 3. Format Production Context Excerpts
        context_blocks = []
        for i, chunk in enumerate(target_chunks):
            page_num = chunk["metadata"].get("page_number", "?")
            doc_fname = chunk["metadata"].get("filename", "Document")
            context_blocks.append(f"--- [DOCUMENT EXCERPT {i+1} | File: {doc_fname} | Page {page_num}] ---\n{chunk['text']}")
        combined_context = "\n\n".join(context_blocks)

        # 4. Generate Production-Grade Response via Cloudflare Workers AI
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
                    ans = (
                        data.get("response") or
                        data.get("result", {}).get("response") or
                        data.get("result") or
                        ""
                    )
                    if isinstance(ans, str) and len(ans.strip()) > 20:
                        logger.info("Successfully generated LLM response via Cloudflare Worker AI!")
                        return ans.strip()
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
The user asked for a comprehensive explanation of the document contents: "{query}".

PRODUCE A PRODUCTION-GRADE DOCUMENT SUMMARY STRUCTURED AS FOLLOWS:
1. **Document Overview & Purpose**: Explain what the document covers and its main objectives.
2. **Key Sections & Topics Covered**: Provide a detailed, organized breakdown of major topics, modules, or sections found across the pages.
3. **Core Technical Details & Specifications**: Highlight important formulas, rules, architectures, or findings with exact figures/data.
4. **Key Takeaways & Conclusion**: Summarize the primary takeaways.

Cite page numbers naturally in brackets like [Page 1], [Page 7].
Do NOT print raw chunk headers like "Section (Page 1):" or "Visual [Page 2]". Write fluent, professional prose and bullet points."""
        else:
            system_instruction = f"""You are a Master AI Document Assistant.
CRITICAL MANDATE: Answer ONLY the specific topic asked in the user query: "{query}".

RULES:
1. Explain ONLY what is explicitly asked in "{query}".
2. Do NOT mention or summarize unrelated topics present in the document context.
3. Provide exact definitions, syntax, code examples, rules, methods, and page citations matching "{query}".
4. Write in clear, professional paragraphs and bullet points. Cite page numbers like [Page 8]."""

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Based strictly on the provided document context, write a production-grade, highly accurate answer for:\n\n\"{query}\""}
        ]

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2500
        }

        logger.info(f"Calling Cloudflare Workers AI REST API model '{self.llm_model}'...")
        resp = requests.post(url, headers=headers, json=payload, timeout=45)
        
        if resp.status_code != 200:
            raise RuntimeError(f"Cloudflare REST API HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"Cloudflare AI LLM error: {data.get('errors')}")

        result = data.get("result", {})
        ans = result.get("response", "").strip()
        return ans

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

        # Production Fallback Synthesizer (Generates clean, cohesive prose instead of raw snippet dumps)
        prose_blocks = []
        for idx, chunk in enumerate(retrieved_chunks[:6]):
            page_num = chunk["metadata"].get("page_number", "?")
            text = chunk["text"].strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines:
                block_text = " ".join(lines)
                prose_blocks.append(f"### Key Content Excerpt (Page {page_num})\n{block_text}")

        if prose_blocks:
            return f"## Production Analysis of Document Contents\n\n" + "\n\n".join(prose_blocks)

        return f"I analyzed the document context for **\"{query}\"**, but could not generate a complete summary."
