import requests
import logging
import re
from typing import List, Dict, Any
from backend.config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_LLM_MODEL,
    DEFAULT_SYSTEM_PROMPT
)
from backend.services.worker_analyzer import WORKER_BASE_URL

logger = logging.getLogger(__name__)

class RAGEngine:
    """
    FAISS-Powered High-Precision RAGEngine:
    Prioritizes query-specific semantic vector search matches and calls Cloudflare Workers AI
    to deliver intelligent, ChatGPT-quality responses without raw chunk headers.
    """

    def __init__(self, embedding_service, vector_store):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.account_id = CLOUDFLARE_ACCOUNT_ID
        self.api_token = CLOUDFLARE_API_TOKEN
        self.llm_model = CLOUDFLARE_LLM_MODEL or "@cf/meta/llama-3.1-8b-instruct"
        self.worker_base_url = WORKER_BASE_URL

    def _clean_response_artifacts(self, text: str) -> str:
        if not text:
            return ""
        # Filter raw OCR image noise & ugly chunk header prefixes
        cleaned = re.sub(r'Visual\s*\[Page\s*\d+\]\s*Visual', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'^\s*Visual\s*$', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        cleaned = re.sub(r'^\s*Page\s*\d+\s*\[Page\s*\d+\]\s*', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

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
                "answer": f"Hello! Ask me any question about {doc_name}, and I will provide a direct, detail-specific answer with page citations.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        is_summary_query = any(k in lower_q for k in [
            "summarize", "summary", "overview", "explain complete details", "full document", "tell me about", "what is this pdf", "describe the pdf"
        ])

        # 1. Retrieve Top Semantic Similarity Chunks for the EXACT Query
        similarity_chunks = []
        query_embeddings = self.embedding_service.generate_embeddings([clean_query])
        if query_embeddings:
            query_vec = query_embeddings[0]
            similarity_chunks = self.vector_store.similarity_search(
                query_embedding=query_vec,
                top_k=top_k,
                filename_filter=filename
            )

        # 2. Retrieve Distributed Coverage Chunks for broad context
        distributed_chunks = self.vector_store.get_distributed_chunks(filename=filename, count=8)

        # Priority Order: Similarity Chunks FIRST for specific questions
        merged_chunks = []
        seen_ids = set()

        if is_summary_query:
            ordered = distributed_chunks + similarity_chunks
        else:
            ordered = similarity_chunks + distributed_chunks

        for c in ordered:
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

        # 3. Format Context Excerpts cleanly
        context_blocks = []
        for i, chunk in enumerate(merged_chunks[:8]):
            page_num = chunk["metadata"].get("page_number", "?")
            doc_fname = chunk["metadata"].get("filename", "Document")
            context_blocks.append(f"--- [DOCUMENT EXCERPT {i+1} | Page {page_num}] ---\n{chunk['text']}")
        combined_context = "\n\n".join(context_blocks)

        # 4. Generate High-Quality LLM Response via Cloudflare Workers AI
        raw_answer = self._generate_detailed_llm_response(
            system_prompt=effective_system_prompt,
            context=combined_context,
            query=clean_query,
            retrieved_chunks=merged_chunks[:8],
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
            for idx, c in enumerate(merged_chunks[:8])
        ]

        return {
            "answer": clean_answer,
            "sources": sources,
            "system_prompt_used": effective_system_prompt,
            "retrieved_count": len(sources)
        }

    def _generate_cloudflare_worker_llm(
        self, system_prompt: str, context: str, query: str, temperature: float
    ) -> str:
        target_url = self.worker_base_url
        if not target_url:
            raise ValueError("Worker base URL is not set.")

        payload = {
            "query": query,
            "text": context,
            "system_prompt": system_prompt,
            "temperature": temperature
        }

        logger.info(f"Calling Cloudflare Worker AI link at '{target_url}'...")
        resp = requests.post(target_url, json=payload, timeout=55)
        
        if resp.status_code == 200:
            data = resp.json()
            ans = (
                data.get("response") or
                data.get("result", {}).get("response") or
                data.get("result") or
                ""
            )
            if isinstance(ans, str) and len(ans.strip()) > 10:
                logger.info("Successfully generated LLM response via Cloudflare Worker AI link!")
                return ans.strip()

        raise RuntimeError(f"Cloudflare Worker HTTP {resp.status_code}: {resp.text}")

    def _generate_cloudflare_rest_llm(
        self, system_prompt: str, context: str, query: str, temperature: float
    ) -> str:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.llm_model}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        system_instruction = f"""You are an expert AI Document Analyst.
Synthesize the provided Document Context into a clear, intelligent, well-structured answer.

STRICT INSTRUCTIONS:
1. Directly answer the user's question: "{query}".
2. Do NOT output raw chunk headers (do NOT print "Page 1 [Page 1]" or "Tensor [Page 1]").
3. Explain concepts, rules, modules, and topics thoroughly using bullet points and clear paragraphs.
4. Cite page numbers naturally in the text (e.g. [Page 1], [Page 2]).
5. Base your response strictly on the provided DOCUMENT CONTEXT."""

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Based strictly on the provided document context, write a clear, well-structured answer for:\n\n\"{query}\""}
        ]

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2500
        }

        logger.info(f"Calling Cloudflare Workers AI REST API model '{self.llm_model}'...")
        resp = requests.post(url, headers=headers, json=payload, timeout=50)
        
        if resp.status_code != 200:
            raise RuntimeError(f"Cloudflare REST API HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"Cloudflare AI LLM error: {data.get('errors')}")

        result = data.get("result", {})
        ans = result.get("response", "").strip()
        return ans

    def _generate_detailed_llm_response(
        self, system_prompt: str, context: str, query: str, retrieved_chunks: List[Dict[str, Any]], temperature: float
    ) -> str:
        # Priority 1: Direct Cloudflare REST API call if credentials present
        if self.account_id and self.api_token:
            try:
                ans = self._generate_cloudflare_rest_llm(system_prompt, context, query, temperature)
                if ans and len(ans) > 20:
                    return ans
            except Exception as e:
                logger.warning(f"Direct Cloudflare REST API LLM call failed: {e}")

        # Priority 2: Worker Base URL call
        if self.worker_base_url:
            try:
                ans = self._generate_cloudflare_worker_llm(system_prompt, context, query, temperature)
                if ans and len(ans) > 20:
                    return ans
            except Exception as e:
                logger.warning(f"Cloudflare Worker AI link call failed: {e}")

        # Clean Factual Fallback Synthesizer (No "Page 1 [Page 1]" snippet dumps)
        body_items = []
        for idx, chunk in enumerate(retrieved_chunks[:6]):
            page_num = chunk["metadata"].get("page_number", "?")
            text = chunk["text"].strip()
            cleaned_text = " ".join([l.strip() for l in text.splitlines() if l.strip()])

            if cleaned_text:
                body_items.append(f"• **Key Topic (Page {page_num})**: {cleaned_text}")

        if body_items:
            return f"Based on the document context, here are the key details for **\"{query}\"**:\n\n" + "\n\n".join(body_items)

        return f"I analyzed the document context for **\"{query}\"**, but could not generate a complete summary."
