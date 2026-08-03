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
    FAISS-Powered Direct Detail-Specific RAG Engine:
    Delivers exact, highly detailed, direct document-backed explanations with natural page citations.
    """

    def __init__(self, embedding_service, vector_store):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.account_id = CLOUDFLARE_ACCOUNT_ID
        self.api_token = CLOUDFLARE_API_TOKEN
        self.llm_model = CLOUDFLARE_LLM_MODEL
        self.worker_base_url = WORKER_BASE_URL

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
                "answer": f"Hello! I am your AI Master Document Assistant. Ask me any question about {doc_name}, and I will analyze the document to provide a direct, highly detailed answer with page citations.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        is_summary_query = any(k in lower_q for k in [
            "summarize", "summary", "overview", "explain complete details", "full document", "tell me about", "what is this pdf", "describe the pdf"
        ])
        is_structure_query = any(k in lower_q for k in [
            "subject", "course", "syllabus", "curriculum", "list out", "list", "name", "structure", "semester", "year", "issue", "analyse", "analyze"
        ])

        retrieved_chunks = []

        # 1. FAISS Full-Document Coverage for Summarization Queries
        if is_summary_query:
            logger.info("Executing FAISS Full-Document Coverage Retrieval for Summarization Query...")
            retrieved_chunks = self.vector_store.get_distributed_chunks(filename=filename, count=12)

        # 2. FAISS Similarity Search for Specific Queries
        if not retrieved_chunks:
            query_embeddings = self.embedding_service.generate_embeddings([clean_query])
            if query_embeddings:
                query_vec = query_embeddings[0]
                retrieved_chunks = self.vector_store.similarity_search(
                    query_embedding=query_vec,
                    top_k=top_k,
                    filename_filter=filename
                )

        if not retrieved_chunks and filename:
            logger.info(f"No chunks found for filename filter '{filename}'. Searching across all indexed FAISS chunks...")
            retrieved_chunks = self.vector_store.get_distributed_chunks(filename=None, count=10)

        # Smart Course Structure & Table Injection: Prepend Pages 1-4
        if is_structure_query:
            toc_chunks = self.vector_store.get_page_chunks(filename=filename, pages=[1, 2, 3, 4], limit=4)
            if toc_chunks:
                existing_ids = set(c["chunk_id"] for c in retrieved_chunks)
                for tc in reversed(toc_chunks):
                    if tc["chunk_id"] not in existing_ids:
                        retrieved_chunks.insert(0, tc)

        if not retrieved_chunks:
            return {
                "answer": "I searched the document, but I could not find relevant information matching your question.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        # 3. Format Context Excerpts
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            page_num = chunk["metadata"].get("page_number", "?")
            doc_fname = chunk["metadata"].get("filename", "Document")
            context_blocks.append(f"--- [FAISS EXCERPT {i+1} | File: {doc_fname} | Page {page_num}] ---\n{chunk['text']}")
        combined_context = "\n\n".join(context_blocks)

        # 4. Generate Direct Detail-Specific Response via Cloudflare Workers AI
        answer = self._generate_detailed_llm_response(
            system_prompt=effective_system_prompt,
            context=combined_context,
            query=clean_query,
            retrieved_chunks=retrieved_chunks,
            temperature=temperature,
            is_summary=is_summary_query
        )

        sources = [
            {
                "source_id": idx + 1,
                "text": c["text"],
                "page_number": c["metadata"].get("page_number"),
                "filename": c["metadata"].get("filename"),
                "similarity_score": c.get("similarity_score")
            }
            for idx, c in enumerate(retrieved_chunks)
        ]

        return {
            "answer": answer,
            "sources": sources,
            "system_prompt_used": effective_system_prompt,
            "retrieved_count": len(retrieved_chunks)
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
        self, system_prompt: str, context: str, query: str, temperature: float, is_summary: bool = False
    ) -> str:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.llm_model}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        system_instruction = f"""You are a Master AI Document Analyst & Technical Educator.
Your task is to provide a direct, exact, highly detailed, and specific answer based strictly on the provided Document Context.

RULES FOR DIRECT & ACCURATE RESPONSES:
1. Answer the user's question DIRECTLY without artificial section titles or template intros (do NOT use "Executive Summary", "Detailed Breakdown", or "Key Takeaways").
2. Detail every relevant concept, step, definition, formula, or specification present in the context thoroughly.
3. Use bold text for key terms (**Term**) and clear bullet points or numbered lists where helpful.
4. Cite page numbers naturally in the text (e.g. [Page 4], [Page 12]).
5. Base your response strictly on the provided document context without making up unverified information.

DOCUMENT CONTEXT:
{context}"""

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Based strictly on the provided FAISS document context, write a direct, highly accurate, and detail-specific answer for:\n\n\"{query}\""}
        ]

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2500
        }

        logger.info(f"Calling Cloudflare Workers AI LLM model '{self.llm_model}'...")
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
        self, system_prompt: str, context: str, query: str, retrieved_chunks: List[Dict[str, Any]], temperature: float, is_summary: bool = False
    ) -> str:
        if self.worker_base_url:
            try:
                ans = self._generate_cloudflare_worker_llm(system_prompt, context, query, temperature)
                if ans:
                    return ans
            except Exception as e:
                logger.warning(f"Cloudflare Worker AI link call failed: {e}")

        if self.account_id and self.api_token:
            try:
                ans = self._generate_cloudflare_rest_llm(system_prompt, context, query, temperature, is_summary)
                if ans:
                    return ans
            except Exception as e:
                logger.warning(f"Direct Cloudflare REST API LLM call failed: {e}")

        # Direct Detail-Specific Fallback Synthesizer (No Artificial Template Section Titles)
        body_paragraphs = []
        for idx, chunk in enumerate(retrieved_chunks[:8]):
            page_num = chunk["metadata"].get("page_number", "?")
            text = chunk["text"].strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            if lines:
                heading = lines[0][:70] if len(lines[0]) < 70 else f"Page {page_num}"
                body = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
                body_paragraphs.append(f"{idx+1}. **{heading}** [Page {page_num}]\n{body}")

        return "\n\n".join(body_paragraphs)
