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
    High-Quality ChatGPT-Style RAG Engine:
    Uses Cloudflare Workers AI Llama-3 8B Instruct with BGE Large vectors
    to deliver rich, detailed, multi-paragraph explanations with page citations.
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
        top_k: int = 5,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        effective_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        clean_query = query.strip()

        # Handle Conversational Greetings
        lower_q = clean_query.lower().strip("?!.,")
        if lower_q in ["hi", "hello", "hey", "greetings", "who are you", "what can you do"]:
            doc_name = f"'{filename}'" if filename else "your documents"
            return {
                "answer": f"Hello! I am your AI Document Assistant. Ask me any question about {doc_name}, and I will analyze the text to provide comprehensive, detailed answers with page citations.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        # 1. Embed query vector using BGE Large / Embedding Service
        query_embeddings = self.embedding_service.generate_embeddings([clean_query])
        if not query_embeddings:
            raise RuntimeError("Failed to generate query vector embedding.")
        query_vec = query_embeddings[0]

        # 2. Retrieve top matching chunks (First try with filename filter, fallback to all chunks)
        retrieved_chunks = self.vector_store.similarity_search(
            query_embedding=query_vec,
            top_k=top_k,
            filename_filter=filename
        )

        if not retrieved_chunks and filename:
            logger.info(f"No chunks found for filename filter '{filename}'. Searching across all indexed chunks...")
            retrieved_chunks = self.vector_store.similarity_search(
                query_embedding=query_vec,
                top_k=top_k,
                filename_filter=None
            )

        if not retrieved_chunks:
            return {
                "answer": "I searched the document context, but I could not find relevant information matching your question.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        # 3. Format Context Excerpts
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            page_num = chunk["metadata"].get("page_number", "?")
            doc_fname = chunk["metadata"].get("filename", "Document")
            context_blocks.append(f"[Excerpt {i+1} | File: {doc_fname} | Page {page_num}]\n{chunk['text']}")
        combined_context = "\n\n".join(context_blocks)

        # 4. Generate ChatGPT-style Detailed Response via Cloudflare Workers AI
        answer = self._generate_detailed_llm_response(
            system_prompt=effective_system_prompt,
            context=combined_context,
            query=clean_query,
            retrieved_chunks=retrieved_chunks,
            temperature=temperature
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

    def _generate_cloudflare_rest_llm(
        self, system_prompt: str, context: str, query: str, temperature: float
    ) -> str:
        """
        Calls Cloudflare Workers AI REST API directly (@cf/meta/llama-3-8b-instruct)
        """
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.llm_model}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        system_instruction = f"""{system_prompt}

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
- Provide a detailed, comprehensive, multi-paragraph response like ChatGPT.
- Use clear Markdown headers, bold terms, and bullet points.
- Explicitly cite page numbers from the context (e.g. Page 4, Page 12).
- Base your answer strictly on the provided document excerpts."""

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": query}
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

    def _generate_cloudflare_worker_llm(
        self, system_prompt: str, context: str, query: str, temperature: float
    ) -> str:
        """
        Calls Cloudflare Worker endpoint if LLM_ANALYSIS_URL is set
        """
        worker_url = f"{self.worker_base_url}/chat" if not self.worker_base_url.endswith("/chat") else self.worker_base_url
        payload = {
            "query": query,
            "text": context,
            "system_prompt": system_prompt,
            "temperature": temperature
        }

        resp = requests.post(worker_url, json=payload, timeout=50)
        if resp.status_code == 200:
            data = resp.json()
            ans = data.get("response") or data.get("result", {}).get("response", "")
            if isinstance(ans, str) and len(ans.strip()) > 10:
                return ans.strip()

        raise RuntimeError("Worker call failed")

    def _generate_detailed_llm_response(
        self, system_prompt: str, context: str, query: str, retrieved_chunks: List[Dict[str, Any]], temperature: float
    ) -> str:
        # 1. Try Direct Cloudflare Workers AI REST API
        if self.account_id and self.api_token:
            try:
                ans = self._generate_cloudflare_rest_llm(system_prompt, context, query, temperature)
                if ans:
                    return ans
            except Exception as e:
                logger.warning(f"Direct Cloudflare REST API LLM call failed: {e}")

        # 2. Try Worker URL endpoint
        if self.worker_base_url:
            try:
                ans = self._generate_cloudflare_worker_llm(system_prompt, context, query, temperature)
                if ans:
                    return ans
            except Exception as e:
                logger.warning(f"Cloudflare Worker LLM call failed: {e}")

        # 3. Comprehensive Detailed Synthesizer Fallback
        pages_referenced = sorted(list(set([
            c["metadata"].get("page_number") for c in retrieved_chunks if c["metadata"].get("page_number")
        ])))
        page_str = f" (Page {', '.join(map(str, pages_referenced))})" if pages_referenced else ""

        response_sections = [
            f"Based on a detailed analysis of the document context{page_str}, here is a comprehensive breakdown for **\"{query}\"**:\n"
        ]

        for idx, chunk in enumerate(retrieved_chunks[:4]):
            page_num = chunk["metadata"].get("page_number", "?")
            text = chunk["text"].strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            if lines:
                section_title = lines[0][:80] if len(lines[0]) < 80 else f"Document Analysis - Page {page_num}"
                section_body = "\n".join(lines[1:]) if len(lines) > 1 else lines[0]

                response_sections.append(f"### {idx+1}. {section_title} *(Page {page_num})*\n")
                clean_body = re.sub(r'\s+', ' ', section_body)
                response_sections.append(f"{clean_body}\n")

        response_sections.append("### Summary & Key Takeaways\n")
        takeaways = []
        for c in retrieved_chunks[:3]:
            snippet = c["text"].strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            pg = c["metadata"].get("page_number", "?")
            takeaways.append(f"- **Page {pg}**: {snippet}")

        response_sections.append("\n".join(takeaways))

        return "\n\n".join(response_sections)
