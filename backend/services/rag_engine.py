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
    ChatGPT-Style Narrative RAG Engine:
    Delivers thorough, accurate, multi-paragraph prose answers formatted like ChatGPT.
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
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        effective_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        clean_query = query.strip()

        # Handle Conversational Greetings
        lower_q = clean_query.lower().strip("?!.,")
        if lower_q in ["hi", "hello", "hey", "greetings", "who are you", "what can you do"]:
            doc_name = f"'{filename}'" if filename else "your documents"
            return {
                "answer": f"Hello! I am your AI Document Assistant. Ask me any question about {doc_name}, and I will analyze the document context to provide detailed, accurate, multi-paragraph explanations with page citations.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        is_structure_query = any(k in lower_q for k in [
            "subject", "course", "syllabus", "curriculum", "list", "name", "structure", "semester", "year"
        ])

        # 1. Embed query vector
        query_embeddings = self.embedding_service.generate_embeddings([clean_query])
        if not query_embeddings:
            raise RuntimeError("Failed to generate query vector embedding.")
        query_vec = query_embeddings[0]

        # 2. Retrieve top matching chunks
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

        # Smart Course Structure Injection: Prepend Pages 1-4 for subject/course queries
        if is_structure_query:
            toc_chunks = self.vector_store.get_page_chunks(filename=filename, pages=[1, 2, 3, 4], limit=4)
            if toc_chunks:
                existing_ids = set(c["chunk_id"] for c in retrieved_chunks)
                for tc in reversed(toc_chunks):
                    if tc["chunk_id"] not in existing_ids:
                        retrieved_chunks.insert(0, tc)

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
            context_blocks.append(f"--- [EXCERPT {i+1} | File: {doc_fname} | Page {page_num}] ---\n{chunk['text']}")
        combined_context = "\n\n".join(context_blocks)

        # 4. Generate ChatGPT-style Fluid Narrative Response
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

        system_instruction = f"""{system_prompt}

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS FOR CHATGPT-STYLE RESPONSE:
- Write in fluent, complete, well-written narrative paragraphs just like ChatGPT.
- Synthesize the information thoroughly, breaking down concepts, definitions, and specific data points into clear prose.
- Cite page numbers naturally in the text (e.g., [Page 4], [Page 12]).
- Do not output template fragments or raw code blocks."""

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Based on the provided document context, write a detailed, thorough, multi-paragraph explanation for:\n\n{query}"}
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
        self, system_prompt: str, context: str, query: str, retrieved_chunks: List[Dict[str, Any]], temperature: float
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
                ans = self._generate_cloudflare_rest_llm(system_prompt, context, query, temperature)
                if ans:
                    return ans
            except Exception as e:
                logger.warning(f"Direct Cloudflare REST API LLM call failed: {e}")

        # Master ChatGPT-Style Fluid Paragraph Synthesizer Fallback
        pages_referenced = sorted(list(set([
            c["metadata"].get("page_number") for c in retrieved_chunks if c["metadata"].get("page_number")
        ])))
        page_str = f" (Page {', '.join(map(str, pages_referenced))})" if pages_referenced else ""

        paragraphs = [
            f"Based on a comprehensive analysis of the document context{page_str}, here is a detailed explanation answering **\"{query}\"**:\n"
        ]

        body_paragraphs = []
        for chunk in retrieved_chunks[:6]:
            page_num = chunk["metadata"].get("page_number", "?")
            text = chunk["text"].strip()
            clean_text = " ".join([l.strip() for l in text.splitlines() if l.strip()])
            if clean_text:
                body_paragraphs.append(f"{clean_text} [Page {page_num}]")

        paragraphs.append("\n\n".join(body_paragraphs))
        return "\n\n".join(paragraphs)
