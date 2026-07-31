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
    Delivers detailed, multi-paragraph, structured, and comprehensive document analysis.
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
        """
        Executes complete ChatGPT-style detailed response generation.
        """
        effective_system_prompt = system_prompt or """You are an expert AI Document Assistant.
Provide comprehensive, detailed, and thoroughly structured explanations like ChatGPT.
Break down topics into clear sections, use bullet points, bold key concepts, and cite page numbers whenever mentioned in the context.
Deliver rich, informative responses that answer the user's question completely based on the provided document excerpts."""

        clean_query = query.strip()

        # Handle Conversational Greetings Naturally
        lower_q = clean_query.lower().strip("?!.,")
        if lower_q in ["hi", "hello", "hey", "greetings", "who are you", "what can you do"]:
            doc_name = f"'{filename}'" if filename else "your documents"
            return {
                "answer": f"Hello! I am your AI Document Assistant. Ask me any question about {doc_name}, and I will analyze the context to provide detailed, structured explanations with page citations.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        # 1. Embed user query
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

        if not retrieved_chunks:
            return {
                "answer": "I searched the document context, but I could not find relevant information matching your question.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        # 3. Format Context
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            page_num = chunk["metadata"].get("page_number", "?")
            context_blocks.append(f"[Document Excerpt | Page {page_num}]\n{chunk['text']}")
        combined_context = "\n\n".join(context_blocks)

        # 4. Generate ChatGPT-style Detailed Answer
        answer = self._generate_detailed_chatbot_response(
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
        """
        Calls Cloudflare Worker /analyze endpoint running @cf/meta/llama-3-8b-instruct
        """
        worker_url = f"{self.worker_base_url}/chat" if self.worker_base_url else ""
        if not worker_url:
            worker_url = f"{self.worker_base_url}/analyze" if self.worker_base_url else ""

        if not worker_url:
            raise ValueError("No worker URL")

        payload = {
            "query": query,
            "text": context,
            "system_prompt": system_prompt,
            "temperature": temperature
        }

        resp = requests.post(worker_url, json=payload, timeout=45)
        if resp.status_code != 200:
            raise RuntimeError(f"Cloudflare Worker LLM returned {resp.status_code}")

        data = resp.json()
        ans = data.get("response") or data.get("result", {}).get("response", "")
        if isinstance(ans, str) and len(ans.strip()) > 10:
            return ans.strip()

        raise ValueError("Invalid worker response")

    def _generate_cloudflare_rest_llm(
        self, system_prompt: str, context: str, query: str, temperature: float
    ) -> str:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.llm_model}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        messages = [
            {"role": "system", "content": f"{system_prompt}\n\nDOCUMENT CONTEXT:\n{context}"},
            {"role": "user", "content": f"Based on the document context provided, answer the following question in detail:\n{query}"}
        ]

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2500
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=45)
        if resp.status_code != 200:
            raise RuntimeError(f"Cloudflare REST API LLM HTTP {resp.status_code}")

        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"Cloudflare AI LLM payload failure")

        result = data.get("result", {})
        ans = result.get("response", "").strip()
        return ans

    def _generate_detailed_chatbot_response(
        self, system_prompt: str, context: str, query: str, retrieved_chunks: List[Dict[str, Any]], temperature: float
    ) -> str:
        # Option A: Cloudflare Worker URL LLM
        if self.worker_base_url:
            try:
                ans = self._generate_cloudflare_worker_llm(system_prompt, context, query, temperature)
                if ans:
                    return ans
            except Exception as e:
                logger.warning(f"Cloudflare Worker LLM call failed: {e}")

        # Option B: Direct Cloudflare REST API LLM
        if self.account_id and self.api_token:
            try:
                ans = self._generate_cloudflare_rest_llm(system_prompt, context, query, temperature)
                if ans:
                    return ans
            except Exception as e:
                logger.warning(f"Direct Cloudflare REST LLM call failed: {e}")

        # Option C: High-Detailed ChatGPT-Style Synthesizer
        pages_referenced = sorted(list(set([
            c["metadata"].get("page_number") for c in retrieved_chunks if c["metadata"].get("page_number")
        ])))
        page_str = f" (Page {', '.join(map(str, pages_referenced))})" if pages_referenced else ""

        response_sections = [
            f"Based on a detailed analysis of the document{page_str}, here is a comprehensive breakdown for **\"{query}\"**:\n"
        ]

        # Group facts into structured sections
        for idx, chunk in enumerate(retrieved_chunks[:4]):
            page_num = chunk["metadata"].get("page_number", "?")
            text = chunk["text"].strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            if lines:
                section_title = lines[0][:80] if len(lines[0]) < 80 else f"Key Insights from Page {page_num}"
                section_body = "\n".join(lines[1:]) if len(lines) > 1 else lines[0]

                response_sections.append(f"### {idx+1}. {section_title} *(Page {page_num})*\n")
                
                # Format paragraph text cleanly
                clean_body = re.sub(r'\s+', ' ', section_body)
                response_sections.append(f"{clean_body}\n")

        # Summary Takeaway
        response_sections.append("### Summary & Key Takeaways\n")
        takeaways = []
        for c in retrieved_chunks[:3]:
            snippet = c["text"].strip().replace("\n", " ")
            if len(snippet) > 140:
                snippet = snippet[:140] + "..."
            pg = c["metadata"].get("page_number", "?")
            takeaways.append(f"- **Page {pg}**: {snippet}")

        response_sections.append("\n".join(takeaways))

        return "\n\n".join(response_sections)
