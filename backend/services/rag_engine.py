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

logger = logging.getLogger(__name__)

class RAGEngine:
    """
    High-Quality Natural RAG Engine:
    Delivers direct, intelligent chatbot responses without artificial template boilerplate.
    """

    def __init__(self, embedding_service, vector_store):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.account_id = CLOUDFLARE_ACCOUNT_ID
        self.api_token = CLOUDFLARE_API_TOKEN
        self.llm_model = CLOUDFLARE_LLM_MODEL

    def answer_query(
        self,
        query: str,
        filename: str = None,
        system_prompt: str = None,
        top_k: int = 5,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Executes complete natural RAG conversation pipeline.
        """
        effective_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        clean_query = query.strip()

        # Handle Conversational Greetings Naturally
        lower_q = clean_query.lower().strip("?!.,")
        if lower_q in ["hi", "hello", "hey", "greetings", "who are you", "what can you do"]:
            doc_name = f"'{filename}'" if filename else "your documents"
            return {
                "answer": f"Hello! I am your Document AI Assistant. Ask me any question about {doc_name}, and I will analyze the content and provide clear, accurate answers for you.",
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
                "answer": "I searched the document, but I could not find relevant information matching your question.",
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

        # 4. Generate Natural Chatbot Answer
        answer = self._generate_chatbot_response(
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

    def _generate_cloudflare_llm(
        self, system_prompt: str, context: str, query: str, temperature: float
    ) -> str:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.llm_model}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        messages = [
            {"role": "system", "content": f"{system_prompt}\n\nDocument Context:\n{context}\n\nAnswer the user query concisely and naturally based strictly on the provided context."},
            {"role": "user", "content": query}
        ]

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=45)
        if resp.status_code != 200:
            raise RuntimeError(f"Cloudflare Workers AI LLM HTTP {resp.status_code}")

        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"Cloudflare AI LLM payload failure")

        result = data.get("result", {})
        ans = result.get("response", "").strip()
        return ans

    def _generate_chatbot_response(
        self, system_prompt: str, context: str, query: str, retrieved_chunks: List[Dict[str, Any]], temperature: float
    ) -> str:
        # Try Cloudflare LLM Generation first
        if self.account_id and self.api_token:
            try:
                llm_ans = self._generate_cloudflare_llm(system_prompt, context, query, temperature)
                if llm_ans and len(llm_ans) > 5:
                    return llm_ans
            except Exception as e:
                logger.warning(f"Cloudflare Workers AI LLM call failed: {e}. Using high-precision synthesis engine.")

        # High-Precision Natural Synthesis Engine (No template fluff)
        pages_referenced = sorted(list(set([
            c["metadata"].get("page_number") for c in retrieved_chunks if c["metadata"].get("page_number")
        ])))
        page_str = f" (Page {', '.join(map(str, pages_referenced))})" if pages_referenced else ""

        # Extract primary facts directly matching the query
        top_text = retrieved_chunks[0]["text"].strip()
        
        # Build natural paragraph output
        if len(retrieved_chunks) == 1:
            return f"Based on the document{page_str}:\n\n{top_text}"

        # Combine facts seamlessly
        key_snippets = []
        for chunk in retrieved_chunks[:3]:
            txt = chunk["text"].strip()
            # Clean up linebreaks
            clean_txt = " ".join([l.strip() for l in txt.splitlines() if l.strip()])
            page_num = chunk["metadata"].get("page_number")
            p_tag = f" *(Page {page_num})*" if page_num else ""
            key_snippets.append(f"• {clean_txt}{p_tag}")

        facts_block = "\n".join(key_snippets)
        return f"Based on the document{page_str}, here are the key details matching your question:\n\n{facts_block}"
