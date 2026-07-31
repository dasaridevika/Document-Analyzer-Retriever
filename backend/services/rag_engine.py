import requests
import logging
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
    RAG Query Engine: Combines vector retrieval with dynamic system prompts and LLM inference.
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
        top_k: int = 4,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Executes complete RAG pipeline:
        1. Embed user question
        2. Retrieve top_k matching chunks
        3. Format system prompt + context
        4. Generate response via Cloudflare Workers AI LLM (or analytical fallback)
        """
        effective_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # 1. Embed query
        query_embeddings = self.embedding_service.generate_embeddings([query])
        if not query_embeddings:
            raise RuntimeError("Failed to generate query embedding.")
        query_vec = query_embeddings[0]

        # 2. Retrieve chunks
        retrieved_chunks = self.vector_store.similarity_search(
            query_embedding=query_vec,
            top_k=top_k,
            filename_filter=filename
        )

        if not retrieved_chunks:
            return {
                "answer": "No relevant context found in the uploaded document to answer your query.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        # 3. Build Context String
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            page_num = chunk["metadata"].get("page_number", "?")
            fname = chunk["metadata"].get("filename", "Doc")
            score = chunk.get("similarity_score", 0.0)
            context_blocks.append(
                f"[Source {i+1} | {fname} | Page {page_num} | Similarity: {score:.2f}]\n{chunk['text']}"
            )
        combined_context = "\n\n".join(context_blocks)

        # 4. Generate Answer via Cloudflare Workers AI LLM (or fallback engine)
        answer = self._generate_llm_response(
            system_prompt=effective_system_prompt,
            context=combined_context,
            query=query,
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
            {"role": "system", "content": f"{system_prompt}\n\nDocument Context:\n{context}"},
            {"role": "user", "content": query}
        ]

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=45)
        if resp.status_code != 200:
            raise RuntimeError(f"Cloudflare Workers AI LLM error {resp.status_code}: {resp.text}")

        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"Cloudflare AI LLM payload failure: {data.get('errors')}")

        result = data.get("result", {})
        return result.get("response", "").strip()

    def _generate_llm_response(
        self, system_prompt: str, context: str, query: str, temperature: float
    ) -> str:
        # Check Cloudflare Workers AI credentials
        if self.account_id and self.api_token:
            try:
                return self._generate_cloudflare_llm(system_prompt, context, query, temperature)
            except Exception as e:
                logger.warning(f"Cloudflare Workers AI LLM API call failed: {e}. Using deterministic synthesis engine.")

        # Deterministic analytical synthesis fallback
        lines = [
            f"### Analysis & Answer (Based on Context Retrieval)",
            f"*(Generated under active System Prompt configuration)*\n",
            f"Based on the analyzed document sections matching **'{query}'**:\n",
        ]
        
        # Summarize retrieved snippets
        snippets = context.split("\n\n")
        for s in snippets:
            if s.startswith("[Source"):
                header = s.split("\n")[0]
                body = "\n".join(s.split("\n")[1:])
                lines.append(f"**From {header}:**")
                lines.append(f"> {body[:350]}...\n")

        lines.append("\n*Note: Configure Cloudflare Workers AI API Token in sidebar/env for full generative chat capabilities.*")
        return "\n".join(lines)
