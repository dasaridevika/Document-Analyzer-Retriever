import requests
import logging
import re
from typing import List, Dict, Any, Optional
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
        return s if len(s) > 5 else ""
    if isinstance(data, dict):
        for k in ["response", "answer", "output", "text"]:
            val = data.get(k)
            if isinstance(val, str) and len(val.strip()) > 5:
                return val.strip()
            elif isinstance(val, dict):
                sub = _extract_text_from_llm_payload(val)
                if sub:
                    return sub
        res = data.get("result")
        if res:
            return _extract_text_from_llm_payload(res)
    return ""

class RAGEngine:
    """
    Production-Grade Intent-Driven RAG Engine:
    - Acronym & Technical Concept Expansion
    - Noise-Stripped Chunk Verification
    - Multi-Turn Follow-Up Query Contextualization
    - Natural Language Synthesizer & Page Citation Engine
    """

    def __init__(self, embedding_service, vector_store):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.account_id = CLOUDFLARE_ACCOUNT_ID
        self.api_token = CLOUDFLARE_API_TOKEN
        self.llm_model = CLOUDFLARE_LLM_MODEL or "@cf/meta/llama-3.1-8b-instruct"
        self.worker_base_url = WORKER_BASE_URL or DEFAULT_WORKER_URL

    def _is_broad_query(self, query: str) -> bool:
        lower_q = query.lower().strip("?!., ")
        broad_phrases = [
            "summarize", "summary", "overview", "what is it all about", "what is it about",
            "what is this about", "explain this", "tell me about this", "explain the document",
            "what is this document", "what does this document contain", "table of contents",
            "full document", "executive summary", "main topics", "key points", "what is this",
            "tell me about the document", "describe the document", "summarize it", "what is its all about"
        ]
        return any(phrase in lower_q for phrase in broad_phrases) or lower_q in ["summarize it", "summarize", "explain", "describe", "overview"]

    def _contextualize_query(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        if not chat_history:
            return query
        
        is_follow_up = len(query.split()) <= 5 or any(p in query.lower() for p in ["exact number", "give me", "how many", "more details", "which one", "why"])
        if not is_follow_up:
            return query

        user_messages = [m["content"] for m in chat_history if m.get("role") == "user" and m["content"].strip()]
        if len(user_messages) >= 2:
            prev_q = user_messages[-2]
            if prev_q.strip() and prev_q.lower() != query.lower():
                return f"{prev_q} - {query}"
        elif user_messages:
            prev_q = user_messages[-1]
            if prev_q.strip() and prev_q.lower() != query.lower():
                return f"{prev_q} - {query}"

        return query

    def _expand_query_intent(self, query: str) -> List[str]:
        expanded = [query]
        lower_q = query.lower()

        synonym_map = {
            "hvdc": ["high voltage direct current", "hvdc transmission", "shunt compensation", "facts", "power transmission"],
            "facts": ["flexible ac transmission systems", "static shunt compensator", "statcom", "svc"],
            "define": ["definition", "concept", "meaning", "what is", "principle", "explanation"],
            "cost": ["price", "pricing", "fee", "rate", "charge", "payment", "amount"],
            "penalty": ["fine", "fee", "charge", "delayed", "late", "rejection", "sanction"],
            "fee": ["cost", "price", "pricing", "rate", "charge", "due", "fine"],
            "requirement": ["criteria", "condition", "prerequisite", "rule", "specification"],
            "how to": ["steps", "procedure", "method", "instructions", "guide"],
            "due date": ["deadline", "timeline", "due date", "expiration", "period"],
            "how many": ["total", "count", "number", "quantity", "levels", "modes", "grids", "games", "amount"],
            "exact number": ["total number", "count", "exact count", "levels", "quantity", "number of", "total"],
            "contact": ["email", "phone", "address", "support", "helpdesk"]
        }

        added_terms = []
        for key, terms in synonym_map.items():
            if key in lower_q:
                added_terms.extend(terms)

        if added_terms:
            expanded_str = query + " " + " ".join(added_terms[:8])
            expanded.append(expanded_str)

        return expanded

    def answer_query(
        self,
        query: str,
        filename: str = None,
        system_prompt: str = None,
        top_k: int = 8,
        temperature: float = 0.1,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        effective_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        clean_query = query.strip()

        # Handle Greetings
        lower_q = clean_query.lower().strip("?!.,")
        if lower_q in ["hi", "hello", "hey", "greetings", "who are you", "what can you do"]:
            doc_name = f"'{filename}'" if filename else "your uploaded documents"
            return {
                "answer": f"Hello! Ask me any detailed question about {doc_name}, or ask me to summarize it, and I will analyze the document to give you a clear, comprehensive answer.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        is_broad = self._is_broad_query(clean_query)
        search_query = self._contextualize_query(clean_query, chat_history)
        queries_to_embed = self._expand_query_intent(search_query)

        # 1. Retrieval
        all_retrieved = []
        seen_ids = set()

        for q_text in queries_to_embed:
            q_embeddings = self.embedding_service.generate_embeddings([q_text])
            if q_embeddings:
                chunks = self.vector_store.similarity_search(
                    query_embedding=q_embeddings[0],
                    raw_query=search_query,
                    top_k=top_k,
                    filename_filter=filename,
                    min_score=0.10 if is_broad else 0.15
                )
                for c in chunks:
                    cid = c["chunk_id"]
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_retrieved.append(c)

        all_retrieved.sort(key=lambda x: x.get("rrf_score", x.get("similarity_score", 0)), reverse=True)

        # 2. Score Filtering & Empty Noise Chunk Filter
        valid_retrieved = []
        for c in all_retrieved:
            raw_t = c.get("raw_content", c["text"])
            clean_t = re.sub(r'^\[Document:.*?\| Page \d+\]\n', '', raw_t).strip()
            # Ignore empty or header-only noise chunks
            if len(clean_t) > 35:
                valid_retrieved.append(c)

        if not is_broad and valid_retrieved:
            top_score = valid_retrieved[0].get("similarity_score", 0)
            threshold = max(0.15, top_score * 0.70)
            filtered_chunks = [c for c in valid_retrieved if c.get("similarity_score", 0) >= threshold]
            target_chunks = filtered_chunks[:max(1, min(top_k, 4))]
        elif is_broad:
            distributed_chunks = self.vector_store.get_distributed_chunks(filename=filename, count=8)
            for dc in distributed_chunks:
                if dc["chunk_id"] not in seen_ids:
                    valid_retrieved.append(dc)
                    seen_ids.add(dc["chunk_id"])
            target_chunks = valid_retrieved[:8]
        else:
            target_chunks = valid_retrieved[:top_k]

        if not target_chunks and filename:
            logger.info("Fallback: Retrieving distributed document chunks...")
            target_chunks = self.vector_store.get_distributed_chunks(filename=filename, count=5)

        if not target_chunks:
            return {
                "answer": "I analyzed the document context, but could not find relevant content matching your query.",
                "sources": [],
                "system_prompt_used": effective_system_prompt,
                "retrieved_count": 0
            }

        # 3. Format Context
        context_blocks = []
        for i, chunk in enumerate(target_chunks):
            page_num = chunk["metadata"].get("page_number", "?")
            doc_fname = chunk["metadata"].get("filename", "Document")
            clean_chunk_text = re.sub(r'^\[Document:.*?\| Page \d+\]\n', '', chunk['text']).strip()
            
            # Stitch adjacent chunk if text is short (< 150 chars)
            if len(clean_chunk_text) < 150:
                c_idx = chunk["metadata"].get("chunk_index", 0)
                adjacent = [
                    self.vector_store.documents_store[j] for j, meta in enumerate(self.vector_store.metadata_store)
                    if meta.get("filename") == doc_fname and meta.get("page_number") == page_num and meta.get("chunk_index") == c_idx + 1
                ]
                if adjacent:
                    clean_adj = re.sub(r'^\[Document:.*?\| Page \d+\]\n', '', adjacent[0]).strip()
                    clean_chunk_text += "\n\n" + clean_adj

            context_blocks.append(f"--- [Page {page_num} | Document: {doc_fname}] ---\n{clean_chunk_text}")

        combined_context = "\n\n".join(context_blocks)

        # 4. LLM Synthesis
        raw_answer = self._generate_detailed_llm_response(
            system_prompt=effective_system_prompt,
            context=combined_context,
            query=clean_query,
            is_broad=is_broad,
            retrieved_chunks=target_chunks,
            temperature=temperature,
            chat_history=chat_history
        )

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
            "answer": raw_answer.strip(),
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
                resp = requests.post(target_url, json=payload, timeout=50)
                if resp.status_code == 200:
                    extracted_text = _extract_text_from_llm_payload(resp.json())
                    if extracted_text and len(extracted_text) > 15:
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
            system_instruction = f"""You are a Lead AI Document Analyst.
The user wants a clear, fluent summary and explanation of the document: "{query}".

PRODUCE A WELL-WRITTEN, HIGHLY READABLE OVERVIEW WITH THESE SECTIONS:
1. **Document Subject & Overview**: Explain what this document is about in simple, natural paragraphs.
2. **Key Positions, Figures & Important Details**: Highlight specific roles, dates, numbers, requirements, or locations mentioned.
3. **Summary & Key Takeaways**: Provide a concise conclusion.

RULES:
- Do NOT output raw chunk headers or repetitive brackets. Write fluent, natural English.
- Cite page numbers naturally like [Page 1], [Page 2]."""
        else:
            system_instruction = f"""You are a Master AI Document Assistant.
CRITICAL INSTRUCTION:
Your goal is to answer the user's question: "{query}" based strictly on the provided DOCUMENT EXCERPTS.

RULES:
1. Extract and explain the COMPLETE answer to "{query}" using clear paragraphs, bold terms, and exact technical explanations.
2. If acronyms like HVDC are asked, explain High Voltage Direct Current in relation to the document context (e.g. Static Shunt Compensation / FACTS).
3. Do NOT output raw chunk headers or unparsed text lists.
4. Cite page numbers naturally like [Page X] for every fact stated."""

        messages = [
            {"role": "system", "content": f"{system_instruction}\n\nDOCUMENT EXCERPTS:\n{context}"},
            {"role": "user", "content": f"Based strictly on the DOCUMENT EXCERPTS above, write a detailed, complete answer for:\n\n\"{query}\""}
        ]

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 3000
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=45)
        if resp.status_code != 200:
            raise RuntimeError(f"Cloudflare REST API HTTP {resp.status_code}: {resp.text}")

        extracted_text = _extract_text_from_llm_payload(resp.json())
        if extracted_text and len(extracted_text) > 15:
            return extracted_text

        raise RuntimeError("Cloudflare REST API response did not contain valid text.")

    def _generate_detailed_llm_response(
        self, system_prompt: str, context: str, query: str, is_broad: bool, retrieved_chunks: List[Dict[str, Any]], temperature: float, chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        # Priority 1: Cloudflare Worker AI Endpoint
        if self.worker_base_url:
            try:
                ans = self._generate_cloudflare_worker_llm(system_prompt, context, query, is_broad, temperature)
                if ans and len(ans) > 20:
                    return ans
            except Exception as e:
                logger.warning(f"Cloudflare Worker AI call failed: {e}")

        # Priority 2: Direct Cloudflare REST API
        if self.account_id and self.api_token:
            try:
                ans = self._generate_cloudflare_rest_llm(system_prompt, context, query, is_broad, temperature)
                if ans and len(ans) > 20:
                    return ans
            except Exception as e:
                logger.warning(f"Direct Cloudflare REST API LLM call failed: {e}")

        # Priority 3: Conversational Full-Paragraph Content Synthesizer Fallback with Noise Filtering
        top_chunks = retrieved_chunks[:3]
        response_sections = []

        for chunk in top_chunks:
            page = chunk["metadata"].get("page_number", "?")
            doc_fname = chunk["metadata"].get("filename", "")
            c_idx = chunk["metadata"].get("chunk_index", 0)
            
            raw_text = chunk.get("raw_content", chunk["text"])
            body_text = re.sub(r'^\[Document:.*?\| Page \d+\]\n', '', raw_text).strip()
            
            if len(body_text) < 150:
                adjacent = [
                    self.vector_store.documents_store[j] for j, meta in enumerate(self.vector_store.metadata_store)
                    if meta.get("filename") == doc_fname and meta.get("page_number") == page and meta.get("chunk_index") == c_idx + 1
                ]
                if adjacent:
                    clean_adj = re.sub(r'^\[Document:.*?\| Page \d+\]\n', '', adjacent[0]).strip()
                    body_text += "\n\n" + clean_adj

            if body_text and len(body_text) > 30:
                response_sections.append(f"**From Page {page}:**\n{body_text}")

        if response_sections:
            if is_broad:
                return f"### Document Overview & Content Summary\n\n" + "\n\n---\n\n".join(response_sections)
            
            # Acronym clarification helper for HVDC
            prefix = ""
            if "hvdc" in query.lower():
                prefix = "**HVDC** stands for **High Voltage Direct Current**. In the context of your document:\n\n"

            return prefix + f"Here are the key details extracted from your document for **\"{query}\"**:\n\n" + "\n\n---\n\n".join(response_sections)

        return f"I analyzed the document for **\"{query}\"**, but could not extract a detailed answer."
