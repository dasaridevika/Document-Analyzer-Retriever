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
    - Broad & Specific Query Intent Classifier
    - Hybrid Search (BGE Dense + BM25 Lexical) via Reciprocal Rank Fusion
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
        """
        Detects if the user is asking for a summary, overview, or broad document explanation.
        """
        lower_q = query.lower().strip("?!., ")
        broad_phrases = [
            "summarize", "summary", "overview", "what is it all about", "what is it about",
            "what is this about", "explain this", "tell me about this", "explain the document",
            "what is this document", "what does this document contain", "table of contents",
            "full document", "executive summary", "main topics", "key points", "what is this",
            "tell me about the document", "describe the document", "summarize it"
        ]
        return any(phrase in lower_q for phrase in broad_phrases) or lower_q in ["summarize it", "summarize", "explain", "describe", "overview"]

    def _expand_query_intent(self, query: str) -> List[str]:
        expanded = [query]
        lower_q = query.lower()

        synonym_map = {
            "cost": ["price", "pricing", "fee", "rate", "charge", "payment", "amount"],
            "penalty": ["fine", "fee", "charge", "delayed", "late", "rejection", "sanction"],
            "fee": ["cost", "price", "pricing", "rate", "charge", "due", "fine"],
            "requirement": ["criteria", "condition", "prerequisite", "rule", "specification"],
            "how to": ["steps", "procedure", "method", "instructions", "guide"],
            "due date": ["deadline", "timeline", "due date", "expiration", "period"],
            "contact": ["email", "phone", "address", "support", "helpdesk"]
        }

        added_terms = []
        for key, terms in synonym_map.items():
            if key in lower_q:
                added_terms.extend(terms)

        if added_terms:
            expanded_str = query + " " + " ".join(added_terms[:6])
            expanded.append(expanded_str)

        return expanded

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
        queries_to_embed = self._expand_query_intent(clean_query)

        # 1. Retrieval
        all_retrieved = []
        seen_ids = set()

        for q_text in queries_to_embed:
            q_embeddings = self.embedding_service.generate_embeddings([q_text])
            if q_embeddings:
                chunks = self.vector_store.similarity_search(
                    query_embedding=q_embeddings[0],
                    raw_query=clean_query,
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

        # 2. Context Selection
        if is_broad:
            distributed_chunks = self.vector_store.get_distributed_chunks(filename=filename, count=8)
            for dc in distributed_chunks:
                if dc["chunk_id"] not in seen_ids:
                    all_retrieved.append(dc)
                    seen_ids.add(dc["chunk_id"])

            target_chunks = all_retrieved[:8]
        else:
            target_chunks = all_retrieved[:max(5, top_k)]

        if not target_chunks and filename:
            logger.info("Fallback: Retrieving distributed document chunks...")
            target_chunks = self.vector_store.get_distributed_chunks(filename=filename, count=6)

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
            clean_chunk_text = re.sub(r'^\[Document:.*?\| Page \d+\]\n', '', chunk['text'])
            context_blocks.append(f"--- [Page {page_num} | Document: {doc_fname}] ---\n{clean_chunk_text}")
        combined_context = "\n\n".join(context_blocks)

        # 4. LLM Synthesis
        raw_answer = self._generate_detailed_llm_response(
            system_prompt=effective_system_prompt,
            context=combined_context,
            query=clean_query,
            is_broad=is_broad,
            retrieved_chunks=target_chunks,
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
Your goal is to answer the user's intent: "{query}" based strictly on the provided DOCUMENT EXCERPTS.

RULES:
1. Understand what the user wants to know and extract ALL exact definitions, numbers, formulas, rules, conditions, and steps matching their query intent.
2. Explain the answer thoroughly in natural, clear prose using paragraphs, bullet points, and bold text.
3. Cite page numbers naturally like [Page X] for every fact or figure stated."""

        messages = [
            {"role": "system", "content": f"{system_instruction}\n\nDOCUMENT EXCERPTS:\n{context}"},
            {"role": "user", "content": f"Based strictly on the DOCUMENT EXCERPTS above, write a natural, comprehensive response answering:\n\n\"{query}\""}
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
        self, system_prompt: str, context: str, query: str, is_broad: bool, retrieved_chunks: List[Dict[str, Any]], temperature: float
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

        # Priority 3: High-Quality Fallback Natural Language Synthesizer
        # Cleans raw chunks, removes header junk, and constructs a readable summary
        pages_seen = set()
        clean_facts = []

        for c in retrieved_chunks:
            page = c["metadata"].get("page_number", 1)
            pages_seen.add(page)
            raw_text = c.get("raw_content", c["text"])
            # Remove metadata header lines
            lines = [l.strip() for l in raw_text.splitlines() if l.strip() and not l.strip().startswith("[Document:")]
            
            for line in lines:
                if len(line) > 10 and line not in clean_facts:
                    clean_facts.append(f"{line} [Page {page}]")

        if is_broad or any(w in query.lower() for w in ["summary", "summarize", "about", "overview"]):
            intro = f"### Document Overview & Summary\n\nThis document covers key details across page(s) {', '.join(str(p) for p in sorted(pages_seen))}. Here is an organized breakdown of the content:\n\n"
            bullet_points = "\n".join([f"• {fact}" for fact in clean_facts[:10]])
            return intro + bullet_points

        intro = f"Based on the relevant sections of your document, here are the extracted details answering **\"{query}\"**:\n\n"
        bullet_points = "\n".join([f"• {fact}" for fact in clean_facts[:8]])
        return intro + bullet_points
