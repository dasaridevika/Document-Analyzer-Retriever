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

class ExtractiveQASynthesizer:
    """
    Zero-Dependency Extractive Natural Language QA Generator.
    Synthesizes fluent, human-like answers when external LLM APIs are offline.
    """

    @staticmethod
    def classify_question_type(query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["how many", "exact number", "total", "count", "number of", "how much"]):
            return "COUNT_QUANTITY"
        if any(w in q for w in ["types", "categories", "kinds", "classification", "methods", "varieties"]):
            return "TYPES_LIST"
        if any(w in q for w in ["what is", "define", "definition of", "meaning of", "explain what"]):
            return "DEFINITION"
        if any(w in q for w in ["summarize", "summary", "overview", "what is it all about", "what is this about"]):
            return "SUMMARY"
        return "GENERAL"

    @staticmethod
    def synthesize_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        q_type = ExtractiveQASynthesizer.classify_question_type(query)
        q_words = set(re.findall(r'\w+', query.lower()))
        stop_words = {"what", "is", "the", "how", "many", "of", "to", "are", "there", "in", "for", "a", "an", "and", "or", "give", "me", "exact", "number"}
        keywords = q_words - stop_words

        all_facts = []
        seen_texts = set()

        for chunk in retrieved_chunks:
            page = chunk["metadata"].get("page_number", "?")
            raw_text = chunk.get("raw_content", chunk["text"])
            clean_text = re.sub(r'^\[Document:.*?\| Page \d+\]\n', '', raw_text).strip()
            
            lines = [l.strip() for l in clean_text.splitlines() if l.strip()]
            for line in lines:
                if len(line) > 10 and line not in seen_texts:
                    seen_texts.add(line)
                    all_facts.append((page, line))

        if not all_facts:
            return f"I searched the document for **\"{query}\"**, but could not find relevant details."

        if q_type == "COUNT_QUANTITY":
            numeric_lines = []
            for page, fact in all_facts:
                if re.search(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|levels|grids|total)\b', fact, re.IGNORECASE):
                    highlighted = re.sub(r'\b(\d+\s*(?:levels|grids|x\d+|modes|types|games|%)?)\b', r'**\1**', fact, flags=re.IGNORECASE)
                    numeric_lines.append(f"• {highlighted} [Page {page}]")
            
            if numeric_lines:
                return f"### Exact Quantitative Details for **\"{query}\"**:\n\n" + "\n".join(numeric_lines[:6])

        elif q_type == "TYPES_LIST":
            type_lines = []
            for page, fact in all_facts:
                fact_words = set(re.findall(r'\w+', fact.lower()))
                if any(w in fact.lower() for w in ["category", "type", "method", "class", "1.", "2.", "3.", "•", "-"]) or (keywords and (keywords & fact_words)):
                    type_lines.append(f"• {fact} [Page {page}]")
            if type_lines:
                return f"### Types & Classification for **\"{query}\"**:\n\n" + "\n".join(type_lines[:8])

        elif q_type == "DEFINITION":
            def_lines = []
            for page, fact in all_facts:
                fact_words = set(re.findall(r'\w+', fact.lower()))
                if keywords and (keywords & fact_words):
                    def_lines.append(f"• {fact} [Page {page}]")
            if def_lines:
                return f"### Definition & Explanation for **\"{query}\"**:\n\n" + "\n".join(def_lines[:6])

        elif q_type == "SUMMARY":
            summary_lines = [f"• {fact} [Page {page}]" for page, fact in all_facts[:8]]
            return f"### Executive Summary & Content Overview\n\n" + "\n".join(summary_lines)

        # General Synthesis
        matched_lines = []
        for page, fact in all_facts:
            fact_words = set(re.findall(r'\w+', fact.lower()))
            if keywords and (keywords & fact_words):
                matched_lines.append(f"• {fact} [Page {page}]")

        if matched_lines:
            return f"Based on your document, here are the extracted details for **\"{query}\"**:\n\n" + "\n".join(matched_lines[:8])

        fallback_lines = [f"• {fact} [Page {page}]" for page, fact in all_facts[:6]]
        return f"Here are the relevant sections for **\"{query}\"**:\n\n" + "\n".join(fallback_lines)


class RAGEngine:
    """
    Production-Grade Conversational Intent-Driven RAG Engine:
    - Multi-Turn Follow-Up Query Contextualization
    - Quantitative & Numeric Intent Expansion
    - Extractive Local QA Synthesizer (Zero-Dependency Fallback)
    - Relative Score Filtering & Heading-Body Context Stitching
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
        
        is_follow_up = len(query.split()) <= 6 or any(p in query.lower() for p in ["exact number", "give me", "how many", "types", "more details", "which one", "why"])
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
            "types": ["categories", "classification", "kinds", "varieties", "forms", "series", "shunt", "methods"],
            "types of": ["categories", "classification", "kinds of", "series compensation", "shunt compensation", "reactors", "capacitors"],
            "how many types": ["types of", "categories", "classification", "kinds", "series", "shunt", "list of types"],
            "hvdc": ["high voltage direct current", "hvdc transmission", "power transmission"],
            "facts": ["flexible ac transmission systems", "static shunt compensator"],
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
            expanded_str = query + " " + " ".join(added_terms[:10])
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
        seen_texts = set()

        for c in all_retrieved:
            raw_t = c.get("raw_content", c["text"])
            clean_t = re.sub(r'^\[Document:.*?\| Page \d+\]\n', '', raw_t).strip()
            if len(clean_t) > 35 and clean_t not in seen_texts:
                seen_texts.add(clean_t)
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

        # 4. LLM / Extractive Synthesis
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
CRITICAL MANDATE:
Answer ONLY the specific query asked by the user: "{query}".

RULES:
1. Focus SPECIFICALLY and ONLY on answering "{query}".
2. If asking for types, categories, or numbers, list and explain each clearly using bold text and bullet points.
3. Do NOT repeat identical text blocks.
4. Cite page numbers naturally like [Page X]."""

        messages = [
            {"role": "system", "content": f"{system_instruction}\n\nDOCUMENT EXCERPTS:\n{context}"},
            {"role": "user", "content": f"Based strictly on the DOCUMENT EXCERPTS above, write a direct, highly accurate answer focusing ONLY on:\n\n\"{query}\""}
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

        # Priority 3: Extractive Natural Language QA Synthesizer (Zero-Dependency Local QA Generator)
        return ExtractiveQASynthesizer.synthesize_answer(query=query, retrieved_chunks=retrieved_chunks)
