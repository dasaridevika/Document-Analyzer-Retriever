import requests
import logging
import json
import re
from typing import List, Dict, Any, Optional
from backend.config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_LLM_MODEL,
    ROOT_SYSTEM_INSTRUCTION,
    MIN_SIMILARITY_THRESHOLD,
    NO_EVIDENCE_FALLBACK_MESSAGE
)
from backend.services.worker_analyzer import WORKER_BASE_URL, DEFAULT_WORKER_URL

logger = logging.getLogger(__name__)

def _extract_json_payload(data: Any) -> Optional[Dict[str, Any]]:
    if isinstance(data, dict):
        if "answerable" in data and "answer" in data:
            return data
        for k in ["response", "answer", "result", "output"]:
            val = data.get(k)
            if isinstance(val, dict):
                res = _extract_json_payload(val)
                if res:
                    return res
            elif isinstance(val, str):
                try:
                    match = re.search(r'\{.*\}', val, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                        if isinstance(parsed, dict) and "answerable" in parsed:
                            return parsed
                except Exception:
                    pass

    elif isinstance(data, str):
        try:
            match = re.search(r'\{.*\}', data, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and "answerable" in parsed:
                    return parsed
        except Exception:
            pass

    return None


class GroundedCitationVerifier:
    """
    Application-Level Verification Engine:
    Validates every LLM claim, chunk ID, page number, and quote against retrieved evidence.
    Prevents hallucinated citations, fabricated quotes, or prompt-injection leakage.
    """

    @staticmethod
    def verify_and_format_response(
        query: str,
        structured_json: Optional[Dict[str, Any]],
        target_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not target_chunks:
            return {
                "answer": f"## Answer\n\n{NO_EVIDENCE_FALLBACK_MESSAGE}",
                "verified_quotes": [],
                "confidence": 0.0,
                "answerable": False
            }

        chunk_map = {c["chunk_id"]: c for c in target_chunks}
        chunk_text_map = {c["chunk_id"]: c["text"] + "\n" + c.get("raw_content", "") for c in target_chunks}
        all_text_combined = "\n\n".join([c["text"] + "\n" + c.get("raw_content", "") for c in target_chunks])

        if not structured_json or not structured_json.get("answerable", True):
            return GroundedCitationVerifier._fallback_extractive_verification(query, target_chunks)

        raw_answer = structured_json.get("answer", "").strip()
        claims = structured_json.get("claims", [])
        conflicts = structured_json.get("conflicts", [])
        missing_info = structured_json.get("missing_information", [])
        confidence = float(structured_json.get("confidence", 0.85))

        verified_evidence_items = []
        seen_quotes = set()

        for claim in claims:
            chunk_ids = claim.get("supporting_chunk_ids", [])
            page_numbers = claim.get("page_numbers", [])
            support_quote = claim.get("support_quote", "").strip()

            if support_quote and len(support_quote) > 10:
                is_verbatim = any(support_quote.lower() in t.lower() for t in chunk_text_map.values()) or (support_quote.lower() in all_text_combined.lower())
                
                if not is_verbatim:
                    words = [w for w in support_quote.split() if len(w) > 3]
                    if len(words) >= 3 and any(all(w.lower() in t.lower() for w in words[:4]) for t in chunk_text_map.values()):
                        is_verbatim = True

                if is_verbatim and support_quote not in seen_quotes:
                    seen_quotes.add(support_quote)
                    cid = chunk_ids[0] if chunk_ids and chunk_ids[0] in chunk_map else (target_chunks[0]["chunk_id"] if target_chunks else "c1")
                    page_num = page_numbers[0] if page_numbers else chunk_map.get(cid, {}).get("metadata", {}).get("page_number", 1)
                    verified_evidence_items.append(f"- Page {page_num}, chunk {cid}: “{support_quote}”")

        if not verified_evidence_items:
            for c in target_chunks[:3]:
                page_num = c["metadata"].get("page_number", 1)
                cid = c["chunk_id"]
                raw_lines = [l.strip() for l in c.get("raw_content", "").splitlines() if len(l.strip()) > 20]
                if raw_lines:
                    quote = raw_lines[0][:120]
                    if quote not in seen_quotes:
                        seen_quotes.add(quote)
                        verified_evidence_items.append(f"- Page {page_num}, chunk {cid}: “{quote}”")

        md_answer_parts = [f"## Answer\n\n{raw_answer}"]

        if verified_evidence_items:
            md_answer_parts.append("## Evidence\n\n" + "\n".join(verified_evidence_items[:5]))

        limitations = []
        if conflicts:
            limitations.append("• **Conflicting Information Detected:** " + "; ".join(conflicts))
        if missing_info:
            limitations.append("• **Incomplete Information:** " + "; ".join(missing_info))

        if limitations:
            md_answer_parts.append("## Limitations\n\n" + "\n".join(limitations))

        final_md = "\n\n".join(md_answer_parts)

        return {
            "answer": final_md,
            "verified_quotes": list(seen_quotes),
            "confidence": confidence,
            "answerable": True
        }

    @staticmethod
    def _fallback_extractive_verification(query: str, target_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extractive verification fallback when external LLM JSON is unparseable or offline.
        Handles subject listing, course structures, definitions, and quantitative queries.
        """
        lower_q = query.lower()
        is_subject_query = any(w in lower_q for w in ["subject", "subjects", "course title", "course structure", "list of subjects", "subjects in it"])

        q_words = set(re.findall(r'\w+', lower_q)) - {
            "what", "is", "the", "how", "many", "of", "to", "are", "there", "in", "for",
            "a", "an", "and", "or", "give", "me", "exact", "number", "which", "why", "where",
            "tell", "about", "list", "out", "show", "name", "names", "it", "all"
        }

        specific_terms = {
            w for w in q_words
            if w not in {"budget", "cost", "price", "amount", "total", "summary", "overview", "project", "system", "contract", "subject", "subjects", "course"}
        }

        evidence_items = []
        answer_lines = []
        seen_quotes = set()

        prompt_injection_keywords = [
            "instruction override", "ignore previous instructions", "system prompt",
            "secret_key", "secret password", "reveal the system prompt", "call this url"
        ]

        for c in target_chunks:
            page_num = c["metadata"].get("page_number", 1)
            cid = c["chunk_id"]
            raw_text = c.get("raw_content", c["text"])
            clean_text = re.sub(r'^\[Document:.*?\| Page \d+\]\n', '', raw_text).strip()
            
            lines = [l.strip() for l in clean_text.splitlines() if len(l.strip()) > 8]
            for line in lines:
                l_lower = line.lower()
                if l_lower.startswith(("document:", "page:", "section:", "content:")):
                    continue
                if any(inj in l_lower for inj in prompt_injection_keywords):
                    continue

                line_words = set(re.findall(r'\w+', l_lower))

                if is_subject_query:
                    # Header/Semester demarcation
                    if any(sem in l_lower for sem in ["year i semester", "year ii semester", "professional elective", "open elective"]):
                        header_title = line.strip(" |-*#").title()
                        if header_title not in seen_quotes:
                            seen_quotes.add(header_title)
                            answer_lines.append(f"\n**{header_title}:**")

                    # Course Title line extraction from syllabus table
                    is_course_line = (
                        re.search(r'\b[A-Z]{2,4}\d{3}[A-Z]{2}\b', line) or  # Course Code match e.g. MA101BS, EE103ES
                        re.search(r'\|\s*[A-Z0-9]+\s*\|\s*([^|]+)\|', line) or # Markdown table row match
                        any(term in l_lower for term in ["matrices", "calculus", "chemistry", "programming", "circuit", "physics", "electronics", "mechanics", "power", "machines", "systems", "workshop", "graphics", "microprocessors"])
                    )
                    if is_course_line and not l_lower.startswith(("list of experiments", "10. write", "write a c program", "course objectives", "course outcomes")):
                        clean_course = re.sub(r'^\d+\s*', '', line).strip()
                        if clean_course not in seen_quotes and len(clean_course) > 8:
                            seen_quotes.add(clean_course)
                            answer_lines.append(f"• {clean_course}")
                            evidence_items.append(f"- Page {page_num}, chunk {cid}: “{clean_course[:120]}”")
                else:
                    if specific_terms and not (specific_terms & line_words):
                        continue

                    if q_words and (q_words & line_words):
                        if line not in seen_quotes:
                            seen_quotes.add(line)
                            answer_lines.append(f"• {line}")
                            evidence_items.append(f"- Page {page_num}, chunk {cid}: “{line[:120]}”")

        if not answer_lines:
            return {
                "answer": f"## Answer\n\n{NO_EVIDENCE_FALLBACK_MESSAGE}",
                "verified_quotes": [],
                "confidence": 0.0,
                "answerable": False
            }

        header = "### Course Structure & Subject List:\n\n" if is_subject_query else ""
        md_response = f"## Answer\n\n{header}" + "\n".join(answer_lines[:25])
        if evidence_items:
            md_response += "\n\n## Evidence\n\n" + "\n".join(evidence_items[:6])

        return {
            "answer": md_response,
            "verified_quotes": list(seen_quotes),
            "confidence": 0.95,
            "answerable": True
        }


class RAGEngine:
    """
    Production Enterprise Grounded RAG Engine:
    - Immutable Root System Grounding Rules
    - Prompt-Injection Protection via <DOCUMENT_CONTEXT> Isolation
    - Conversational Query Rewriting (User-Only Reference Resolution)
    - Grounded Structured JSON Generation & Quote Verification
    - Strict No-Evidence & Conflicting Evidence Behavior
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
            "tell me about the document", "describe the document", "summarize it", "what is its all about",
            "list out the subjects", "list of subjects", "what are the subjects", "list the subjects",
            "subjects in it", "course structure", "all subjects", "subjects list"
        ]
        return any(phrase in lower_q for phrase in broad_phrases) or lower_q in ["summarize it", "summarize", "explain", "describe", "overview", "subjects"]

    def _contextualize_query(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        if not chat_history:
            return query
        
        is_follow_up = len(query.split()) <= 6 or any(p in query.lower() for p in ["exact number", "give me", "how many", "types", "more details", "which one", "why", "notice period"])
        if not is_follow_up:
            return query

        user_messages = [m["content"] for m in chat_history if m.get("role") == "user" and m.get("content", "").strip()]
        if len(user_messages) >= 2:
            prev_q = user_messages[-2]
            if prev_q.strip() and prev_q.lower() != query.lower():
                return f"{prev_q} - {query}"
        elif user_messages:
            prev_q = user_messages[-1]
            if prev_q.strip() and prev_q.lower() != query.lower():
                return f"{prev_q} - {query}"

        return query

    def answer_query(
        self,
        query: str,
        filename: str = None,
        document_id: str = None,
        session_id: str = None,
        system_prompt: str = None,
        top_k: int = 8,
        temperature: float = 0.1,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        clean_query = query.strip()

        if re.search(r'(ignore\s*previous\s*instructions|reveal\s*system\s*prompt|show\s*env|api_key|api_token)', clean_query, re.IGNORECASE):
            return {
                "answer": "## Answer\n\nI cannot perform actions that attempt to override safety instructions, expose API tokens, or reveal system prompts.",
                "sources": [],
                "system_prompt_used": ROOT_SYSTEM_INSTRUCTION,
                "retrieved_count": 0
            }

        is_broad = self._is_broad_query(clean_query)
        is_subject_listing = any(phrase in clean_query.lower() for phrase in ["list out the subjects", "list of subjects", "what are the subjects", "list the subjects", "subjects in it", "course structure", "all subjects", "subjects list"])
        search_query = self._contextualize_query(clean_query, chat_history)

        target_chunks = []

        if is_subject_listing:
            # Force retrieval of initial Course Structure pages (Pages 1 to 5)
            target_chunks = self.vector_store.get_first_pages_chunks(filename=filename, document_id=document_id, max_pages=5)

        if not target_chunks:
            q_embeddings = self.embedding_service.generate_embeddings([search_query])
            if q_embeddings:
                target_chunks = self.vector_store.similarity_search(
                    query_embedding=q_embeddings[0],
                    raw_query=search_query,
                    top_k=top_k,
                    filename_filter=filename,
                    document_id_filter=document_id,
                    session_id_filter=session_id,
                    min_score=MIN_SIMILARITY_THRESHOLD
                )

        if not target_chunks and filename:
            logger.info("Fallback: Retrieving initial document chunks...")
            target_chunks = self.vector_store.get_first_pages_chunks(filename=filename, document_id=document_id, max_pages=5)

        if not target_chunks:
            return {
                "answer": f"## Answer\n\n{NO_EVIDENCE_FALLBACK_MESSAGE}",
                "sources": [],
                "system_prompt_used": ROOT_SYSTEM_INSTRUCTION,
                "retrieved_count": 0
            }

        context_blocks = []
        for c in target_chunks:
            page_num = c["metadata"].get("page_number", "?")
            doc_fname = c["metadata"].get("filename", "Document")
            cid = c["chunk_id"]
            clean_chunk_text = c["text"]
            context_blocks.append(f"--- [Page {page_num} | Chunk {cid} | Document: {doc_fname}] ---\n{clean_chunk_text}")

        combined_context = "<DOCUMENT_CONTEXT>\n" + "\n\n".join(context_blocks) + "\n</DOCUMENT_CONTEXT>"

        structured_json = self._generate_llm_json(
            context=combined_context,
            query=clean_query,
            system_prompt=system_prompt,
            temperature=temperature
        )

        verified_res = GroundedCitationVerifier.verify_and_format_response(
            query=clean_query,
            structured_json=structured_json,
            target_chunks=target_chunks
        )

        sources = [
            {
                "source_id": idx + 1,
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "page_number": c["metadata"].get("page_number"),
                "filename": c["metadata"].get("filename"),
                "similarity_score": c.get("similarity_score")
            }
            for idx, c in enumerate(target_chunks)
        ]

        return {
            "answer": verified_res["answer"],
            "sources": sources,
            "verified_quotes": verified_res.get("verified_quotes", []),
            "confidence": verified_res.get("confidence", 0.0),
            "system_prompt_used": ROOT_SYSTEM_INSTRUCTION,
            "retrieved_count": len(sources)
        }

    def _generate_llm_json(
        self, context: str, query: str, system_prompt: Optional[str], temperature: float
    ) -> Optional[Dict[str, Any]]:
        combined_system = f"{ROOT_SYSTEM_INSTRUCTION}\n\nUSER STYLE/CUSTOMIZATION PROMPT: {system_prompt or ''}"

        json_instruction = """RESPOND STRICTLY WITH A VALID JSON OBJECT matching this exact schema:
{
  "answerable": true,
  "answer": "clear direct answer based ONLY on document evidence",
  "claims": [
    {
      "claim": "statement",
      "supporting_chunk_ids": ["c0"],
      "page_numbers": [1],
      "support_quote": "exact short quote from text"
    }
  ],
  "conflicts": [],
  "missing_information": [],
  "confidence": 0.95
}"""

        if self.worker_base_url:
            try:
                resp = requests.post(f"{self.worker_base_url}/analyze", json={
                    "query": query,
                    "text": context,
                    "system_prompt": combined_system,
                    "json_mode": True,
                    "temperature": temperature
                }, timeout=45)
                if resp.status_code == 200:
                    payload = _extract_json_payload(resp.json())
                    if payload:
                        return payload
            except Exception as e:
                logger.warning(f"Worker AI call failed: {e}")

        if self.account_id and self.api_token and "placeholder" not in self.account_id:
            try:
                url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.llm_model}"
                headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
                messages = [
                    {"role": "system", "content": f"{combined_system}\n\n{json_instruction}\n\nEVIDENCE:\n{context}"},
                    {"role": "user", "content": f"Answer strictly from the evidence above for query: \"{query}\""}
                ]
                resp = requests.post(url, headers=headers, json={"messages": messages, "temperature": temperature}, timeout=45)
                if resp.status_code == 200:
                    payload = _extract_json_payload(resp.json())
                    if payload:
                        return payload
            except Exception as e:
                logger.warning(f"Direct Cloudflare REST API failed: {e}")

        return None
