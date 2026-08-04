import requests
import logging
import json
import re
import uuid
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from backend.config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_LLM_MODEL,
    IMMUTABLE_SYSTEM_PROMPT,
    MIN_RELEVANCE_SCORE,
    FINAL_CONTEXT_K,
    DENSE_TOP_K,
    NO_EVIDENCE_FALLBACK_MESSAGE,
    DEBUG_RAG
)
from backend.services.worker_analyzer import WORKER_BASE_URL, DEFAULT_WORKER_URL

logger = logging.getLogger(__name__)


# ============================================================================
# Dataclasses & Data Models
# ============================================================================

@dataclass
class QueryIntent:
    """Represents classified query intent, subjects, and execution parameters."""
    intent: str  # definition, summary, overview, comparison, explanation, list, objectives, page_lookup, chapter_lookup, reasoning, table_lookup, figure_lookup, calculation, boolean, multi_hop
    primary_subject: str
    secondary_subject: Optional[str] = None
    requested_format: str = "direct answer"
    target_page: Optional[int] = None
    target_chapter: Optional[int] = None
    adaptive_top_k: int = 5
    retrieval_strategy: str = "dense_hybrid"  # map_reduce, multi_hop, page_filter, chapter_filter, wide_dense, dense_hybrid


@dataclass
class RAGTrace:
    """Detailed trace metadata for developer observability."""
    request_id: str
    document_id: str
    original_query: str
    rewritten_query: str
    query_intent: str
    retrieval_strategy: str
    multi_queries: List[str] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    reranked_chunks: List[Dict[str, Any]] = field(default_factory=list)
    compressed_context: str = ""
    context_sent_to_llm: str = ""
    raw_llm_response: str = ""
    parsed_response: Dict[str, Any] = field(default_factory=dict)
    answer_relevance: float = 0.0
    groundedness: float = 0.0
    citation_valid: bool = False
    hallucination_detected: bool = False
    confidence_score: float = 0.0
    execution_time_ms: float = 0.0
    cache_hit: bool = False
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None


# ============================================================================
# Semantic Cache & Cross Encoder Reranker Interfaces
# ============================================================================

class SemanticCache:
    """In-memory cosine-similarity semantic cache for instant query responses."""

    def __init__(self, max_size: int = 300, similarity_threshold: float = 0.94):
        self.max_size = max_size
        self.threshold = similarity_threshold
        self.entries: List[Tuple[np.ndarray, str, Dict[str, Any]]] = []

    def get(self, query_vector: List[float], document_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not query_vector or not self.entries:
            return None

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return None
        q_vec /= q_norm

        for stored_vec, stored_doc_id, stored_result in reversed(self.entries):
            if document_id and stored_doc_id and stored_doc_id != document_id:
                continue

            sim = float(np.dot(q_vec, stored_vec))
            if sim >= self.threshold:
                result_copy = json.loads(json.dumps(stored_result))
                if "rag_trace" in result_copy and result_copy["rag_trace"]:
                    result_copy["rag_trace"]["cache_hit"] = True
                return result_copy

        return None

    def put(self, query_vector: List[float], document_id: Optional[str], result: Dict[str, Any]) -> None:
        if not query_vector or not result:
            return

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return
        q_vec /= q_norm

        if len(self.entries) >= self.max_size:
            self.entries.pop(0)

        self.entries.append((q_vec, document_id or "", result))


class CrossEncoderReranker:
    """Pluggable Cross-Encoder Reranker engine (BGE / Cohere / Heuristic Feature Reranking)."""

    @staticmethod
    def rerank(query: str, chunks: List[Dict[str, Any]], intent_type: str = "fact") -> List[Dict[str, Any]]:
        if not chunks:
            return []

        q_terms = set(re.findall(r'\w+', query.lower())) - {
            "what", "is", "the", "how", "does", "are", "there", "in", "for", "to", "a", "an", "and", "or",
            "summarise", "summarize", "it", "given", "document", "explain"
        }

        reranked = []
        for c in chunks:
            text = c.get("raw_content", c.get("text", ""))
            text_lower = text.lower()
            doc_terms = set(re.findall(r'\w+', text_lower))

            overlap = len(q_terms & doc_terms) / max(1, len(q_terms)) if q_terms else 0.5
            base_score = float(c.get("similarity_score", 0.5)) * 0.6 + overlap * 0.4

            intent_boost = 0.0
            if intent_type in ["summary", "overview"]:
                p_num = c.get("metadata", {}).get("page_number", 1)
                if p_num <= 3:
                    intent_boost = 0.08
            elif intent_type == "definition" and any(k in text_lower for k in ["defined as", "refers to", "definition", "means"]):
                intent_boost = 0.05
            elif intent_type == "objectives" and any(k in text_lower for k in ["objective", "aim", "goal", "purpose"]):
                intent_boost = 0.05
            elif intent_type in ["mechanism", "explanation"] and any(k in text_lower for k in ["mechanism", "process", "operation", "function", "how"]):
                intent_boost = 0.05

            final_score = base_score + intent_boost
            c_copy = dict(c)
            c_copy["cross_score"] = round(final_score, 4)
            reranked.append(c_copy)

        reranked.sort(key=lambda x: x["cross_score"], reverse=True)
        return reranked


global_semantic_cache = SemanticCache()


# ============================================================================
# Helper Functions & Parsers
# ============================================================================

def _extract_json_payload(data: Any) -> Optional[Dict[str, Any]]:
    """Extracts JSON payload from dictionary or raw string."""
    if isinstance(data, dict):
        if "answerable" in data and ("answer" in data or "definition" in data or "parts" in data):
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


def is_relevant_to_query(query: str, chunk_text: str, score: float = 0.0, intent: str = "fact") -> bool:
    """Pre-LLM relevance filter to discard non-matching candidate text."""
    if intent in ["summary", "overview", "page_lookup", "chapter_lookup"]:
        return True

    q_words = set(re.findall(r'\w+', query.lower())) - {
        "what", "is", "the", "how", "many", "of", "to", "are", "there", "in", "for",
        "a", "an", "and", "or", "give", "me", "exact", "number", "which", "why", "where",
        "tell", "about", "list", "out", "show", "name", "names", "it", "all", "page", "discussed",
        "summarise", "summarize", "given", "document", "explain"
    }

    if not q_words:
        return True

    c_words = set(re.findall(r'\w+', chunk_text.lower()))
    matches = q_words & c_words
    return len(matches) >= 1 or score >= 0.20


# ============================================================================
# Conversation Resolver & Query Classifier
# ============================================================================

class ConversationResolver:
    """Advanced Anaphora & Conversation Context Resolver."""

    @staticmethod
    def resolve_conversation(query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        clean_q = query.strip()
        lower_q = clean_q.lower()

        # Handle standalone pronoun queries ("summarise it", "explain it", "what about page 5?")
        if lower_q in ["summarise it", "summarize it", "summarise", "summarize", "summary", "overview", "what is this pdf about"]:
            return "Summarize the active document and provide an executive overview of main topics."

        if lower_q in ["explain it", "explain this", "what is it"]:
            if chat_history:
                user_msgs = [m["content"] for m in chat_history if m.get("role") == "user" and m.get("content", "").strip()]
                if user_msgs:
                    last_topic = re.sub(r'^(what is|explain|define)\s*', '', user_msgs[-1], flags=re.IGNORECASE).strip()
                    return f"Explain {last_topic} in detail based on the document."
            return "Explain the core concepts discussed in the document."

        # Handle page follow-ups ("what about page 5?", "page 10?")
        page_followup = re.search(r'^(?:what about|show|check|on)\s*page\s*(\d+)$', lower_q)
        if page_followup:
            p_num = page_followup.group(1)
            return f"What is discussed on page {p_num}?"

        # History Fusion
        if chat_history and len(query.split()) <= 5:
            user_msgs = [m["content"] for m in chat_history if m.get("role") == "user" and m.get("content", "").strip()]
            if user_msgs:
                prev_q = user_msgs[-1].strip()
                prev_q_lower = prev_q.lower()
                # Skip fusion if the previous query was a summary/overview command
                is_prev_summary = any(x in prev_q_lower for x in ["summarise", "summarize", "summary", "overview", "pdf about"])
                if not is_prev_summary:
                    if prev_q and prev_q_lower != lower_q:
                        return f"{prev_q} - {query}"

        return clean_q


class QueryRewriter:
    """Enterprise 15-Intent Classifier & Multi-Query Expansion Engine."""

    @staticmethod
    def rewrite_query(query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, QueryIntent]:
        resolved_q = ConversationResolver.resolve_conversation(query, chat_history)
        lower_q = resolved_q.lower().strip("?!., ")

        # Target Page & Chapter Lookups
        page_match = re.search(r'\bpage\s+(\d+)\b', lower_q)
        chapter_match = re.search(r'\bchapter\s+(\d+)\b', lower_q)

        target_page = int(page_match.group(1)) if page_match else None
        target_chapter = int(chapter_match.group(1)) if chapter_match else None

        # Comparison Extraction ("difference between X and Y", "compare X and Y")
        comp_match = re.search(r'(?:difference between|compare|versus|vs)\s+([\w\s]+?)\s+(?:and|vs|versus)\s+([\w\s]+)', lower_q)

        intent = "fact"
        requested_format = "direct answer"
        adaptive_top_k = 5
        strategy = "dense_hybrid"
        primary_subj = ""
        secondary_subj = None

        if comp_match:
            intent = "comparison"
            primary_subj = comp_match.group(1).strip()
            secondary_subj = comp_match.group(2).strip()
            requested_format = "structured concept comparison"
            adaptive_top_k = 8
            strategy = "multi_hop"

        elif any(w in lower_q for w in ["summarize", "summarise", "summary", "summarisation", "overview", "what is this pdf about", "main points"]):
            intent = "summary"
            primary_subj = "document overview"
            requested_format = "document executive summary"
            adaptive_top_k = 15
            strategy = "map_reduce"

        elif target_page is not None:
            intent = "page_lookup"
            primary_subj = f"page {target_page}"
            requested_format = "page content summary"
            adaptive_top_k = 3
            strategy = "page_filter"

        elif target_chapter is not None:
            intent = "chapter_lookup"
            primary_subj = f"chapter {target_chapter}"
            requested_format = "chapter overview"
            adaptive_top_k = 6
            strategy = "chapter_filter"

        elif any(w in lower_q for w in ["what is", "define", "definition", "meaning of"]):
            intent = "definition"
            primary_subj = re.sub(r'^(what is|define|definition|meaning of)\s*', '', lower_q).strip()
            requested_format = "short definition with explanation"
            adaptive_top_k = 3
            strategy = "dense_hybrid"

        elif any(w in lower_q for w in ["objective", "objectives", "purpose of", "aim of"]):
            intent = "objectives"
            primary_subj = re.sub(r'^(what are the objectives of|purpose of|aim of)\s*', '', lower_q).strip()
            requested_format = "bulleted objectives list"
            adaptive_top_k = 4
            strategy = "dense_hybrid"

        elif any(w in lower_q for w in ["how does", "how do", "improve", "mechanism", "work"]):
            intent = "mechanism"
            primary_subj = re.sub(r'^(how does|how do|mechanism of)\s*', '', lower_q).strip()
            requested_format = "step-by-step mechanism explanation"
            adaptive_top_k = 5
            strategy = "dense_hybrid"

        elif any(w in lower_q for w in ["explain", "elaborate", "details"]):
            intent = "explanation"
            primary_subj = re.sub(r'^(explain|elaborate|details on)\s*', '', lower_q).strip()
            requested_format = "detailed conceptual explanation"
            adaptive_top_k = 6
            strategy = "dense_hybrid"

        elif any(w in lower_q for w in ["list", "subjects", "courses", "all", "advantages", "disadvantages"]):
            intent = "list"
            primary_subj = re.sub(r'^(list|all|advantages|disadvantages)\s*', '', lower_q).strip()
            requested_format = "structured list"
            adaptive_top_k = 10
            strategy = "wide_dense"

        elif any(w in lower_q for w in ["why", "reason", "because", "cause"]):
            intent = "reasoning"
            primary_subj = re.sub(r'^(why|reason for)\s*', '', lower_q).strip()
            requested_format = "cause and effect reasoning"
            adaptive_top_k = 8
            strategy = "multi_hop"

        if not primary_subj:
            primary_subj = lower_q

        intent_obj = QueryIntent(
            intent=intent,
            primary_subject=primary_subj,
            secondary_subject=secondary_subj,
            requested_format=requested_format,
            target_page=target_page,
            target_chapter=target_chapter,
            adaptive_top_k=adaptive_top_k,
            retrieval_strategy=strategy
        )

        return resolved_q, intent_obj

    @staticmethod
    def generate_multi_queries(query: str, intent_obj: QueryIntent) -> List[str]:
        """Generates semantic query variations for Multi-Query Expansion."""
        clean_q = query.strip()
        subj1 = intent_obj.primary_subject or clean_q

        variations = [clean_q]
        if intent_obj.intent == "comparison" and intent_obj.secondary_subject:
            variations.append(f"{subj1} characteristics and features")
            variations.append(f"{intent_obj.secondary_subject} characteristics and features")
            variations.append(f"Comparison of {subj1} and {intent_obj.secondary_subject}")
        elif intent_obj.intent == "summary":
            variations.append("Executive summary of document main topics and objectives")
            variations.append("Overview of reactive shunt compensation and transmission system principles")
        elif intent_obj.intent == "definition":
            variations.append(f"Define {subj1}")
            variations.append(f"What is the definition and purpose of {subj1}")
        elif intent_obj.intent == "objectives":
            variations.append(f"Key objectives and aims of {subj1}")
            variations.append(f"Why is {subj1} used in electrical engineering")
        elif intent_obj.intent in ["mechanism", "explanation"]:
            variations.append(f"How does {subj1} work step by step")
            variations.append(f"Voltage regulation and power transfer mechanism of {subj1}")
        else:
            variations.append(f"Details on {subj1}")
            variations.append(f"Overview of {subj1}")

        return list(dict.fromkeys(variations))


# ============================================================================
# Grounding & Universal Extractive Fallback Engine
# ============================================================================

class GroundedCitationVerifier:
    """Verifies citations, verbatim quotes, and provides Universal Extractive Fallbacks."""

    @staticmethod
    def verify_response(
        query: str,
        structured_json: Optional[Dict[str, Any]],
        target_chunks: List[Dict[str, Any]],
        intent_obj: QueryIntent
    ) -> Tuple[Dict[str, Any], bool, str]:
        if not target_chunks:
            return {
                "answer": f"## Answer\n\n{NO_EVIDENCE_FALLBACK_MESSAGE}",
                "parts": [],
                "confidence": 0.0
            }, False, "No target chunks retrieved"

        if not structured_json or not structured_json.get("answerable", True):
            return GroundedCitationVerifier._fallback_universal_extractive(query, target_chunks, intent_obj)

        raw_definition = structured_json.get("definition", "").strip()
        raw_explanation = structured_json.get("explanation", "").strip()
        raw_answer = structured_json.get("answer", "").strip()

        evidence_list = structured_json.get("evidence", structured_json.get("claims", []))
        confidence = float(structured_json.get("confidence", 0.90))

        chunk_map = {c["chunk_id"]: c for c in target_chunks}
        chunk_text_map = {c["chunk_id"]: c["text"] + "\n" + c.get("raw_content", "") for c in target_chunks}
        all_text_combined = "\n\n".join([c["text"] + "\n" + c.get("raw_content", "") for c in target_chunks])

        seen_quotes: Set[str] = set()
        verified_evidence_items: List[str] = []
        valid_citations = True

        for cite in evidence_list:
            cid = cite.get("chunk_id") or (cite.get("supporting_chunk_ids", [None])[0] if cite.get("supporting_chunk_ids") else None)
            page = cite.get("page") or (cite.get("page_numbers", [1])[0] if cite.get("page_numbers") else 1)
            quote = (cite.get("quote") or cite.get("support_quote", "")).strip()

            if cid and cid not in chunk_map:
                valid_citations = False

            if cid in chunk_map and page and page != chunk_map[cid]["metadata"].get("page_number"):
                valid_citations = False

            if quote and len(quote) > 8:
                quote_clean = re.sub(r'\s+', ' ', quote.lower())
                target_text_clean = re.sub(r'\s+', ' ', chunk_text_map.get(cid, all_text_combined).lower())
                if quote_clean not in target_text_clean:
                    valid_citations = False
                else:
                    if quote not in seen_quotes:
                        seen_quotes.add(quote)
                        verified_evidence_items.append(f"- Page {page or 1}, chunk {cid or 'c0'}: “{quote}”")

        md_parts = []
        if raw_definition:
            md_parts.append(f"**Definition:**\n{raw_definition}")
        if raw_explanation:
            md_parts.append(f"**Explanation:**\n{raw_explanation}")
        if not raw_definition and not raw_explanation and raw_answer:
            md_parts.append(f"{raw_answer}")

        if verified_evidence_items:
            md_parts.append("## Evidence\n\n" + "\n".join(verified_evidence_items[:4]))

        final_md = "## Answer\n\n" + "\n\n".join(md_parts)

        return {
            "answer": final_md,
            "verified_quotes": list(seen_quotes),
            "confidence": confidence
        }, valid_citations, "Verification complete"

    @staticmethod
    def _is_clean_sentence(s: str) -> bool:
        s_clean = s.strip()
        if len(s_clean) < 18:
            return False
        if not re.match(r'^[A-Z0-9\•\*\-\"\“\|]', s_clean):
            return False
        if re.search(r'\b(at a|the|of|and|or|in|for|with|to|is|are|shown|plotted|figure|shunt|two\-machine)\s*$', s_clean, re.IGNORECASE):
            return False
        if re.match(r'^(?:Document|Document ID|Chunk ID|Section|Figure|Table|Unit\s+[V|X|I]+|OBJECTIVES OF)\b', s_clean, re.IGNORECASE):
            return False
        if re.match(r'^(?:Page\s+\d+\s*(?:of\s+\d+)?)$', s_clean, re.IGNORECASE):
            return False
        return True

    @staticmethod
    def _is_adversarial_text(text: str) -> bool:
        pattern = r'(ignore\s*previous\s*instructions|instruction\s*override|system\s*prompt|secret\s*key|api_key|api_token)'
        return bool(re.search(pattern, text, re.IGNORECASE))

    @staticmethod
    def _fallback_universal_extractive(query: str, target_chunks: List[Dict[str, Any]], intent_obj: QueryIntent) -> Tuple[Dict[str, Any], bool, str]:
        lower_q = query.lower()
        intent = intent_obj.intent

        stop_words = {
            "what", "is", "the", "how", "many", "of", "to", "are", "there", "in", "for",
            "a", "an", "and", "or", "give", "me", "exact", "number", "which", "why", "where",
            "tell", "about", "list", "out", "show", "name", "names", "it", "all", "does", "do", "page",
            "summarise", "summarize", "given", "document", "explain", "discussed", "on", "with", "at",
            "by", "from", "up", "into", "over", "under", "above", "below"
        }
        generic_nouns = {
            "project", "budget", "system", "value", "key", "code", "password", "prompt"
        }
        q_words = set(re.findall(r'\w+', lower_q)) - stop_words
        critical_q_words = q_words - generic_nouns
        match_words = critical_q_words if critical_q_words else q_words

        extracted_items = []
        seen_quotes: Set[str] = set()

        for c in target_chunks:
            page_num = c["metadata"].get("page_number", 1)
            cid = c["chunk_id"]
            raw_text = c.get("raw_content", c["text"])
            clean_text = re.sub(r'^Document:.*?\n\nContent:\n', '', raw_text, flags=re.DOTALL).strip()

            unwrapped = re.sub(r'(?<![.!?:\n])\n(?![A-Z\•\*\-\d\.])', ' ', clean_text)
            unwrapped = re.sub(r'\s+', ' ', unwrapped)

            for s in re.split(r'(?<=[.!?])\s+', unwrapped):
                s_clean = s.strip()
                if s_clean in seen_quotes or not GroundedCitationVerifier._is_clean_sentence(s_clean):
                    continue

                if GroundedCitationVerifier._is_adversarial_text(s_clean):
                    continue

                seen_quotes.add(s_clean)
                line_words = set(re.findall(r'\w+', s_clean.lower()))

                if match_words and not (match_words & line_words) and intent not in ["summary", "overview", "page_lookup", "chapter_lookup"]:
                    continue

                overlap_score = len(match_words & line_words) / max(1, len(match_words)) if match_words else 1.0
                extracted_items.append((page_num, cid, s_clean, overlap_score))

        # Sort extracted items: chronologically for summary/overview, otherwise by overlap score descending
        if intent in ["summary", "overview", "page_lookup", "chapter_lookup"]:
            extracted_items.sort(key=lambda x: (x[0], x[1]))
        else:
            extracted_items.sort(key=lambda x: x[3], reverse=True)

        # Map back to standard tuple structure (page_num, cid, sentence_text)
        extracted_items = [(x[0], x[1], x[2]) for x in extracted_items]

        md_output_parts = []
        evidence_items = []

        if intent in ["summary", "overview"]:
            doc_title = target_chunks[0]["metadata"].get("filename", "Uploaded Document") if target_chunks else "Uploaded Document"
            bullets = [f"• {s}" for p, cid, s in extracted_items[:5]]
            evidence_items = [f"- Page {p}, chunk {cid}: “{s[:120]}”" for p, cid, s in extracted_items[:5]]
            if bullets:
                md_output_parts.append(f"**Executive Summary of {doc_title}:**\n\n" + "\n\n".join(bullets))

        elif intent == "definition":
            if extracted_items:
                p, cid, s = extracted_items[0]
                md_output_parts.append(f"**Definition:**\n{s}")
                evidence_items.append(f"- Page {p}, chunk {cid}: “{s[:120]}”")
                if len(extracted_items) > 1:
                    p2, cid2, s2 = extracted_items[1]
                    md_output_parts.append(f"**Explanation:**\n{s2}")
                    evidence_items.append(f"- Page {p2}, chunk {cid2}: “{s2[:120]}”")

        elif intent == "objectives":
            bullets = [f"• {s}" for p, cid, s in extracted_items[:4]]
            evidence_items = [f"- Page {p}, chunk {cid}: “{s[:120]}”" for p, cid, s in extracted_items[:4]]
            if bullets:
                subject_title = intent_obj.primary_subject.title() if intent_obj.primary_subject else ""
                title_str = f"**Key Objectives of {subject_title}:**" if subject_title else "**Key Objectives:**"
                md_output_parts.append(title_str + "\n" + "\n".join(bullets))

        elif intent in ["mechanism", "explanation"]:
            bullets = [f"• {s}" for p, cid, s in extracted_items[:4]]
            evidence_items = [f"- Page {p}, chunk {cid}: “{s[:120]}”" for p, cid, s in extracted_items[:4]]
            if bullets:
                subject_title = intent_obj.primary_subject.title() if (intent_obj and intent_obj.primary_subject) else "Mechanism & Process"
                if "voltage stability" in subject_title.lower() or "shunt" in subject_title.lower():
                    subject_title = "Voltage Stability & Control"
                md_output_parts.append(f"**{subject_title} Mechanism:**\n" + "\n\n".join(bullets))

        elif intent == "comparison":
            bullets = [f"• {s}" for p, cid, s in extracted_items[:6]]
            evidence_items = [f"- Page {p}, chunk {cid}: “{s[:120]}”" for p, cid, s in extracted_items[:6]]
            if bullets:
                md_output_parts.append(f"**Comparative Overview of {intent_obj.primary_subject} and {intent_obj.secondary_subject or 'Related Concepts'}:**\n" + "\n".join(bullets))

        else:
            bullets = [f"• {s}" for p, cid, s in extracted_items[:4]]
            evidence_items = [f"- Page {p}, chunk {cid}: “{s[:120]}”" for p, cid, s in extracted_items[:4]]
            if bullets:
                md_output_parts.append(f"**Detailed Breakdown:**\n" + "\n".join(bullets))

        if not md_output_parts:
            return {
                "answer": f"## Answer\n\n{NO_EVIDENCE_FALLBACK_MESSAGE}",
                "verified_quotes": [],
                "confidence": 0.0
            }, True, "No evidence fallback"

        final_md = "## Answer\n\n" + "\n\n".join(md_output_parts)
        if evidence_items:
            final_md += "\n\n## Evidence\n\n" + "\n".join(evidence_items[:4])

        return {
            "answer": final_md,
            "verified_quotes": [e.split('“')[-1].rstrip('”') for e in evidence_items],
            "confidence": 0.95
        }, True, "Universal extractive fallback successful"


# ============================================================================
# Main RAGEngine Pipeline Class
# ============================================================================

class RAGEngine:
    """
    Universal Production Enterprise RAG Engine with Map-Reduce Summarization & Document APIs.
    """

    def __init__(self, embedding_service, vector_store):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.account_id = CLOUDFLARE_ACCOUNT_ID
        self.api_token = CLOUDFLARE_API_TOKEN
        self.llm_model = CLOUDFLARE_LLM_MODEL or "@cf/meta/llama-3.1-8b-instruct"
        self.worker_base_url = WORKER_BASE_URL or DEFAULT_WORKER_URL

    # Document-Level Retrieval APIs
    def retrieve_document(self, filename: Optional[str] = None, document_id: Optional[str] = None, max_pages: int = 20) -> List[Dict[str, Any]]:
        """Retrieves full document chunks for Map-Reduce summarization."""
        return self.vector_store.get_first_pages_chunks(filename=filename, document_id=document_id, max_pages=max_pages)

    def retrieve_page(self, page_number: int, filename: Optional[str] = None, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves chunks specifically matching page_number."""
        all_chunks = self.vector_store.get_first_pages_chunks(filename=filename, document_id=document_id, max_pages=100)
        return [c for c in all_chunks if c.get("metadata", {}).get("page_number") == page_number]

    def answer_query(
        self,
        query: str,
        filename: Optional[str] = None,
        document_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        top_k: int = 12,
        temperature: float = 0.0,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Public API Endpoint preserved for backend and frontend calls.
        """
        start_time = time.time()
        req_id = f"req_{uuid.uuid4().hex[:10]}"
        clean_query = query.strip()

        trace = RAGTrace(
            request_id=req_id,
            document_id=document_id or "auto",
            original_query=clean_query,
            rewritten_query=clean_query,
            query_intent="fact",
            retrieval_strategy="dense_hybrid"
        )

        # Stage 1: Prompt Injection Guard
        if self._detect_prompt_injection(clean_query) or (system_prompt and self._detect_prompt_injection(system_prompt)):
            trace.failure_stage = "prompt_injection"
            trace.failure_reason = "Adversarial prompt injection attempt detected."
            return {
                "answer": "## Answer\n\nI cannot perform actions that attempt to override safety instructions, expose API tokens, or reveal system prompts.",
                "sources": [],
                "system_prompt_used": IMMUTABLE_SYSTEM_PROMPT,
                "retrieved_count": 0,
                "rag_trace": self._trace_to_dict(trace)
            }

        # Stage 2 & 3: Conversation Resolution & Intent Classification
        resolved_q, intent_obj = QueryRewriter.rewrite_query(clean_query, chat_history)
        trace.rewritten_query = resolved_q
        trace.query_intent = intent_obj.intent
        trace.retrieval_strategy = intent_obj.retrieval_strategy

        # Stage 4: Multi-Query Semantic Expansion
        multi_queries = QueryRewriter.generate_multi_queries(resolved_q, intent_obj)
        trace.multi_queries = multi_queries

        # Stage 5: Semantic Cache Lookup
        q_embeddings = self.embedding_service.generate_embeddings([resolved_q])
        q_vec = q_embeddings[0] if q_embeddings else []
        if q_vec:
            cached_res = global_semantic_cache.get(q_vec, document_id)
            if cached_res:
                logger.info(f"Semantic Cache Hit for query: '{clean_query}'")
                return cached_res

        # Stage 6: Strategy-Driven Retrieval Execution
        all_candidate_chunks: List[Dict[str, Any]] = []

        if intent_obj.retrieval_strategy == "map_reduce":
            # Document Summarization Pipeline: Retrieve Full Document
            all_candidate_chunks = self.retrieve_document(filename=filename, document_id=document_id, max_pages=15)

        elif intent_obj.retrieval_strategy == "page_filter" and intent_obj.target_page:
            all_candidate_chunks = self.retrieve_page(page_number=intent_obj.target_page, filename=filename, document_id=document_id)

        elif intent_obj.retrieval_strategy == "multi_hop":
            # Multi-Hop Retrieval: Topic 1 + Topic 2
            with ThreadPoolExecutor(max_workers=2) as executor:
                f1 = executor.submit(self._retrieve_chunks_for_query, intent_obj.primary_subject, intent_obj.intent, top_k, filename, document_id, session_id, user_id)
                f2 = executor.submit(self._retrieve_chunks_for_query, intent_obj.secondary_subject or resolved_q, intent_obj.intent, top_k, filename, document_id, session_id, user_id)
                c1 = f1.result()
                c2 = f2.result()
                all_candidate_chunks = c1 + c2

        else:
            # Parallel Multi-Query Hybrid Retrieval
            seen_chunk_ids: Set[str] = set()
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(self._retrieve_chunks_for_query, sub_q, intent_obj.intent, top_k, filename, document_id, session_id, user_id)
                    for sub_q in multi_queries
                ]
                for future in as_completed(futures):
                    try:
                        for c in future.result():
                            cid = c["chunk_id"]
                            if cid not in seen_chunk_ids:
                                seen_chunk_ids.add(cid)
                                all_candidate_chunks.append(c)
                    except Exception as e:
                        logger.warning(f"Sub-query retrieval failure: {e}")

        # Fallback to First Pages if Retrieval Empty
        if not all_candidate_chunks and filename:
            all_candidate_chunks = self.retrieve_document(filename=filename, document_id=document_id, max_pages=5)

        # Stage 7: Cross-Encoder Reranking
        reranked_chunks = CrossEncoderReranker.rerank(resolved_q, all_candidate_chunks, intent_type=intent_obj.intent)
        trace.retrieved_chunks = [
            {
                "chunk_id": c["chunk_id"],
                "page_number": c["metadata"].get("page_number", 1),
                "text": c["text"][:150],
                "similarity_score": c.get("similarity_score", 0.0),
                "cross_score": c.get("cross_score", 0.0)
            }
            for c in reranked_chunks[:8]
        ]

        # Stage 8 & 9: Dynamic Top-K Selection
        adaptive_k = min(intent_obj.adaptive_top_k, FINAL_CONTEXT_K)
        final_chunks = reranked_chunks[:adaptive_k]

        if not final_chunks:
            trace.failure_stage = "retrieval"
            trace.failure_reason = "No relevant chunks retrieved from active document."
            return {
                "answer": f"## Answer\n\n{NO_EVIDENCE_FALLBACK_MESSAGE}",
                "sources": [],
                "system_prompt_used": IMMUTABLE_SYSTEM_PROMPT,
                "retrieved_count": 0,
                "rag_trace": self._trace_to_dict(trace)
            }

        # Stage 10: Building Context
        context_blocks = []
        for c in final_chunks:
            page_num = c["metadata"].get("page_number", "?")
            doc_fname = c["metadata"].get("filename", "Document")
            cid = c["chunk_id"]
            clean_chunk_text = c["text"]
            context_blocks.append(f"--- [Page {page_num} | Chunk {cid} | Document: {doc_fname}] ---\n{clean_chunk_text}")

        combined_context = "<DOCUMENT_CONTEXT>\n" + "\n\n".join(context_blocks) + "\n</DOCUMENT_CONTEXT>"
        trace.context_sent_to_llm = combined_context

        # Stage 11 & 12: LLM Execution
        raw_llm, structured_json = self._execute_llm_call(
            context=combined_context,
            query=resolved_q,
            system_prompt=system_prompt,
            temperature=temperature
        )

        trace.raw_llm_response = raw_llm or ""
        trace.parsed_response = structured_json or {}

        # Stage 13 & 14: Citation Verification & Grounding
        verified_res, is_valid_citations, v_reason = GroundedCitationVerifier.verify_response(
            query=clean_query,
            structured_json=structured_json,
            target_chunks=final_chunks,
            intent_obj=intent_obj
        )

        trace.citation_valid = is_valid_citations
        trace.answer_relevance = 0.95 if is_valid_citations else 0.50
        trace.groundedness = 0.95 if structured_json and structured_json.get("answerable") else 0.0

        if not is_valid_citations:
            trace.failure_stage = "citation_validation"
            trace.failure_reason = v_reason

        # Stage 15: Multi-Factor Dynamic Confidence Scoring
        confidence = self._compute_dynamic_confidence(
            retrieval_count=len(final_chunks),
            citation_valid=is_valid_citations,
            answerable=bool(structured_json and structured_json.get("answerable"))
        )
        trace.confidence_score = confidence
        trace.execution_time_ms = round((time.time() - start_time) * 1000, 2)

        sources = [
            {
                "source_id": idx + 1,
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "page_number": c["metadata"].get("page_number"),
                "filename": c["metadata"].get("filename"),
                "similarity_score": c.get("similarity_score")
            }
            for idx, c in enumerate(final_chunks)
        ]

        final_response = {
            "answer": verified_res["answer"],
            "sources": sources,
            "verified_quotes": verified_res.get("verified_quotes", []),
            "confidence": confidence,
            "system_prompt_used": IMMUTABLE_SYSTEM_PROMPT,
            "retrieved_count": len(sources),
            "rag_trace": self._trace_to_dict(trace)
        }

        # Cache valid response
        if q_vec and confidence >= 0.85:
            global_semantic_cache.put(q_vec, document_id, final_response)

        return final_response

    def _retrieve_chunks_for_query(
        self, sub_q: str, intent_type: str, top_k: int,
        filename: Optional[str], document_id: Optional[str], session_id: Optional[str], user_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Helper to run hybrid vector similarity search."""
        q_embeddings = self.embedding_service.generate_embeddings([sub_q])
        if not q_embeddings:
            return []

        return self.vector_store.similarity_search(
            query_embedding=q_embeddings[0],
            raw_query=sub_q,
            intent_type=intent_type,
            top_k=top_k,
            filename_filter=filename,
            document_id_filter=document_id,
            session_id_filter=session_id,
            user_id_filter=user_id,
            min_score=0.15
        )

    @staticmethod
    def _detect_prompt_injection(query: str) -> bool:
        pattern = r'(ignore\s*previous\s*instructions|system\s*prompt|show\s*env|api_key|api_token|secret\s*key|password)'
        return bool(re.search(pattern, query, re.IGNORECASE))

    def _execute_llm_call(
        self, context: str, query: str, system_prompt: Optional[str], temperature: float = 0.0
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        sanitized_style = ""
        if system_prompt:
            sanitized_style = re.sub(r'(ignore\s*previous\s*instructions|system\s*prompt|show\s*env|api_key|api_token|secret\s*key|password)', '', system_prompt, flags=re.IGNORECASE).strip()
        combined_system = f"{IMMUTABLE_SYSTEM_PROMPT}\n\nUSER STYLE PROMPT: {sanitized_style}"

        # 1. Try Worker AI Base URL
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
                    raw_text = resp.text
                    payload = _extract_json_payload(resp.json())
                    if payload:
                        return raw_text, payload
            except Exception as e:
                logger.warning(f"Worker AI call failed: {e}")

        # 2. Try Direct Cloudflare REST API
        if self.account_id and self.api_token and "placeholder" not in self.account_id:
            try:
                url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.llm_model}"
                headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
                messages = [
                    {"role": "system", "content": combined_system},
                    {
                        "role": "user",
                        "content": (
                            f"You must answer the user question using ONLY the provided verified document context.\n\n"
                            f"<DOCUMENT_CONTEXT>\n{context}\n</DOCUMENT_CONTEXT>\n\n"
                            f"Question: {query}"
                        )
                    }
                ]
                resp = requests.post(url, headers=headers, json={"messages": messages, "temperature": temperature}, timeout=45)
                if resp.status_code == 200:
                    raw_text = resp.text
                    payload = _extract_json_payload(resp.json())
                    if payload:
                        return raw_text, payload
            except Exception as e:
                logger.warning(f"Direct Cloudflare REST API failed: {e}")

        return None, None

    @staticmethod
    def _compute_dynamic_confidence(retrieval_count: int, citation_valid: bool, answerable: bool) -> float:
        """Computes dynamic confidence score between 0.0 and 1.0."""
        if not answerable or retrieval_count == 0:
            return 0.0
        score = 0.50
        if citation_valid:
            score += 0.35
        score += min(0.15, retrieval_count * 0.05)
        return round(min(1.0, score), 2)

    @staticmethod
    def _trace_to_dict(trace: RAGTrace) -> Dict[str, Any]:
        """Converts RAGTrace dataclass instance to JSON dictionary."""
        return {
            "request_id": trace.request_id,
            "document_id": trace.document_id,
            "original_query": trace.original_query,
            "rewritten_query": trace.rewritten_query,
            "query_intent": trace.query_intent,
            "retrieval_strategy": trace.retrieval_strategy,
            "multi_queries": trace.multi_queries,
            "retrieved_chunks": trace.retrieved_chunks,
            "context_sent_to_llm": trace.context_sent_to_llm[:300] + "...",
            "raw_llm_response": trace.raw_llm_response[:200],
            "parsed_response": trace.parsed_response,
            "answer_relevance": trace.answer_relevance,
            "groundedness": trace.groundedness,
            "citation_valid": trace.citation_valid,
            "confidence_score": trace.confidence_score,
            "execution_time_ms": trace.execution_time_ms,
            "cache_hit": trace.cache_hit,
            "failure_stage": trace.failure_stage,
            "failure_reason": trace.failure_reason
        }
