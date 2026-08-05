import requests
import logging
import json
import re
import uuid
import time
import os
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from backend.config import DATA_DIR

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from backend.config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_LLM_MODEL,
    IMMUTABLE_SYSTEM_PROMPT,
    MIN_RELEVANCE_SCORE,
    FINAL_CONTEXT_K,
    DENSE_TOP_K,
    NO_EVIDENCE_FALLBACK_MESSAGE,
    DEBUG_RAG,
    ENABLE_ONNX_RERANKER
)
from backend.services.worker_analyzer import WORKER_BASE_URL, DEFAULT_WORKER_URL

logger = logging.getLogger(__name__)


# ============================================================================
# Dataclasses & Data Models
# ============================================================================

@dataclass
class QueryIntent:
    """Represents classified query intent, subjects, and execution parameters."""
    intent: str  # summary, review, list_items, extract_fields, qa, compare, rewrite, action_items, unknown
    reason: str
    retrieval_mode: str  # broad, hybrid, focused, sectional
    response_format: str  # prose, bullets, table, json
    broad_coverage: bool
    exact_extraction: bool
    section_specific: bool
    targets: List[str]
    rewritten_query: str
    confidence: float
    ambiguity: bool
    clarification_needed: bool
    clarification_question: str
    primary_subject: str = ""
    secondary_subject: Optional[str] = None
    target_page: Optional[int] = None
    target_chapter: Optional[int] = None

    @property
    def retrieval_strategy(self) -> str:
        if self.retrieval_mode == "broad":
            return "map_reduce"
        elif self.retrieval_mode == "sectional":
            return "multi_hop"
        elif self.target_page:
            return "page_filter"
        elif self.target_chapter:
            return "chapter_filter"
        else:
            return "dense_hybrid"

    @property
    def adaptive_top_k(self) -> int:
        if self.retrieval_mode == "broad":
            return 15
        elif self.retrieval_mode == "hybrid":
            return 10
        elif self.retrieval_mode == "sectional":
            return 8
        else:
            return 5


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
    context_recall: float = 0.0
    answer_faithfulness: float = 0.0
    citation_correctness: float = 0.0
    response_latency_ms: float = 0.0


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


class ONNXCrossEncoderReranker:
    """
    Local Neural Cross-Encoder Reranker using ONNX runtime and Xenova/bge-reranker-base.
    Computes joint sequence attention for true semantic reranking.
    """
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or (DATA_DIR / "onnx_reranker")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.cache_dir / "model.onnx"
        self.tokenizer_path = self.cache_dir / "tokenizer.json"
        self.ort_session = None
        self.tokenizer = None
        self.enabled = False

    def load(self):
        import urllib.request
        model_url = "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/onnx/model.onnx"
        tokenizer_url = "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/tokenizer.json"
        
        try:
            if not self.model_path.exists():
                logger.info(f"Downloading ONNX reranker model to {self.model_path}...")
                urllib.request.urlretrieve(model_url, self.model_path)
            if not self.tokenizer_path.exists():
                logger.info(f"Downloading reranker tokenizer to {self.tokenizer_path}...")
                urllib.request.urlretrieve(tokenizer_url, self.tokenizer_path)
            
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            self.ort_session = ort.InferenceSession(str(self.model_path))
            self.enabled = True
            logger.info("Successfully loaded local ONNX Neural Cross-Encoder Reranker.")
        except Exception as e:
            logger.warning(f"Failed to load ONNX reranker: {e}. Falling back to heuristic reranking.")
            self.enabled = False

    def compute_score(self, query: str, document: str) -> float:
        if not self.ort_session or not self.tokenizer:
            self.load()
        if not self.enabled:
            return 0.0

        encoded = self.tokenizer.encode(query, document)
        input_ids = encoded.ids
        attention_mask = encoded.attention_mask

        in_ids = np.array([input_ids], dtype=np.int64)
        attn_mask = np.array([attention_mask], dtype=np.int64)
        
        inputs = {
            "input_ids": in_ids,
            "attention_mask": attn_mask
        }
        
        outputs = self.ort_session.run(None, inputs)
        logits = outputs[0]  # Shape: [1, 1]
        
        # Sigmoid to get probability/score
        score = 1.0 / (1.0 + math.exp(-float(logits[0][0])))
        return score


class CrossEncoderReranker:
    """Pluggable Cross-Encoder Reranker engine (Neural / Heuristic)."""
    
    _neural_reranker = None

    @staticmethod
    def rerank(query: str, chunks: List[Dict[str, Any]], intent_type: str = "fact") -> List[Dict[str, Any]]:
        if not chunks:
            return []

        # Try to use local ONNX Neural Cross-Encoder if available
        if HAS_ONNX and ENABLE_ONNX_RERANKER:
            if CrossEncoderReranker._neural_reranker is None:
                try:
                    CrossEncoderReranker._neural_reranker = ONNXCrossEncoderReranker()
                    CrossEncoderReranker._neural_reranker.load()
                except Exception as e:
                    logger.warning(f"Could not load neural reranker: {e}")
            
            if CrossEncoderReranker._neural_reranker and CrossEncoderReranker._neural_reranker.enabled:
                logger.info(f"Running local ONNX Neural Cross-Encoder Reranker on {len(chunks)} candidate chunks...")
                scored_chunks = []
                for chunk in chunks:
                    try:
                        score = CrossEncoderReranker._neural_reranker.compute_score(query, chunk["text"])
                        chunk_copy = chunk.copy()
                        chunk_copy["cross_score"] = round(score, 4)
                        scored_chunks.append(chunk_copy)
                    except Exception as e:
                        logger.warning(f"Neural rerank failed for chunk: {e}")
                        chunk_copy = chunk.copy()
                        chunk_copy["cross_score"] = chunk.get("similarity_score", 0.0)
                        scored_chunks.append(chunk_copy)
                
                scored_chunks.sort(key=lambda x: x["cross_score"], reverse=True)
                return scored_chunks

        # Heuristic fallback (if ONNX is not available or fails)
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
                res = _extract_json_payload(val)
                if res:
                    return res

    elif isinstance(data, str):
        try:
            match = re.search(r'\{.*\}', data, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and "answerable" in parsed:
                    return parsed
        except Exception:
            pass

        # Fallback regex parsing for broken JSON strings
        ans_match = re.search(r'"answer"\s*:\s*"(.*?)"', data, re.DOTALL)
        if ans_match:
            try:
                ans_str = ans_match.group(1).encode().decode('unicode-escape')
                return {
                    "answerable": True,
                    "answer": ans_str,
                    "parts": [],
                    "confidence": 0.85
                }
            except Exception:
                return {
                    "answerable": True,
                    "answer": ans_match.group(1),
                    "parts": [],
                    "confidence": 0.85
                }

        # Generative Fallback: If it is conversational prose, wrap it as the answer directly
        clean_text = data.strip()
        if clean_text and not clean_text.startswith("{") and len(clean_text) > 30:
            return {
                "answerable": True,
                "answer": clean_text,
                "parts": [],
                "confidence": 0.85
            }

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
        if chat_history and len(re.findall(r'\w+', query)) <= 5:
            user_msgs = [m["content"] for m in chat_history if m.get("role") == "user" and m.get("content", "").strip()]
            if user_msgs:
                prev_q = user_msgs[-1].strip()
                prev_q_lower = prev_q.lower()
                # Do not fuse if previous query was a summary/overview query
                is_prev_summary = any(x in prev_q_lower for x in ["summarise", "summarize", "summary", "overview", "pdf about"])
                if not is_prev_summary:
                    pronouns = {"he", "she", "it", "they", "him", "her", "them", "his", "their", "its", "this", "that", "these", "those", "himself", "herself", "itself"}
                    words = set(re.findall(r'\w+', lower_q))
                    has_pronouns = bool(words & pronouns)
                    is_tiny_fragment = len(re.findall(r'\w+', query)) <= 2
                    
                    question_words = {"what", "how", "why", "who", "where", "define", "explain", "describe", "compare", "versus", "vs", "list"}
                    is_complete_question = bool(words & question_words)
                    
                    should_fuse = False
                    if has_pronouns:
                        should_fuse = True
                    elif is_tiny_fragment and not is_complete_question:
                        should_fuse = True
                        
                    is_subject_change = ("hvdc" in prev_q_lower and "shunt" in lower_q) or ("shunt" in prev_q_lower and "hvdc" in lower_q)
                    
                    if should_fuse and not is_subject_change:
                        if prev_q and prev_q_lower != lower_q:
                            return f"{prev_q} - {query}"

        return clean_q


class QueryRewriter:
    """Enterprise 15-Intent Classifier & Multi-Query Expansion Engine."""

    @staticmethod
    def rewrite_query(query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, QueryIntent]:
        resolved_q = ConversationResolver.resolve_conversation(query, chat_history)
        lower_q = resolved_q.lower().strip("?!., ")

        # Target Page & Chapter Lookups (preserved for backward compatibility helper routing)
        page_match = re.search(r'\bpage\s+(\d+)\b', lower_q)
        chapter_match = re.search(r'\bchapter\s+(\d+)\b', lower_q)
        target_page = int(page_match.group(1)) if page_match else None
        target_chapter = int(chapter_match.group(1)) if chapter_match else None

        # 1. Detect Intent
        intent = "qa"
        reason = "Direct factual query matching the question-answering intent."
        retrieval_mode = "focused"
        response_format = "prose"
        broad_coverage = False
        exact_extraction = False
        section_specific = False
        targets = ["factual answer"]

        # Check rules in priority order
        if any(w in lower_q for w in ["insight", "analysis", "review", "strength", "weakness", "risk", "gap", "concern", "recommendation"]):
            intent = "review"
            reason = "User requested high-level analysis, insights, risk evaluation, or recommendations."
            retrieval_mode = "broad"
            response_format = "prose"
            broad_coverage = True
            targets = ["insights", "analysis", "observations", "risks", "gaps", "strengths", "weaknesses", "recommendations"]

        elif any(w in lower_q for w in ["summarize", "summarise", "summary", "overview", "what is this pdf about", "main points"]):
            intent = "summary"
            reason = "User requested a concise summary or general overview of the document contents."
            retrieval_mode = "broad"
            response_format = "prose"
            broad_coverage = True
            targets = ["summary", "overview", "main points", "executive summary"]

        elif any(w in lower_q for w in ["list", "enumerate", "show all", "provide bullets", "collect items", "bullet points", "clause", "obligation", "requirement", "key point", "items", "enumeration", "objective", "objectives", "purpose"]):
            intent = "list_items"
            reason = "User wants key points, bullet lists, clauses, obligations, or requirement items extracted."
            retrieval_mode = "hybrid"
            response_format = "bullets"
            exact_extraction = True
            targets = ["key points", "bulleted items", "obligations", "requirements", "clauses"]

        elif any(w in lower_q for w in ["extract", "find names", "find dates", "invoice number", "total", "monetary amount", "entity", "labeled field", "clause", "risk"]):
            intent = "extract_fields"
            reason = "User requested structured entities, exact dates, names, invoice numbers, or monetary values."
            retrieval_mode = "hybrid"
            response_format = "json"
            exact_extraction = True
            targets = ["entities", "names", "dates", "monetary amounts", "IDs", "invoice numbers", "totals"]

        elif any(w in lower_q for w in ["difference", "similarity", "versus", "vs", "compare", "similarities"]):
            intent = "compare"
            reason = "User requested a comparison of differences, similarities, or versions."
            retrieval_mode = "sectional"
            response_format = "table"
            section_specific = True
            targets = ["comparison", "differences", "similarities"]

        elif any(w in lower_q for w in ["todo", "to-do", "action item", "deadline", "deliverable", "next step", "what needs to be done", "obligation", "tasks"]):
            intent = "action_items"
            reason = "User requested obligations, deadlines, next steps, or project deliverables."
            retrieval_mode = "hybrid"
            response_format = "bullets"
            exact_extraction = True
            targets = ["tasks", "deadlines", "action items", "deliverables"]

        elif any(w in lower_q for w in ["simplify", "rephrase", "rewrite", "clean", "translate", "paraphrase"]):
            intent = "rewrite"
            reason = "User requested rephrased, simplified, or translated text representation."
            retrieval_mode = "broad"
            response_format = "prose"
            broad_coverage = True
            targets = ["rewritten text", "simplified content", "rephrasing"]

        elif len(lower_q.split()) <= 1 or lower_q in ["test", "hello", "hi", "help"]:
            intent = "unknown"
            reason = "The query has unclear or unsupported intent characteristics."
            retrieval_mode = "focused"
            response_format = "prose"
            targets = []

        # 2. Check Ambiguity and Clarification triggers
        ambiguity = False
        clarification_needed = False
        clarification_question = ""

        if "list it" in lower_q or (re.search(r'\blist\b', lower_q) and not any(x in lower_q for x in ["points", "items", "terms", "clauses", "dates", "names", "numbers", "obligations", "requirements", "details", "everything", "facts"])):
            ambiguity = True
            clarification_needed = True
            clarification_question = "What specifically would you like me to list from the document?"

        elif "compare" in lower_q and not any(x in lower_q for x in ["sections", "terms", "concepts", "clauses", "versions", "and", "vs", "versus"]):
            ambiguity = True
            clarification_needed = True
            clarification_question = "What specific sections, terms, or versions would you like me to compare?"

        elif "extract" in lower_q and not any(x in lower_q for x in ["names", "dates", "invoices", "totals", "clauses", "risks", "items", "fields"]):
            ambiguity = True
            clarification_needed = True
            clarification_question = "Which specific fields (e.g., names, dates, amounts) would you like me to extract?"

        elif intent == "unknown":
            ambiguity = True
            clarification_needed = True
            clarification_question = "Please provide more details on what you would like to analyze, extract, or ask about the document."

        # 3. Formulate Search-Friendly Query
        rewritten_query = lower_q
        if intent == "summary":
            rewritten_query = f"Executive summary, main topics, and overview of the document contents for {lower_q}"
        elif intent == "review":
            rewritten_query = f"Detailed document analysis, observations, risks, strengths, weaknesses, and recommendations for {lower_q}"
        elif intent == "list_items":
            rewritten_query = f"List of key points, requirements, obligations, and clauses in {lower_q}"
        elif intent == "extract_fields":
            rewritten_query = f"Extraction of names, dates, amounts, invoice numbers, clauses, and structured entities in {lower_q}"
        elif intent == "compare":
            rewritten_query = f"Comparison of differences and similarities in {lower_q}"
        elif intent == "action_items":
            rewritten_query = f"Extraction of action items, deadlines, deliverables, obligations, and tasks in {lower_q}"
        elif intent == "rewrite":
            rewritten_query = f"Rephrased, simplified, and cleaned version of {lower_q}"

        confidence = 0.0 if clarification_needed else 0.95

        intent_obj = QueryIntent(
            intent=intent,
            reason=reason,
            retrieval_mode=retrieval_mode,
            response_format=response_format,
            broad_coverage=broad_coverage,
            exact_extraction=exact_extraction,
            section_specific=section_specific,
            targets=targets,
            rewritten_query=rewritten_query,
            confidence=confidence,
            ambiguity=ambiguity,
            clarification_needed=clarification_needed,
            clarification_question=clarification_question,
            primary_subject=lower_q,
            target_page=target_page,
            target_chapter=target_chapter
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
            variations.append(f"Overview of {subj1} main themes and general principles")
        elif intent_obj.intent == "definition":
            variations.append(f"Define {subj1}")
            variations.append(f"What is the definition and purpose of {subj1}")
        elif intent_obj.intent == "objectives":
            variations.append(f"Key objectives and aims of {subj1}")
            variations.append(f"Why is {subj1} used and what does it accomplish")
        elif intent_obj.intent in ["mechanism", "explanation"]:
            variations.append(f"How does {subj1} work step by step")
            variations.append(f"Operation, function, and implementation mechanism of {subj1}")
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
            res_dict, valid, msg = GroundedCitationVerifier._fallback_universal_extractive(query, target_chunks, intent_obj)
            res_dict["answer_faithfulness"] = 1.0
            res_dict["citation_correctness"] = 1.0
            return res_dict, valid, msg

        raw_definition = structured_json.get("definition", "").strip()
        raw_explanation = structured_json.get("explanation", "").strip()
        raw_answer = structured_json.get("answer", "").strip()

        evidence_list = structured_json.get("evidence", structured_json.get("claims", []))
        confidence = float(structured_json.get("confidence", 0.90))

        chunk_map = {c["chunk_id"]: c for c in target_chunks}
        chunk_text_map = {
            c["chunk_id"]: c["text"] + "\n" + c.get("raw_content", "") + "\n" + c["metadata"].get("parent_text", "")
            for c in target_chunks
        }
        all_text_combined = "\n\n".join([
            c["text"] + "\n" + c.get("raw_content", "") + "\n" + c["metadata"].get("parent_text", "")
            for c in target_chunks
        ])

        seen_quotes: Set[str] = set()
        verified_evidence_items: List[str] = []
        valid_citations = True
        
        correct_citations = 0
        total_citations = len(evidence_list)

        for cite in evidence_list:
            cid = cite.get("chunk_id") or (cite.get("supporting_chunk_ids", [None])[0] if cite.get("supporting_chunk_ids") else None)
            page = cite.get("page") or (cite.get("page_numbers", [1])[0] if cite.get("page_numbers") else 1)
            quote = (cite.get("quote") or cite.get("support_quote", "")).strip()

            is_correct = True
            if cid and cid not in chunk_map:
                valid_citations = False
                is_correct = False

            if cid in chunk_map and page and page != chunk_map[cid]["metadata"].get("page_number"):
                valid_citations = False
                is_correct = False

            if quote and len(quote) > 8:
                quote_clean = re.sub(r'\s+', ' ', quote.lower())
                target_text_clean = re.sub(r'\s+', ' ', chunk_text_map.get(cid, all_text_combined).lower())
                
                is_match = (quote_clean in target_text_clean)
                if not is_match:
                    quote_words = set(re.findall(r'\w+', quote_clean))
                    chunk_words = set(re.findall(r'\w+', target_text_clean))
                    overlap_ratio = len(quote_words & chunk_words) / max(1, len(quote_words))
                    if overlap_ratio >= 0.75:
                        is_match = True
                if not is_match:
                    valid_citations = False
                    is_correct = False
                else:
                    if quote not in seen_quotes:
                        seen_quotes.add(quote)
                        verified_evidence_items.append(f"- Page {page or 1}, chunk {cid or 'c0'}: “{quote}”")
            if is_correct:
                correct_citations += 1

        citation_correctness = (correct_citations / total_citations) if total_citations > 0 else 1.0

        # Construct Final Formatted Output Answer
        formatted_answer_parts = []

        if raw_answer:
            formatted_answer_parts.append(f"## Answer\n\n{raw_answer}")
        elif raw_definition or raw_explanation:
            if raw_definition:
                formatted_answer_parts.append(f"## Definition\n\n{raw_definition}")
            if raw_explanation:
                formatted_answer_parts.append(f"\n## Detailed Breakdown\n\n{raw_explanation}")

        # List/Table/Structured JSON representations handling
        parts_data = structured_json.get("parts", [])
        if parts_data and isinstance(parts_data, list):
            formatted_answer_parts.append("\n### Extracted Details")
            for idx, part in enumerate(parts_data, 1):
                if isinstance(part, dict):
                    title = part.get("title", part.get("item", f"Point {idx}"))
                    desc = part.get("description", part.get("details", ""))
                    p_num = part.get("page", "")
                    page_str = f" (Page {p_num})" if p_num else ""
                    formatted_answer_parts.append(f"{idx}. **{title}**{page_str}: {desc}")
                elif isinstance(part, str):
                    formatted_answer_parts.append(f"- {part}")

        final_text = "\n\n".join(formatted_answer_parts).strip()
        if not final_text:
            return GroundedCitationVerifier._fallback_universal_extractive(query, target_chunks, intent_obj)

        res_dict = {
            "answer": final_text,
            "parts": parts_data,
            "confidence": confidence,
            "evidence": verified_evidence_items,
            "answer_faithfulness": round(confidence, 2),
            "citation_correctness": round(citation_correctness, 2)
        }

        return res_dict, valid_citations, "Citations verified successfully"

    @staticmethod
    def _fallback_universal_extractive(
        query: str,
        target_chunks: List[Dict[str, Any]],
        intent_obj: QueryIntent
    ) -> Tuple[Dict[str, Any], bool, str]:
        """Universal Extractive Fallback Engine to guarantee answer generation."""
        logger.warning("LLM response not structure-compliant. Triggering Universal Extractive Fallback.")
        
        selected_text_blocks = []
        verified_evidence = []
        
        for idx, chunk in enumerate(target_chunks[:3], 1):
            text = chunk.get("raw_content", chunk.get("text", "")).strip()
            page = chunk.get("metadata", {}).get("page_number", 1)
            cid = chunk.get("chunk_id", f"c{idx}")
            
            selected_text_blocks.append(f"**From Page {page} (Excerpt {idx}):**\n> {text}")
            snippet = text[:100].replace('\n', ' ') + "..."
            verified_evidence.append(f"- Page {page}, chunk {cid}: “{snippet}”")

        fallback_answer = (
            f"## Direct Document Excerpts\n\n"
            f"Based directly on the most relevant sections of the document:\n\n"
            + "\n\n".join(selected_text_blocks)
        )

        res_dict = {
            "answer": fallback_answer,
            "parts": [],
            "confidence": 0.70,
            "evidence": verified_evidence,
            "answer_faithfulness": 1.0,
            "citation_correctness": 1.0
        }

        return res_dict, True, "Generated using Universal Extractive Fallback"
