import os
import logging
import httpx
import re
import json
from json import JSONDecodeError
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Base Worker URL (Loaded strictly from service environment variables)
WORKER_BASE_URL = os.getenv("LLM_ANALYSIS_URL", "").strip().rstrip("/")

WORKER_ANALYZE_URL = f"{WORKER_BASE_URL}/analyze" if WORKER_BASE_URL and not WORKER_BASE_URL.endswith("/analyze") else WORKER_BASE_URL
WORKER_EMBEDDINGS_URL = f"{WORKER_BASE_URL}/embeddings" if WORKER_BASE_URL else ""

MAX_ANALYSIS_CHARS = 12000
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120.0"))

DEFAULT_ANALYSIS = {
    "summary": "",
    "topics": [],
    "keywords": [],
    "sentiment": "neutral",
    "important_points": [],
    "action_items": [],
}

try:
    import repairjson  # pip install repairjson
except Exception:
    repairjson = None


def _safe_analysis(error: str = "", raw_response: str = "") -> dict:
    payload = DEFAULT_ANALYSIS.copy()
    if error:
        payload["error"] = error
    if raw_response:
        payload["raw_response"] = raw_response[:4000]
    return payload


def _normalize_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = [item.strip(" \t\r\n\"'[]") for item in value.split(",")]
        return [item for item in parts if item]
    return [str(value).strip()]


def _normalize_analysis_dict(data: dict) -> dict:
    result = DEFAULT_ANALYSIS.copy()
    if not isinstance(data, dict):
        return result

    summary = data.get("summary", "")
    sentiment = data.get("sentiment", "neutral")

    result["summary"] = str(summary).strip() if summary is not None else ""
    result["sentiment"] = str(sentiment).strip().lower() if sentiment else "neutral"
    result["topics"] = _normalize_list(data.get("topics", []))
    result["keywords"] = _normalize_list(data.get("keywords", []))
    result["important_points"] = _normalize_list(
        data.get("important_points", data.get("takeaways", []))
    )
    result["action_items"] = _normalize_list(data.get("action_items", []))

    if "error" in data and data["error"]:
        result["error"] = str(data["error"])
    if "raw_response" in data and data["raw_response"]:
        result["raw_response"] = str(data["raw_response"])[:4000]

    return result


def extract_text_for_llm(result: dict) -> str:
    if not result or not isinstance(result, dict):
        return ""

    for key in ["extracted_content", "text", "content", "markdown", "result", "extracted_text"]:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        elif value and not isinstance(value, (str, bool, int, float)):
            try:
                return json.dumps(value, indent=2, ensure_ascii=False)
            except Exception:
                pass

    data = result.get("extracted_data") or result.get("data")
    if isinstance(data, dict):
        pages = data.get("pages")
        if isinstance(pages, list):
            extracted = []
            for page in pages:
                if not isinstance(page, dict):
                    continue
                for key in ["markdown", "text", "content"]:
                    value = page.get(key)
                    if isinstance(value, str) and value.strip():
                        extracted.append(value.strip())
                        break
            if extracted:
                return "\n\n".join(extracted)

        for key in ["text", "content", "markdown"]:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            elif value and not isinstance(value, (str, bool, int, float)):
                try:
                    return json.dumps(value, indent=2, ensure_ascii=False)
                except Exception:
                    pass

    pages = result.get("pages")
    if isinstance(pages, list):
        extracted = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            for key in ["markdown", "text", "content"]:
                value = page.get(key)
                if isinstance(value, str) and value.strip():
                    extracted.append(value.strip())
                    break
        if extracted:
            return "\n\n".join(extracted)

    return ""


def clean_text(text: str) -> str:
    if not text:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""

    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)

    lines = text.split("\n")
    cleaned_lines = []

    boilerplate_patterns = [
        r".*privacy\s*policy.*",
        r".*terms\s*of\s*(?:service|use).*",
        r".*all\s*rights\s*reserved.*",
        r".*copyright\s*(?:©|c|\(c\))?\s*\d{4}.*",
        r".*cookie\s*policy.*",
        r".*contact\s*us.*",
        r".*about\s*us.*",
        r".*careers.*",
        r".*help\s*&\s*support.*",
        r"^\s*sign\s*in\s*/\s*register\s*$",
        r"^\s*login\s*$",
        r"^\s*sign\s*up\s*$",
        r"^\s*forgot\s*password\s*$",
        r"^\s*skip\s*to\s*content\s*$",
        r"^\s*navigation\s*$",
        r"^\s*menu\s*$",
        r"^#+\s*navigation\s*$",
        r"^#+\s*menu\s*$",
        r".*share\s*on.*",
        r".*follow\s*us.*",
        r".*subscribe.*",
        r".*create\s*account.*",
        r".*join\s*for\s*free.*",
        r".*download\s*our\s*app.*",
    ]

    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in boilerplate_patterns]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if len(stripped) < 80 and any(pattern.match(stripped) for pattern in compiled_patterns):
            continue

        if re.match(r"^[_\\-\\*\\=\\#\\s\\d\\|\\|]+$", stripped) and len(stripped) > 2:
            continue

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_message_content(response_json) -> str:
    if isinstance(response_json, str):
        return response_json.strip()

    if not isinstance(response_json, dict):
        return str(response_json).strip()

    result = response_json.get("result", response_json)

    if isinstance(result, dict) and any(
        key in result for key in ["summary", "topics", "keywords", "important_points", "action_items"]
    ):
        return json.dumps(result, ensure_ascii=False)

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {}) or {}
            content = message.get("content") or message.get("reasoning_content") or ""
            if isinstance(content, list):
                return " ".join(str(item) for item in content).strip()
            return str(content).strip()

        if "response" in result and isinstance(result["response"], str):
            return result["response"].strip()

    return json.dumps(response_json, ensure_ascii=False)


def _parse_key_value_fallback(raw_text: str) -> dict:
    structured = DEFAULT_ANALYSIS.copy()

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue

        key, val = line.split(":", 1)
        key = key.strip().lower()
        val = val.strip()

        if "summary" == key or key.endswith("summary"):
            structured["summary"] = val.strip("\"'")
        elif "topic" in key:
            structured["topics"] = _normalize_list(val)
        elif "keyword" in key:
            structured["keywords"] = _normalize_list(val)
        elif "sentiment" in key:
            structured["sentiment"] = val.strip("\"'").lower() or "neutral"
        elif "important" in key or "takeaway" in key:
            structured["important_points"] = _normalize_list(val)
        elif "action" in key:
            structured["action_items"] = _normalize_list(val)

    if not any([
        structured["summary"],
        structured["topics"],
        structured["keywords"],
        structured["important_points"],
        structured["action_items"],
    ]):
        structured["error"] = "Model returned invalid JSON."
        structured["raw_response"] = raw_text[:4000]

    return structured


def parse_llm_response(response_json) -> dict:
    raw_text = _extract_message_content(response_json)

    if not raw_text:
        return _safe_analysis("The model returned an empty response.")

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return _normalize_analysis_dict(parsed)
    except Exception:
        pass

    if repairjson is not None:
        try:
            repaired = repairjson.loads(raw_text)
            if isinstance(repaired, dict):
                return _normalize_analysis_dict(repaired)
        except Exception:
            pass

    key_value_guess = _parse_key_value_fallback(raw_text)
    if any([
        key_value_guess.get("summary"),
        key_value_guess.get("topics"),
        key_value_guess.get("keywords"),
        key_value_guess.get("important_points"),
        key_value_guess.get("action_items"),
    ]):
        return _normalize_analysis_dict(key_value_guess)

    return _safe_analysis("Model returned invalid JSON.", raw_text)


async def generate_bge_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts or not WORKER_EMBEDDINGS_URL:
        return []

    payload = {"text": texts}

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(WORKER_EMBEDDINGS_URL, json=payload)

        if response.status_code != 200:
            logger.error(f"Embedding worker failed with status {response.status_code}: {response.text}")
            raise RuntimeError(f"Embedding worker failed with status {response.status_code}")

        body = response.json()
        data = body.get("data", [])
        if not data and "result" in body:
            data = body["result"].get("data", [])

        return data

    except Exception as exc:
        logger.exception("Failed to generate BGE embeddings from Cloudflare Worker")
        raise RuntimeError(f"Embedding generation error: {str(exc)}")


async def analyze_extracted_data(
    url: str,
    title: str,
    extracted_text: str,
    analysis_type: str = "summary"
):
    if not extracted_text or not extracted_text.strip():
        return _safe_analysis("No extracted text available for LLM analysis.")

    if not WORKER_ANALYZE_URL:
        return _safe_analysis("LLM_ANALYSIS_URL environment variable is not configured.")

    raw_len = len(extracted_text)
    cleaned_text = clean_text(extracted_text)

    if not cleaned_text.strip():
        cleaned_text = extracted_text.strip()

    cleaned_len = len(cleaned_text)
    reduction = ((raw_len - cleaned_len) / raw_len) * 100 if raw_len else 0
    logger.info(
        f"LLM analysis context compression: {raw_len} chars -> {cleaned_len} chars ({reduction:.1f}% reduction)"
    )

    if len(cleaned_text) > MAX_ANALYSIS_CHARS:
        cleaned_text = cleaned_text[:MAX_ANALYSIS_CHARS]
        logger.info(f"Context truncated to fit limit: {len(cleaned_text)} chars")

    payload = {
        "url": url or "",
        "title": title or "",
        "text": cleaned_text,
        "analysis_type": analysis_type or "summary",
        "target_language": "English",
        "prompt": "Analyze the provided web content and produce all output fields (summary, topics, keywords, sentiment, important_points, action_items) strictly in English language."
    }

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(WORKER_ANALYZE_URL, json=payload)

        if response.status_code != 200:
            return _safe_analysis(
                f"LLM analysis failed with status {response.status_code}.",
                response.text,
            )

        try:
            body = response.json()
        except (JSONDecodeError, ValueError):
            return _safe_analysis(
                "Worker returned non-JSON response.",
                response.text,
            )

        return parse_llm_response(body)

    except httpx.TimeoutException:
        return _safe_analysis("LLM analysis timed out.")
    except httpx.RequestError as exc:
        return _safe_analysis(f"LLM request error: {str(exc)}")
    except Exception as exc:
        logger.exception("Unexpected LLM analysis failure")
        return _safe_analysis(f"Unexpected LLM analysis error: {str(exc)}")
