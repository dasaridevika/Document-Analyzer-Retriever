# backend/services/intent_router.py
from enum import Enum
from pydantic import BaseModel

class TaskType(str, Enum):
    SUMMARY = "summary"
    REVIEW = "review"
    LIST_ITEMS = "list_items"
    EXTRACT_FIELDS = "extract_fields"
    QA = "qa"
    COMPARE = "compare"
    REWRITE = "rewrite"
    ACTION_ITEMS = "action_items"
    UNKNOWN = "unknown"

class IntentConfig(BaseModel):
    task: TaskType
    top_k: int
    query_expansion: bool

# Map tasks to intent-based retrieval strategies
INTENT_STRATEGY_MAP = {
    TaskType.EXTRACT_FIELDS: IntentConfig(task=TaskType.EXTRACT_FIELDS, top_k=3, query_expansion=False),
    TaskType.QA: IntentConfig(task=TaskType.QA, top_k=5, query_expansion=False),
    TaskType.LIST_ITEMS: IntentConfig(task=TaskType.LIST_ITEMS, top_k=8, query_expansion=True),
    TaskType.ACTION_ITEMS: IntentConfig(task=TaskType.ACTION_ITEMS, top_k=8, query_expansion=True),
    TaskType.COMPARE: IntentConfig(task=TaskType.COMPARE, top_k=10, query_expansion=True),
    TaskType.SUMMARY: IntentConfig(task=TaskType.SUMMARY, top_k=15, query_expansion=False),
    TaskType.REVIEW: IntentConfig(task=TaskType.REVIEW, top_k=15, query_expansion=True),
    TaskType.REWRITE: IntentConfig(task=TaskType.REWRITE, top_k=5, query_expansion=False),
    TaskType.UNKNOWN: IntentConfig(task=TaskType.UNKNOWN, top_k=0, query_expansion=False),
}

class IntentRouter:
    def __init__(self, llm_client):
        self.llm = llm_client

    def classify_intent(self, user_query: str) -> IntentConfig:
        system_prompt = (
            "Classify the user's request into one primary task type:\n"
            "summary, review, list_items, extract_fields, qa, compare, rewrite, action_items, unknown.\n"
            "Return ONLY the task name in lowercase."
        )
        
        try:
            # Fast classification call using your existing LLM provider
            task_str = self.llm.generate(system_prompt=system_prompt, user_prompt=user_query).strip().lower()
            task_enum = TaskType(task_str)
        except Exception:
            task_enum = TaskType.QA  # Fallback to QA
            
        return INTENT_STRATEGY_MAP.get(task_enum, INTENT_STRATEGY_MAP[TaskType.QA])
