"""Groq LLM features: summaries of computed facts and tool-calling Q&A.

Core rule: pandas computes every number (analysis/filters); the model only narrates
numbers it is given and chooses which read-only tool to call. It never sees raw CSVs and
never executes code. grounding.verify_numbers checks the output against the inputs.
"""

from starbucks_nutrition.llm.client import (
    ChatReply,
    GroqClient,
    LLMBudgetExceeded,
    LLMClient,
    LLMError,
    LLMNotConfigured,
    LLMRateLimited,
    LLMToolCallFailed,
)

__all__ = [
    "ChatReply",
    "GroqClient",
    "LLMBudgetExceeded",
    "LLMClient",
    "LLMError",
    "LLMNotConfigured",
    "LLMRateLimited",
    "LLMToolCallFailed",
]
