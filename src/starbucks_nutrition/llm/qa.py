"""Bonus: natural-language questions answered through tool calling.

Why tool calling: the model never sees the table (no context bloat, no misreading of
rows) and never writes code (no exec of model output). It picks a read-only tool, the
tool runs tested pandas code, and the model phrases the result. Every number in the
answer is then checked against ALL tool results of the exchange (grounding).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from starbucks_nutrition.llm.client import ChatReply, LLMClient, LLMToolCallFailed
from starbucks_nutrition.llm.grounding import GroundingResult, verify_numbers
from starbucks_nutrition.llm.prompts import QA_SYSTEM
from starbucks_nutrition.llm.tools import TOOL_SCHEMAS, ToolExecutor

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500
MAX_TOOL_ROUNDS = 4
HISTORY_TURNS = 3
TRACE_RESULT_CHARS = 1500
LOOKUP_FAILED_ANSWER = (
    "I couldn't complete that lookup. Please rephrase the question, for example naming "
    "the metric (calories, fat, carbs, protein, fiber, sodium) and drinks or food."
)
ROUNDS_EXHAUSTED_ANSWER = (
    "I couldn't finish that lookup within the allowed number of steps. Please ask a "
    "narrower question."
)


@dataclass(frozen=True)
class QAResult:
    question: str
    answer: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    grounding: GroundingResult = field(default_factory=GroundingResult)
    model: str = ""
    total_tokens: int = 0
    latency_s: float = 0.0
    truncated_question: bool = False


class _LookupFailed(Exception):
    pass


def _valid_arguments(reply: ChatReply) -> bool:
    for call in reply.tool_calls:
        try:
            if not isinstance(json.loads(call.arguments or "{}"), dict):
                return False
        except json.JSONDecodeError:
            return False
    return True


def _chat_once_retried(
    client: LLMClient, messages: list[dict[str, Any]], tool_choice: str
) -> ChatReply:
    """One model turn; retried ONCE on a failed/malformed tool call, then gives up."""
    for attempt in (1, 2):
        try:
            reply = client.chat(messages, TOOL_SCHEMAS, tool_choice)
        except LLMToolCallFailed:
            logger.warning("tool_use_failed (attempt %d)", attempt)
            continue
        if _valid_arguments(reply):
            return reply
        logger.warning("malformed tool-call arguments (attempt %d)", attempt)
    raise _LookupFailed


def _history_messages(history: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for question, answer in list(history)[-HISTORY_TURNS:]:
        messages += [
            {"role": "user", "content": question[:MAX_QUESTION_CHARS]},
            {"role": "assistant", "content": answer},
        ]
    return messages


def answer_question(
    client: LLMClient,
    question: str,
    menu: pd.DataFrame,
    history: Sequence[tuple[str, str]] = (),
    include_unreliable: bool = False,
) -> QAResult:
    """Answer ``question`` about ``menu`` (sample or uploaded) via read-only tools.

    ``history`` holds the last (question, answer) pairs so follow-ups like "and for
    food?" work. LLM errors other than a failed tool call (rate limit, no key, budget)
    propagate for the UI to explain.
    """
    question = question.strip()
    truncated = len(question) > MAX_QUESTION_CHARS
    question = question[:MAX_QUESTION_CHARS]
    executor = ToolExecutor(menu, include_unreliable)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": QA_SYSTEM},
        *_history_messages(history),
        {"role": "user", "content": question},
    ]
    trace: list[dict[str, Any]] = []
    tokens = 0
    model = client.model
    start = time.perf_counter()
    answer = ROUNDS_EXHAUSTED_ANSWER

    try:
        for round_no in range(1, MAX_TOOL_ROUNDS + 2):
            force_answer = round_no > MAX_TOOL_ROUNDS
            reply = _chat_once_retried(client, messages, "none" if force_answer else "auto")
            tokens += reply.total_tokens
            model = reply.model or model
            if not reply.tool_calls:
                answer = (reply.content or "").strip() or ROUNDS_EXHAUSTED_ANSWER
                break
            if force_answer:  # model ignored tool_choice="none"
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                        }
                        for c in reply.tool_calls
                    ],
                }
            )
            for call in reply.tool_calls:
                result = executor.execute(call.name, call.arguments)
                trace.append(
                    {
                        "round": round_no,
                        "tool": call.name,
                        "arguments": call.arguments,
                        "result": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(result, ensure_ascii=False, allow_nan=False),
                    }
                )
    except _LookupFailed:
        answer = LOOKUP_FAILED_ANSWER

    results = [step["result"] for step in trace]
    return QAResult(
        question=question,
        answer=answer,
        trace=trace,
        grounding=verify_numbers(answer, *results)
        if answer
        not in (
            LOOKUP_FAILED_ANSWER,
            ROUNDS_EXHAUSTED_ANSWER,
        )
        else GroundingResult(),
        model=model,
        total_tokens=tokens,
        latency_s=round(time.perf_counter() - start, 2),
        truncated_question=truncated,
    )


def trace_for_display(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trace with each result as truncated JSON text, for the "How I got this" expander."""
    shown = []
    for step in trace:
        text = json.dumps(step["result"], ensure_ascii=False, indent=1)
        if len(text) > TRACE_RESULT_CHARS:
            text = text[:TRACE_RESULT_CHARS] + "\n… (truncated)"
        shown.append({**step, "result": text})
    return shown
