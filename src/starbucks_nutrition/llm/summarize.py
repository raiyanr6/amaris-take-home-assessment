"""Part 2: natural-language summary of the computed facts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from starbucks_nutrition.llm.client import LLMClient
from starbucks_nutrition.llm.grounding import GroundingResult, verify_numbers
from starbucks_nutrition.llm.prompts import FOCUSES, PROMPT_VERSION, SUMMARY_SYSTEM


@dataclass(frozen=True)
class SummaryResult:
    text: str
    focus: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    grounding: GroundingResult
    prompt_version: str = PROMPT_VERSION


def focus_rankings(focus: str) -> tuple[str, ...]:
    """Ranking lists to expand in build_facts for this focus."""
    return FOCUSES[focus][1]


def system_prompt(focus: str) -> str:
    if focus not in FOCUSES:
        raise ValueError(f"Unknown focus {focus!r}; choose from {list(FOCUSES)}")
    return SUMMARY_SYSTEM.format(focus_instruction=FOCUSES[focus][0])


def build_messages(facts: dict[str, Any], focus: str) -> list[dict[str, str]]:
    facts_json = json.dumps(facts, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return [
        {"role": "system", "content": system_prompt(focus)},
        {"role": "user", "content": f"FACTS:\n```json\n{facts_json}\n```"},
    ]


def generate_summary(
    client: LLMClient, facts: dict[str, Any], focus: str = "overview"
) -> SummaryResult:
    """Ask the model to narrate ``facts``; the output is grounding-checked against them."""
    messages = build_messages(facts, focus)
    start = time.perf_counter()
    reply = client.chat(messages)
    latency = time.perf_counter() - start
    text = (reply.content or "").strip()
    return SummaryResult(
        text=text,
        focus=focus,
        model=reply.model or client.model,
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        latency_s=round(latency, 2),
        grounding=verify_numbers(text, facts),
    )
