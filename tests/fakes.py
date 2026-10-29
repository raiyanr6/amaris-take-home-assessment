"""Test doubles for the LLM layer: no network in tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from starbucks_nutrition.llm.client import ChatReply, ToolCall


def tool_call(name: str, arguments: Mapping[str, Any] | str = "{}", call_id: str = "") -> ToolCall:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(id=call_id or f"call_{name}", name=name, arguments=args)


def reply(content: str | None = None, *calls: ToolCall) -> ChatReply:
    return ChatReply(
        content=content, tool_calls=calls, model="fake-model", prompt_tokens=10, completion_tokens=5
    )


@dataclass
class FakeLLMClient:
    """Returns scripted replies (or raises scripted exceptions) in order; records calls.

    When the script runs out it keeps answering with ``default``.
    """

    script: list[ChatReply | Exception] = field(default_factory=list)
    default: ChatReply = field(default_factory=lambda: reply("Fake answer."))
    model: str = "fake-model"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ChatReply:
        self.calls.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "tool_choice": tool_choice}
        )
        if not self.script:
            return self.default
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def sdk_response(content: str = "ok", tool_calls: list[Any] | None = None) -> Any:
    """An object shaped like groq's ChatCompletion, for GroqClient._parse."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        model="llama-test",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


@dataclass
class FakeGroqSDK:
    """Stands in for groq.Groq: chat.completions.create pops scripted outcomes."""

    outcomes: list[Any]
    kwargs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def chat(self) -> Any:
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
