"""Thin wrapper over the official ``groq`` SDK: config, retries, typed errors.

Callers depend on the small LLMClient protocol, not on Groq, so tests inject a fake
client and never touch the network. The API key is read from the environment / .env
(or passed in from Streamlit secrets) and is never logged or put into error messages.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

import groq

from starbucks_nutrition import config

logger = logging.getLogger(__name__)

KEY_ENV = "GROQ_API_KEY"
MODEL_ENV = "GROQ_MODEL"


# --------------------------------------------------------------------------------------
# Errors (the UI maps each to a friendly message)
# --------------------------------------------------------------------------------------
class LLMError(Exception):
    """Any failure talking to the LLM. Messages are safe to show (no key, no payload)."""


class LLMNotConfigured(LLMError):
    """No API key, or the key was rejected."""


class LLMRateLimited(LLMError):
    """Still rate-limited (429) after retrying."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMToolCallFailed(LLMError):
    """Groq 400 ``tool_use_failed``: the model produced an invalid tool call."""


class LLMBudgetExceeded(LLMError):
    """The per-session call cap was reached."""


# --------------------------------------------------------------------------------------
# Provider-neutral reply + protocol
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model (may be malformed)


@dataclass(frozen=True)
class ChatReply:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    model: str

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ChatReply: ...


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
def _load_dotenv() -> None:
    """Load the project's .env for local runs. Existing env vars win (override=False)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # optional at runtime: Streamlit Cloud uses secrets instead
        return
    load_dotenv(config.PROJECT_ROOT / ".env", override=False)


def resolve_setting(name: str, secrets: Mapping[str, Any] | None = None) -> str | None:
    """A setting from Streamlit secrets first, then the environment (incl. .env)."""
    if secrets is not None:
        value = secrets.get(name)
        if value:
            return str(value).strip()
    _load_dotenv()
    value = os.environ.get(name, "").strip()
    return value or None


def resolve_model(secrets: Mapping[str, Any] | None = None) -> str:
    return resolve_setting(MODEL_ENV, secrets) or config.LLM_DEFAULT_MODEL


# --------------------------------------------------------------------------------------
# Groq implementation
# --------------------------------------------------------------------------------------
def _retry_after(exc: groq.APIStatusError) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after", ""))
    except (TypeError, ValueError):
        return None


def _is_tool_use_failed(exc: groq.APIStatusError) -> bool:
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    return code == "tool_use_failed" or "tool_use_failed" in str(exc.message)


@dataclass
class GroqClient:
    """LLMClient backed by Groq, with backoff retries on 429 / 5xx / timeouts.

    The SDK's own retries are disabled (max_retries=0) so retry policy lives in one
    place and is testable: ``sdk`` and ``sleep`` can be injected.
    """

    api_key: str | None = None
    model: str = field(default_factory=resolve_model)
    temperature: float = config.LLM_TEMPERATURE
    max_tokens: int = config.LLM_MAX_TOKENS
    timeout_s: float = config.LLM_TIMEOUT_S
    max_attempts: int = config.LLM_MAX_ATTEMPTS
    sdk: Any = None
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if self.sdk is None:
            key = self.api_key or resolve_setting(KEY_ENV)
            if not key:
                raise LLMNotConfigured(
                    "No Groq API key configured. Set GROQ_API_KEY in .env or in "
                    ".streamlit/secrets.toml."
                )
            self.sdk = groq.Groq(api_key=key, max_retries=0, timeout=self.timeout_s)
        self.api_key = None  # don't keep the key on the object (repr, pickling, logs)

    def __repr__(self) -> str:  # never show the SDK/client internals
        return f"GroqClient(model={self.model!r})"

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        delay = retry_after if retry_after is not None else 2 ** (attempt - 1)
        return min(delay + random.uniform(0, 0.25), config.LLM_MAX_BACKOFF_S)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ChatReply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = tool_choice or "auto"
        if self.model.startswith(config.LLM_REASONING_MODEL_PREFIXES):
            # Keep the reasoning short and out of the reply (only the answer is shown).
            kwargs["reasoning_effort"] = config.LLM_REASONING_EFFORT
            kwargs["include_reasoning"] = False

        for attempt in range(1, self.max_attempts + 1):
            last = attempt == self.max_attempts
            try:
                return self._parse(self.sdk.chat.completions.create(**kwargs))
            except groq.RateLimitError as exc:
                wait = _retry_after(exc)
                logger.warning("Groq 429 (attempt %d/%d)", attempt, self.max_attempts)
                if last:
                    raise LLMRateLimited(
                        "The Groq rate limit was reached. Please try again shortly.", wait
                    ) from None
                self.sleep(self._backoff(attempt, wait))
            except groq.AuthenticationError:
                raise LLMNotConfigured("The Groq API key was rejected.") from None
            except groq.BadRequestError as exc:
                if _is_tool_use_failed(exc):
                    raise LLMToolCallFailed("The model produced an invalid tool call.") from None
                logger.warning("Groq 400: %s", exc.message)
                raise LLMError("The request to the language model was rejected.") from None
            except groq.NotFoundError:
                raise LLMError(
                    f"The model {self.model!r} is not available for this API key. Set "
                    "GROQ_MODEL to a model listed at console.groq.com/docs/models."
                ) from None
            except groq.APIStatusError as exc:
                if exc.status_code < 500 or last:
                    logger.warning("Groq HTTP %s", exc.status_code)
                    raise LLMError(
                        f"The language model service returned an error ({exc.status_code})."
                    ) from None
                self.sleep(self._backoff(attempt, _retry_after(exc)))
            except groq.APIConnectionError:  # includes APITimeoutError
                logger.warning("Groq connection error/timeout (attempt %d)", attempt)
                if last:
                    raise LLMError("Could not reach the language model service.") from None
                self.sleep(self._backoff(attempt, None))
        raise LLMError("Unreachable")  # pragma: no cover

    @staticmethod
    def _parse(response: Any) -> ChatReply:
        message = response.choices[0].message
        calls = tuple(
            ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments or "{}")
            for c in (message.tool_calls or [])
        )
        usage = getattr(response, "usage", None)
        return ChatReply(
            content=message.content,
            tool_calls=calls,
            model=getattr(response, "model", "") or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


# --------------------------------------------------------------------------------------
# Per-session call cap
# --------------------------------------------------------------------------------------
@dataclass
class CallBudget:
    limit: int = config.LLM_MAX_REQUESTS_PER_SESSION
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)


@dataclass
class BudgetedClient:
    """Wraps any LLMClient and refuses requests once the budget is spent.

    Each chat() call costs one unit, unless it runs inside ``request()``: a Q&A question
    may take several tool rounds and retries, but the user asked one thing, so the whole
    question costs one unit.
    """

    inner: LLMClient
    budget: CallBudget
    _in_request: bool = field(default=False, init=False, repr=False)

    def _charge(self) -> None:
        if self.budget.remaining <= 0:
            raise LLMBudgetExceeded(
                f"This session has used its {self.budget.limit} AI requests. Reload the "
                "page later to start a new session."
            )
        self.budget.used += 1

    @contextmanager
    def request(self) -> Iterator[BudgetedClient]:
        """Charge one unit now; chat() calls inside the block are free."""
        self._charge()
        self._in_request = True
        try:
            yield self
        finally:
            self._in_request = False

    @property
    def model(self) -> str:
        return self.inner.model

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ChatReply:
        if not self._in_request:
            self._charge()
        return self.inner.chat(messages, tools, tool_choice)
