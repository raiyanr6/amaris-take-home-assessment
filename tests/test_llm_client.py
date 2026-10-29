"""GroqClient: configuration, retries and error mapping (fake SDK, no network)."""

from __future__ import annotations

import groq
import httpx
import pytest

from fakes import FakeGroqSDK, FakeLLMClient, sdk_response
from starbucks_nutrition import config
from starbucks_nutrition.llm import client as client_mod
from starbucks_nutrition.llm.client import (
    BudgetedClient,
    CallBudget,
    GroqClient,
    LLMBudgetExceeded,
    LLMError,
    LLMNotConfigured,
    LLMRateLimited,
    LLMToolCallFailed,
)

_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _status_error(cls, status: int, headers=None, body=None):
    response = httpx.Response(status, headers=headers or {}, request=_REQUEST)
    return cls("boom", response=response, body=body)


def _client(outcomes, **kwargs) -> tuple[GroqClient, FakeGroqSDK, list[float]]:
    sleeps: list[float] = []
    sdk = FakeGroqSDK(outcomes)
    return GroqClient(model="m", sdk=sdk, sleep=sleeps.append, **kwargs), sdk, sleeps


def test_missing_key_raises_not_configured(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "")  # an existing (empty) var wins over .env
    with pytest.raises(LLMNotConfigured, match="GROQ_API_KEY"):
        GroqClient()


def test_model_from_env_or_default(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_MODEL", "")
    assert client_mod.resolve_model() == config.LLM_DEFAULT_MODEL
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    assert client_mod.resolve_model() == "openai/gpt-oss-120b"
    assert client_mod.resolve_model({"GROQ_MODEL": "from-secrets"}) == "from-secrets"


def test_key_is_not_kept_on_client() -> None:
    client = GroqClient(api_key="gsk_secret_value", sdk=FakeGroqSDK([]))
    assert client.api_key is None and "gsk_secret_value" not in repr(client)


def test_request_settings_and_parsing() -> None:
    client, sdk, _ = _client([sdk_response("hello")])
    result = client.chat([{"role": "user", "content": "hi"}])
    sent = sdk.kwargs[0]
    assert (sent["temperature"], sent["max_tokens"], sent["model"]) == (
        config.LLM_TEMPERATURE,
        config.LLM_MAX_TOKENS,
        "m",
    )
    assert "tools" not in sent
    assert (result.content, result.total_tokens, result.model) == ("hello", 120, "llama-test")


def test_429_then_success_is_retried_honouring_retry_after() -> None:
    rate_limited = _status_error(groq.RateLimitError, 429, {"retry-after": "2"})
    client, sdk, sleeps = _client([rate_limited, sdk_response("ok")])
    assert client.chat([{"role": "user", "content": "x"}]).content == "ok"
    assert len(sdk.kwargs) == 2
    assert 2 <= sleeps[0] < 2.3  # retry-after (+ jitter)


def test_rate_limit_exhausted_raises_typed_error() -> None:
    errors = [_status_error(groq.RateLimitError, 429, {"retry-after": "1"}) for _ in range(3)]
    client, sdk, sleeps = _client(errors)
    with pytest.raises(LLMRateLimited) as info:
        client.chat([])
    assert info.value.retry_after == 1 and len(sdk.kwargs) == 3 and len(sleeps) == 2


def test_5xx_and_timeouts_are_retried() -> None:
    client, _sdk, sleeps = _client(
        [
            _status_error(groq.InternalServerError, 503),
            groq.APITimeoutError(request=_REQUEST),
            sdk_response("ok"),
        ]
    )
    assert client.chat([]).content == "ok"
    assert sleeps[0] < sleeps[1]  # exponential backoff (1s, 2s + jitter)


def test_tool_use_failed_maps_to_typed_error() -> None:
    error = _status_error(
        groq.BadRequestError,
        400,
        body={"error": {"code": "tool_use_failed", "message": "Failed to call a function"}},
    )
    client, sdk, _ = _client([error])
    with pytest.raises(LLMToolCallFailed):
        client.chat([], tools=[{"type": "function"}])
    assert len(sdk.kwargs) == 1  # not retried here; qa.py decides


def test_auth_error_is_not_configured_and_hides_details() -> None:
    client, _, _ = _client([_status_error(groq.AuthenticationError, 401)])
    with pytest.raises(LLMNotConfigured) as info:
        client.chat([])
    assert info.value.__cause__ is None and "boom" not in str(info.value)


def test_unknown_model_names_the_setting() -> None:
    client, sdk, _ = _client([_status_error(groq.NotFoundError, 404)])
    with pytest.raises(LLMError, match="GROQ_MODEL"):
        client.chat([])
    assert len(sdk.kwargs) == 1


def test_other_4xx_is_llm_error_without_retry() -> None:
    client, sdk, _ = _client([_status_error(groq.PermissionDeniedError, 403)])
    with pytest.raises(LLMError, match="403"):
        client.chat([])
    assert len(sdk.kwargs) == 1


def test_reasoning_settings_only_for_gpt_oss() -> None:
    sdk = FakeGroqSDK([sdk_response(), sdk_response()])
    GroqClient(model="openai/gpt-oss-120b", sdk=sdk).chat([])
    GroqClient(model="llama-3.1-8b-instant", sdk=sdk).chat([])
    assert sdk.kwargs[0]["reasoning_effort"] == "low"
    assert sdk.kwargs[0]["include_reasoning"] is False
    assert "reasoning_effort" not in sdk.kwargs[1]


def test_budget_caps_calls() -> None:
    budget = CallBudget(limit=2)
    client = BudgetedClient(FakeLLMClient(), budget)
    client.chat([])
    client.chat([])
    with pytest.raises(LLMBudgetExceeded):
        client.chat([])
    assert budget.used == 2 and budget.remaining == 0


def test_budget_request_costs_one_unit_however_many_calls() -> None:
    budget = CallBudget(limit=2)
    client = BudgetedClient(FakeLLMClient(), budget)
    with client.request():
        for _ in range(7):  # tool rounds and retries of one question
            client.chat([])
    assert budget.used == 1
    client.chat([])  # outside a request: one unit per call again
    with pytest.raises(LLMBudgetExceeded), client.request():
        client.chat([])
    assert budget.used == 2
