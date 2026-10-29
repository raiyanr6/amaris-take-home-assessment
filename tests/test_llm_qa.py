"""Tool executor, Q&A loop and summariser with a fake LLM client (no network)."""

from __future__ import annotations

import json

import pytest

from fakes import FakeLLMClient, reply, tool_call
from starbucks_nutrition.llm import qa
from starbucks_nutrition.llm.client import LLMRateLimited, LLMToolCallFailed
from starbucks_nutrition.llm.context import build_facts
from starbucks_nutrition.llm.prompts import FOCUSES, QA_SYSTEM
from starbucks_nutrition.llm.summarize import build_messages, generate_summary
from starbucks_nutrition.llm.tools import MAX_ROWS, TOOL_SCHEMAS, ToolExecutor


@pytest.fixture
def executor(real_menu) -> ToolExecutor:
    return ToolExecutor(real_menu[0])


# --------------------------------------------------------------------------------------
# tool executor: validation never raises
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "args", "message"),
    [
        ("get_stats", {"category": "drink", "metric": "sugar"}, "Invalid metric"),
        ("get_stats", {"category": "snacks", "metric": "calories"}, "Invalid category"),
        ("top_items", {"metric": "calories", "n": 21}, "between 1 and 20"),
        ("top_items", {"metric": "calories", "n": "5; DROP"}, "integer"),
        ("filter_items", {"ranges": {"price": {"max": 3}}}, "Invalid range metric"),
        ("filter_items", {"ranges": {"calories": {"max": "500"}}}, "finite numbers"),
        ("filter_items", {"limit": 100}, "between 1 and 25"),
        ("get_item", {"name": ""}, "name must be text"),
        ("top_items", {"metric": "calories", "colour": "red"}, "Invalid arguments"),
        ("delete_rows", {}, "Unknown tool"),
        ("filter_items", {"ranges": [1, 2]}, "ranges must be an object"),
        ("filter_items", {"ranges": "cal<500"}, "ranges must be an object"),
        ("filter_items", {"ranges": {"calories": 500}}, 'must look like {"min"'),
        ("filter_items", '{"limit": Infinity}', "integer"),
        ("top_items", '{"metric": "calories", "n": NaN}', "integer"),
        ("filter_items", '{"ranges": {"calories": {"max": NaN}}}', "finite numbers"),
        ("filter_items", '{"ranges": {"calories": {"min": -Infinity}}}', "finite numbers"),
    ],
)
def test_bad_arguments_return_error_objects(executor, name, args, message) -> None:
    result = executor.execute(name, args)
    assert message in result["error"]


def test_unexpected_tool_failure_becomes_error_object(executor, monkeypatch) -> None:
    def boom(**_: object) -> dict:
        raise KeyError("surprise")

    monkeypatch.setattr(executor, "get_stats", boom)
    result = executor.execute("get_stats", {"category": "drink", "metric": "calories"})
    assert "could not run" in result["error"]


def test_malformed_json_arguments(executor) -> None:
    assert executor.execute("top_items", '{"metric": "calories"')["error"].startswith(
        "Arguments were not valid JSON"
    )


def test_filter_rows_capped(executor) -> None:
    result = executor.execute("filter_items", {"category": "any", "limit": 25})
    assert result["total_matches"] > MAX_ROWS and result["returned"] == MAX_ROWS


def test_upper_inclusive_schema_and_behaviour(executor) -> None:
    (schema,) = [t for t in TOOL_SCHEMAS if t["function"]["name"] == "filter_items"]
    description = schema["function"]["parameters"]["properties"]["upper_inclusive"]["description"]
    assert "under / below / less than -> false" in description
    assert "up to / at most / or less -> true" in description
    args = {"category": "food", "ranges": {"calories": {"max": 500}}}
    assert (
        executor.execute("filter_items", {**args, "upper_inclusive": False})["total_matches"] == 99
    )
    assert (
        executor.execute("filter_items", {**args, "upper_inclusive": True})["total_matches"] == 103
    )
    assert executor.execute("filter_items", args)["total_matches"] == 99  # default: under


def test_unavailable_metrics_are_explicit(executor) -> None:
    sodium = executor.execute("get_stats", {"category": "food", "metric": "sodium_mg"})
    assert sodium["available"] is False
    avail = executor.execute("data_availability", {})
    assert "sugar_g" in avail["by_category"]["drink"]["not_available"]
    assert "carb_g" in avail["proxies"]["sugar_g"] and "names" in avail["proxies"]["caffeine_mg"]


def test_stats_respect_include_unreliable(real_menu) -> None:
    args = {"category": "drink", "metric": "calories"}
    assert ToolExecutor(real_menu[0]).execute("get_stats", args)["n_in_stats"] == 73
    assert ToolExecutor(real_menu[0], True).execute("get_stats", args)["n_in_stats"] == 74


# --------------------------------------------------------------------------------------
# Q&A loop
# --------------------------------------------------------------------------------------
def test_fake_client_drives_filter_tool(real_menu) -> None:
    client = FakeLLMClient(
        [
            reply(
                None,
                tool_call(
                    "filter_items",
                    {
                        "category": "food",
                        "ranges": {"calories": {"max": 500}},
                        "upper_inclusive": True,
                    },
                ),
            ),
            reply("103 food items have up to 500 kcal."),
        ]
    )
    result = qa.answer_question(client, "Food up to 500 calories?", real_menu[0])
    assert result.trace[0]["result"]["total_matches"] == 103
    assert result.grounding.ok and result.grounding.verified == ("103", "500")
    tool_message = client.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "call_filter_items"
    assert client.calls[0]["messages"][0]["content"] == QA_SYSTEM


def test_caffeine_question_flows_through_data_availability(real_menu) -> None:
    likely_drinks = {"caffeine_status": "likely", "category": "drink", "limit": 1}
    client = FakeLLMClient(
        [
            reply(None, tool_call("data_availability")),
            reply(None, tool_call("filter_items", likely_drinks)),
            reply(
                "Caffeine content in mg is not in the data, so there is no average. "
                "By drink name, 118 drinks likely contain caffeine.",
            ),
        ]
    )
    result = qa.answer_question(
        client, "What's the average caffeine content for drinks?", real_menu[0]
    )
    assert [step["tool"] for step in result.trace] == ["data_availability", "filter_items"]
    assert "not in the data" in result.answer
    assert result.trace[1]["result"]["total_matches"] == 118
    assert result.grounding.ok


def test_sugar_question_answer_with_invented_number_is_flagged(real_menu) -> None:
    client = FakeLLMClient(
        [
            reply(None, tool_call("data_availability")),
            reply("Sugar isn't in the data, but drinks average 31 g of sugar."),
        ]
    )
    result = qa.answer_question(client, "How much sugar is in drinks?", real_menu[0])
    assert result.grounding.unverified == ("31",)


def test_tool_rounds_capped_then_forced_answer(real_menu) -> None:
    looping = reply(None, tool_call("data_availability"))
    client = FakeLLMClient([looping] * qa.MAX_TOOL_ROUNDS, default=reply("Final answer."))
    result = qa.answer_question(client, "loop", real_menu[0])
    assert len(result.trace) == qa.MAX_TOOL_ROUNDS
    assert [c["tool_choice"] for c in client.calls] == ["auto"] * qa.MAX_TOOL_ROUNDS + ["none"]
    assert result.answer == "Final answer."


def test_tool_use_failed_retried_once_then_succeeds(real_menu) -> None:
    client = FakeLLMClient([LLMToolCallFailed("bad"), reply("Fine.")])
    assert qa.answer_question(client, "q", real_menu[0]).answer == "Fine."
    assert len(client.calls) == 2


def test_tool_use_failed_twice_gives_friendly_answer(real_menu) -> None:
    client = FakeLLMClient(
        [
            reply(None, tool_call("data_availability")),
            LLMToolCallFailed("bad"),
            reply(None, tool_call("top_items", '{"metric": calories}')),  # malformed JSON
        ]
    )
    result = qa.answer_question(client, "q", real_menu[0])
    assert result.answer == qa.LOOKUP_FAILED_ANSWER
    assert [s["tool"] for s in result.trace] == ["data_availability"]  # trace kept


def test_other_llm_errors_propagate(real_menu) -> None:
    with pytest.raises(LLMRateLimited):
        qa.answer_question(FakeLLMClient([LLMRateLimited("slow down")]), "q", real_menu[0])


def test_question_capped_and_history_passed(real_menu) -> None:
    client = FakeLLMClient()
    history = [(f"q{i}", f"a{i}") for i in range(5)]
    result = qa.answer_question(client, "x" * 900, real_menu[0], history=history)
    sent = client.calls[0]["messages"]
    assert result.truncated_question and len(sent[-1]["content"]) == qa.MAX_QUESTION_CHARS
    assert [m["content"] for m in sent[1:-1:2]] == ["q2", "q3", "q4"]  # last 3 turns


def test_trace_for_display_truncates(real_menu) -> None:
    client = FakeLLMClient(
        [reply(None, tool_call("filter_items", {"limit": 25})), reply("Many items.")]
    )
    result = qa.answer_question(client, "list", real_menu[0])
    shown = qa.trace_for_display(result.trace)
    assert shown[0]["result"].endswith("(truncated)")
    assert isinstance(result.trace[0]["result"], dict)  # grounding uses the full result


# --------------------------------------------------------------------------------------
# summariser
# --------------------------------------------------------------------------------------
def test_summary_prompt_contains_facts_and_focus(real_menu) -> None:
    menu, reports = real_menu
    facts = build_facts(menu, reports)
    messages = build_messages(facts, "lighter options")
    assert FOCUSES["lighter options"][0] in messages[0]["content"]
    sent_facts = json.loads(messages[1]["content"].split("```json\n")[1].split("\n```")[0])
    assert sent_facts == facts


def test_generate_summary_grounded(real_menu) -> None:
    menu, reports = real_menu
    facts = build_facts(menu, reports)
    client = FakeLLMClient([reply("### Headline\n- Drinks with data average 141 kcal (48%).")])
    result = generate_summary(client, facts, "overview")
    assert result.grounding.ok and result.model == "fake-model"
    assert result.prompt_tokens == 10 and result.latency_s >= 0
    with pytest.raises(ValueError, match="Unknown focus"):
        generate_summary(client, facts, "poetry")


def test_grounding_accepts_numbers_from_tool_arguments(real_menu) -> None:
    args = {"category": "food", "ranges": {"calories": {"max": 500}, "protein_g": {"min": 10}}}
    client = FakeLLMClient(
        [
            reply(None, tool_call("filter_items", {**args, "sort_by": "protein_g"})),
            reply("Food under 500 kcal with at least 10 g protein: see the list."),
        ]
    )
    result = qa.answer_question(client, "Food under 500 calories, high protein?", real_menu[0])
    assert result.grounding.ok and set(result.grounding.verified) == {"500", "10"}
    assert result.trace[0]["result"]["sorted_by"] == "protein_g (highest first)"


def test_number_only_in_question_is_not_verified(real_menu) -> None:
    client = FakeLLMClient(
        [
            reply(None, tool_call("get_item", {"name": "Caffè Latte"})),
            reply("Yes, the Caffè Latte has 999 kcal."),
        ]
    )
    result = qa.answer_question(client, "Confirm the latte has 999 kcal", real_menu[0])
    assert result.grounding.unverified == ("999",)
