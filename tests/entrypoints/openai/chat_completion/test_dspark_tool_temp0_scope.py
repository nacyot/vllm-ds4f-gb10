# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSPARK_TOOL_TEMP0_SCOPE: how far the tool-turn temperature 0 reaches."""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.v1.core.sched.dspark_loop_break import (
    DSPARK_ANSWER_TEMP_KEY,
    DSPARK_REASONING_TEMP_KEY,
    DSPARK_SCOPE_KEY,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _request(**kwargs) -> ChatCompletionRequest:
    body = {
        "model": "dsv4",
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(kwargs)
    return ChatCompletionRequest.model_validate(body)


def _params(request: ChatCompletionRequest, defaults: dict | None = None):
    return request.to_sampling_params(
        max_tokens=16,
        default_sampling_params=defaults if defaults is not None else {},
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DSPARK_TOOL_TEMP0", raising=False)
    monkeypatch.delenv("DSPARK_TOOL_TEMP0_SCOPE", raising=False)


def test_off_by_default():
    params = _params(_request(tools=[WEATHER_TOOL], temperature=0.7))
    assert params.temperature == 0.7
    assert not (params.extra_args or {})


def test_no_tools_is_untouched(monkeypatch):
    monkeypatch.setenv("DSPARK_TOOL_TEMP0", "1")
    params = _params(_request(temperature=0.7))
    assert params.temperature == 0.7
    assert not (params.extra_args or {})


def test_default_scope_is_dsml(monkeypatch):
    monkeypatch.setenv("DSPARK_TOOL_TEMP0", "1")
    params = _params(_request(tools=[WEATHER_TOOL], temperature=0.7))
    # The client's own temperature survives reasoning AND answer prose; only
    # the DSML tool-call block goes greedy, and the tracker decides when.
    assert params.temperature == 0.7
    assert params.extra_args[DSPARK_SCOPE_KEY] == "dsml"
    assert params.extra_args[DSPARK_ANSWER_TEMP_KEY] == 0.0
    assert params.extra_args[DSPARK_REASONING_TEMP_KEY] == 0.7


@pytest.mark.parametrize("scope", ["dsml", "DSML", " dsml "])
def test_dsml_scope_spellings(monkeypatch, scope):
    monkeypatch.setenv("DSPARK_TOOL_TEMP0", "1")
    monkeypatch.setenv("DSPARK_TOOL_TEMP0_SCOPE", scope)
    params = _params(_request(tools=[WEATHER_TOOL], temperature=0.7))
    assert params.temperature == 0.7
    assert params.extra_args[DSPARK_SCOPE_KEY] == "dsml"


def test_answer_scope_still_works(monkeypatch):
    monkeypatch.setenv("DSPARK_TOOL_TEMP0", "1")
    monkeypatch.setenv("DSPARK_TOOL_TEMP0_SCOPE", "answer")
    params = _params(_request(tools=[WEATHER_TOOL], temperature=0.7))
    assert params.temperature == 0.7
    assert params.extra_args[DSPARK_SCOPE_KEY] == "answer"
    assert params.extra_args[DSPARK_ANSWER_TEMP_KEY] == 0.0


def test_model_default_temperature_is_used_when_unset(monkeypatch):
    monkeypatch.setenv("DSPARK_TOOL_TEMP0", "1")
    params = _params(_request(tools=[WEATHER_TOOL]), {"temperature": 1.0})
    assert params.temperature == 1.0
    assert params.extra_args[DSPARK_REASONING_TEMP_KEY] == 1.0


@pytest.mark.parametrize("scope", ["request", "REQUEST", " request ", "nonsense"])
def test_request_scope_is_the_rollback(monkeypatch, scope):
    monkeypatch.setenv("DSPARK_TOOL_TEMP0", "1")
    monkeypatch.setenv("DSPARK_TOOL_TEMP0_SCOPE", scope)
    params = _params(_request(tools=[WEATHER_TOOL], temperature=0.7))
    assert params.temperature == 0.0
    assert DSPARK_ANSWER_TEMP_KEY not in (params.extra_args or {})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chat_template_kwargs": {"thinking": False}},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"reasoning_effort": "none"},
    ],
)
def test_answer_scope_with_thinking_off_keeps_whole_request_zero(monkeypatch, kwargs):
    monkeypatch.setenv("DSPARK_TOOL_TEMP0", "1")
    monkeypatch.setenv("DSPARK_TOOL_TEMP0_SCOPE", "answer")
    params = _params(_request(tools=[WEATHER_TOOL], temperature=0.7, **kwargs))
    assert params.temperature == 0.0
    assert DSPARK_ANSWER_TEMP_KEY not in (params.extra_args or {})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chat_template_kwargs": {"thinking": False}},
        {"reasoning_effort": "none"},
        {"chat_template_kwargs": {"thinking": True}},
        {},
    ],
)
def test_dsml_scope_does_not_care_about_thinking(monkeypatch, kwargs):
    monkeypatch.setenv("DSPARK_TOOL_TEMP0", "1")
    params = _params(_request(tools=[WEATHER_TOOL], temperature=0.7, **kwargs))
    assert params.temperature == 0.7
    assert params.extra_args[DSPARK_SCOPE_KEY] == "dsml"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chat_template_kwargs": {"thinking": True}},
        {"chat_template_kwargs": {"enable_thinking": True}},
        {},  # unstated: assume thinking is on, the tracker decides
    ],
)
def test_answer_scope_with_thinking_on_or_unstated_splits(monkeypatch, kwargs):
    monkeypatch.setenv("DSPARK_TOOL_TEMP0", "1")
    monkeypatch.setenv("DSPARK_TOOL_TEMP0_SCOPE", "answer")
    params = _params(_request(tools=[WEATHER_TOOL], temperature=0.7, **kwargs))
    assert params.temperature == 0.7
    assert params.extra_args[DSPARK_ANSWER_TEMP_KEY] == 0.0


def test_kv_transfer_extra_args_are_preserved(monkeypatch):
    monkeypatch.setenv("DSPARK_TOOL_TEMP0", "1")
    params = _params(
        _request(
            tools=[WEATHER_TOOL],
            temperature=0.7,
            kv_transfer_params={"remote": "x"},
        )
    )
    assert params.extra_args["kv_transfer_params"] == {"remote": "x"}
    assert params.extra_args[DSPARK_ANSWER_TEMP_KEY] == 0.0
