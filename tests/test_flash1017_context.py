# SPDX-License-Identifier: Apache-2.0
"""Total-context admission contracts without model allocation or inference."""

import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from starlette.requests import Request as HTTPRequest

from omlx.engine.vlm import VLMBatchedEngine
from omlx.exceptions import ContextWindowExceededError, InvalidRequestError
from omlx.model_settings import ModelSettings
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


def scheduler(limit=10):
    instance = Scheduler.__new__(Scheduler)
    instance.config = SchedulerConfig(max_num_seqs=1, max_total_tokens=limit)
    instance.requests = {}
    instance.waiting = deque()
    instance.tokenizer = SimpleNamespace(encode=lambda prompt: list(range(len(prompt))))
    instance._resolve_generation_prompt_start = Mock()
    instance.moe_offload_stats = None
    instance.block_aware_cache = None
    instance.preflight_or_raise = Mock()
    return instance


def request(prompt, budget, **kwargs):
    return Request(
        request_id="fixture",
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=budget),
        **kwargs,
    )


def test_exact_total_budget_is_admitted_without_changing_output_budget():
    instance = scheduler()
    submitted = request("12345678", 2)
    instance.add_request(submitted)
    assert instance.requests == {"fixture": submitted}
    assert list(instance.waiting) == [submitted]
    assert submitted.sampling_params.max_tokens == 2
    instance.preflight_or_raise.assert_called_once()


@pytest.mark.parametrize(
    "prompt,budget", [("123456789", 2), ([1] * 10, 1), ([1], 131073)]
)
def test_total_overflow_refused_before_admission_or_memory_work(prompt, budget):
    instance = scheduler()
    with pytest.raises(ContextWindowExceededError) as caught:
        instance.add_request(request(prompt, budget))
    assert caught.value.prompt_tokens == len(prompt)
    assert caught.value.output_tokens == budget
    assert caught.value.limit == 10
    assert instance.requests == {} and not instance.waiting
    instance._resolve_generation_prompt_start.assert_not_called()
    instance.preflight_or_raise.assert_not_called()


def test_actual_full_token_ids_count_even_with_cached_prefix_and_stale_counter():
    instance = scheduler()
    submitted = request("ignored", 2, prompt_token_ids=[1] * 9)
    submitted.num_prompt_tokens = 1
    submitted.cached_tokens = 9
    submitted.remaining_tokens = [1]
    with pytest.raises(ContextWindowExceededError) as caught:
        instance.add_request(submitted)
    assert caught.value.prompt_tokens == 9
    assert instance.requests == {} and not instance.waiting


def test_unconfigured_scheduler_retains_existing_budget_behavior():
    assert SchedulerConfig().max_total_tokens is None
    instance = scheduler(None)
    submitted = request([1] * 20, 131073)
    instance.add_request(submitted)
    assert instance.requests["fixture"] is submitted
    assert submitted.sampling_params.max_tokens == 131073


@pytest.mark.parametrize("budget", [None, 0, -1, True, 1.5])
def test_context_limited_profile_requires_bounded_positive_output_budget(budget):
    instance = scheduler()
    with pytest.raises(InvalidRequestError, match="positive integer"):
        instance.add_request(request([1], budget))
    assert instance.requests == {} and not instance.waiting


@pytest.mark.parametrize(
    "configured,expected", [(None, 131072), (1048576, 131072), (8192, 8192)]
)
def test_profile_engine_applies_effective_total_context_limit(configured, expected):
    instance = VLMBatchedEngine(
        "unused",
        model_settings=ModelSettings(
            deepseek_v41_flash1017_enabled=True, max_context_window=configured
        ),
    )
    config = SchedulerConfig()
    instance._configure_flash1017_scheduler(config)
    assert config.max_total_tokens == expected
    assert config.max_num_seqs == 1 and config.prefill_step_size == 8192


def test_ordinary_engine_does_not_change_scheduler_context_or_concurrency():
    instance = VLMBatchedEngine("unused", model_settings=ModelSettings())
    config = SchedulerConfig()
    previous = config.max_num_seqs
    instance._configure_flash1017_scheduler(config)
    assert config.max_total_tokens is None and config.max_num_seqs == previous


class GuardedEngine:
    tokenizer = None

    def __init__(self):
        self.scheduler = scheduler()

    async def stream_chat(self, **kwargs):
        self.scheduler.add_request(request([1] * 9, kwargs["max_tokens"]))
        raise AssertionError("A refused request reached inference")
        yield

    async def stream_generate(self, **kwargs):
        self.scheduler.add_request(request([1] * 9, kwargs["max_tokens"]))
        raise AssertionError("A refused request reached inference")
        yield


def sse_data(chunks):
    return [
        json.loads(line[6:])
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "completion", "anthropic", "responses"])
async def test_all_streaming_api_families_preserve_context_rejection(api, monkeypatch):
    from omlx import server

    engine = GuardedEngine()
    messages = [{"role": "user", "content": "hello"}]
    if api == "chat":
        api_request = server.ChatCompletionRequest(
            model="flash", messages=messages, stream=True
        )
        stream = server.stream_chat_completion(
            engine, messages, api_request, max_tokens=2
        )
    elif api == "completion":
        monkeypatch.setattr(
            server,
            "get_sampling_params",
            lambda *a, **k: (1, 1, 0, 1, 0, 0, 0, 2, 0, 0),
        )
        monkeypatch.setattr(server, "_resolve_thinking_budget", lambda *a, **k: None)
        api_request = server.CompletionRequest(
            model="flash", prompt="hello", max_tokens=2, stream=True
        )
        stream = server.stream_completion(engine, "hello", api_request)
    elif api == "anthropic":
        api_request = server.AnthropicMessagesRequest(
            model="flash", messages=messages, max_tokens=2, stream=True
        )
        stream = server.stream_anthropic_messages(
            engine, messages, api_request, max_tokens=2
        )
    else:
        api_request = server.ResponsesRequest(
            model="flash", input="hello", max_output_tokens=2, stream=True
        )
        stream = server.stream_responses_api(
            engine, messages, api_request, max_tokens=2, store_response=False
        )
    events = sse_data([chunk async for chunk in stream])
    if api == "responses":
        failure = next(item for item in events if item.get("type") == "response.failed")
        error = failure["response"]["error"]
    else:
        error = next(item["error"] for item in events if "error" in item)
    assert "Maximum context length" in error["message"]
    if api == "anthropic":
        assert error["type"] == "invalid_request_error"
    else:
        assert error["code"] == "context_length_exceeded"
    assert engine.scheduler.requests == {} and not engine.scheduler.waiting


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses"]
)
async def test_nonstream_api_context_error_is_http400(path):
    from omlx import server

    http_request = HTTPRequest(
        {"type": "http", "method": "POST", "path": path, "headers": []}
    )
    response = await server.invalid_request_error_handler(
        http_request, ContextWindowExceededError(9, 2, 10)
    )
    assert response.status_code == 400
    error = json.loads(response.body)["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "context_length_exceeded"


@pytest.mark.asyncio
async def test_keepalive_wrapper_keeps_context_error_code():
    from omlx import server

    async def rejected():
        scheduler().add_request(request([1] * 9, 2))
        yield

    events = sse_data(
        [
            chunk
            async for chunk in server._with_sse_keepalive(
                rejected(), keepalive_chunk=None
            )
        ]
    )
    assert events[0]["error"]["code"] == "context_length_exceeded"
