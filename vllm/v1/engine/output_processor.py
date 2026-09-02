# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os  # [dspark-dsml-leak-guard-0271]
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import (
    STREAM_FINISHED,
    CompletionOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestOutput,
)
from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import TokenizerLike
from vllm.tracing import (
    SpanAttributes,
    SpanKind,
    extract_trace_context,
    instrument_manual,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.detokenizer import IncrementalDetokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.metrics.stats import (
    IterationStats,
    LoRARequestStates,
    RequestStateStats,
    SchedulerStats,
)

logger = init_logger(__name__)

# [dspark-dsml-leak-guard-0271] DSML 누출 가드 (기본 꺼짐).
_DSPARK_DSML_LEAK_GUARD = os.environ.get("DSPARK_DSML_LEAK_GUARD", "").strip() == "1"
# 파서 미복구 대체 방언 토큰: <dsml: (128840), </dsml: (128841). 출현 = 누출.
_DSPARK_DSML_DIALECT_TOKEN_IDS = frozenset({128840, 128841})
# 마커(｜DSML｜) 탈락 닫기 태그(유효 close 는 </｜DSML｜...> 라 부분열 불일치 → 오탐 0).
_DSPARK_DSML_BARE_CLOSERS = ("</invoke>", "</parameter>", "</result>")
# DSPARK: the guard is scoped to the answer section by default. Inside a
# reasoning (<think>) section the model legitimately quotes DSML syntax (e.g.
# an agent reading this very code), and a stop there silently discards the
# whole turn (observed 2026-09-02). DSPARK_DSML_LEAK_GUARD_SCOPE=request
# restores the old whole-request behaviour.
_DSPARK_DSML_LEAK_GUARD_SCOPE = (
    os.environ.get("DSPARK_DSML_LEAK_GUARD_SCOPE", "answer").strip() or "answer"
)
_DSPARK_THINK_START = "<think>"
_DSPARK_THINK_END = "</think>"
# DSPARK: at long context the model may open a tool call without </think>
# (vLLM #48931); the reasoning parser treats the opener as an implicit end
# and so does the guard, otherwise it would stay disarmed for the whole turn.
_DSPARK_DSML_TOOL_START = "<｜DSML｜tool_calls>"
_DSPARK_DSML_DIALECT_STRS = ("<dsml:", "</dsml:")


def _dspark_dialect_token_ids(tokenizer) -> frozenset[int]:
    """Resolve the DSML dialect tokens with the live tokenizer; fall back to
    the DeepSeek-V4 ids when it cannot be asked."""
    if tokenizer is None:
        return _DSPARK_DSML_DIALECT_TOKEN_IDS
    ids: set[int] = set()
    for text in _DSPARK_DSML_DIALECT_STRS:
        try:
            enc = tokenizer.encode(text, add_special_tokens=False)
        except Exception:  # noqa: BLE001
            return _DSPARK_DSML_DIALECT_TOKEN_IDS
        if len(enc) == 1:
            ids.add(int(enc[0]))
    return frozenset(ids) if ids else _DSPARK_DSML_DIALECT_TOKEN_IDS


def _dspark_think_token_ids(
    tokenizer,
) -> tuple[list[int], list[int], list[int]]:
    """Token ids of the DeepSeek think markers and of the DSML tool-call
    opener ([] if unknown)."""
    if tokenizer is None:
        return [], [], []
    try:
        start = tokenizer.encode(_DSPARK_THINK_START, add_special_tokens=False)
        end = tokenizer.encode(_DSPARK_THINK_END, add_special_tokens=False)
    except Exception:  # noqa: BLE001 - best effort, guard falls back to text
        return [], [], []
    if not (1 <= len(start) <= 3 and 1 <= len(end) <= 3):
        return [], [], []
    try:
        dsml = list(tokenizer.encode(_DSPARK_DSML_TOOL_START, add_special_tokens=False))
    except Exception:  # noqa: BLE001
        dsml = []
    return list(start), list(end), dsml


def _dspark_last_index(target: list[int], seq: list[int]) -> int:
    if not seq:
        return -1
    n = len(seq)
    for i in range(len(target) - n, -1, -1):
        if target[i : i + n] == seq:
            return i
    return -1


# shared empty CPU tensor used as a placeholder pooling output
EMPTY_CPU_TENSOR = torch.empty(0, device="cpu")


class RequestOutputCollector:
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """

    def __init__(self, output_kind: RequestOutputKind, request_id: str):
        self.aggregate = output_kind == RequestOutputKind.DELTA
        self.request_id = request_id
        self.output: RequestOutput | PoolingRequestOutput | Exception | None = None
        self.ready = asyncio.Event()

        self._input_stream_task: asyncio.Task | None = None

    def put(self, output: RequestOutput | PoolingRequestOutput | Exception) -> None:
        """Non-blocking put operation."""
        if self.output is None or isinstance(output, Exception):
            self.output = output
            self.ready.set()
        elif isinstance(self.output, RequestOutput) and isinstance(
            output, RequestOutput
        ):
            # This ensures that request outputs with different request indexes
            # (if n > 1) do not override each other.
            self.output.add(output, aggregate=self.aggregate)
        elif isinstance(self.output, PoolingRequestOutput) and isinstance(
            output, PoolingRequestOutput
        ):
            self.output = output

    async def get(self) -> RequestOutput | PoolingRequestOutput:
        """Get operation blocks on put event."""
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    def get_nowait(self) -> RequestOutput | PoolingRequestOutput | None:
        """Non-blocking get operation."""
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    def close(self):
        if self._input_stream_task is not None:
            self._input_stream_task.cancel()
        self._input_stream_task = None

    def __del__(self):
        if (task := self._input_stream_task) is not None:
            task.get_loop().call_soon_threadsafe(task.cancel)
            self._input_stream_task = None


@dataclass
class OutputProcessorOutput:
    request_outputs: list[RequestOutput | PoolingRequestOutput]
    reqs_to_abort: list[str]


@dataclass
class StreamingUpdate:
    """Streaming input update data for output processor.

    Contains the incremental prompt data to be applied to a request state
    when the current sub-request completes.
    """

    prompt: str | None
    prompt_token_ids: list[int] | None
    arrival_time: float
    final: bool = False


class RequestState:
    def __init__(
        self,
        request_id: str,
        external_req_id: str,
        parent_req: ParentRequest | None,
        request_index: int,
        lora_request: LoRARequest | None,
        output_kind: RequestOutputKind,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        logprobs_processor: LogprobsProcessor | None,
        detokenizer: IncrementalDetokenizer | None,
        max_tokens_param: int | None,
        arrival_time: float,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
        top_p: float | None = None,
        n: int | None = None,
        temperature: float | None = None,
        stream_input: bool = False,
    ):
        self.request_id = request_id
        self.external_req_id = external_req_id
        self.parent_req = parent_req
        self.request_index = request_index
        self.lora_request = lora_request
        self.lora_name = lora_request.lora_name if lora_request is not None else None
        self.output_kind = output_kind
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds
        )
        self.logprobs_processor = logprobs_processor
        self.detokenizer = detokenizer
        # DSPARK: True while the output is still inside a <think> section
        # (leak guard scope); maintained by OutputProcessor.process_outputs.
        self.dspark_in_reasoning = False
        self.dspark_answer_text_start = 0
        self.max_tokens_param = max_tokens_param
        self.top_p = top_p
        self.n = n
        self.temperature = temperature
        self.is_prefilling = True
        self.queue = queue
        self.num_cached_tokens = 0
        self.num_cache_creation_tokens = 0

        self.stats = RequestStateStats(arrival_time=arrival_time) if log_stats else None

        # Routed experts accumulation (prompt + sample chunks)
        self.routed_experts_chunks: list[np.ndarray] = []

        # Stream Interval
        self.stream_interval = stream_interval
        self.sent_tokens_offset = 0  # Offset of sent tokens

        # Streaming input queue
        self.streaming_input = stream_input
        self.input_chunk_queue: deque[StreamingUpdate] | None = (
            deque() if stream_input else None
        )

    def apply_streaming_update(self, update: StreamingUpdate) -> None:
        # Apply the update to the request state.
        self.streaming_input = not update.final
        # TODO also include relevant output tokens in new prompt here
        #     (match scheduler behavior).
        if update.prompt:
            self.prompt = (
                (self.prompt + update.prompt) if self.prompt else update.prompt
            )
        if self.prompt_token_ids:
            self.prompt_token_ids.extend(update.prompt_token_ids or ())
        else:
            self.prompt_token_ids = update.prompt_token_ids or []
        assert self.prompt_token_ids is not None
        self.prompt_len = len(self.prompt_token_ids)
        if self.stats is not None:
            self.stats.arrival_time = update.arrival_time
        self.is_prefilling = True

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        request_index: int,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
    ) -> "RequestState":
        if sampling_params := request.sampling_params:
            if not sampling_params.detokenize:
                tokenizer = None
            output_kind = sampling_params.output_kind
            if sampling_params.stream_interval is not None:
                # clamp to the engine-level stream interval.
                stream_interval = max(sampling_params.stream_interval, stream_interval)
            logprobs_processor = LogprobsProcessor.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            detokenizer = IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            max_tokens_param = sampling_params.max_tokens
            top_p = sampling_params.top_p
            n = sampling_params.n
            temperature = sampling_params.temperature
        else:
            logprobs_processor = None
            detokenizer = None
            max_tokens_param = None
            top_p = None
            n = None
            temperature = None
            assert request.pooling_params is not None
            output_kind = request.pooling_params.output_kind

        assert request.external_req_id is not None
        return cls(
            request_id=request.request_id,
            external_req_id=request.external_req_id,
            parent_req=parent_req,
            request_index=request_index,
            lora_request=request.lora_request,
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            max_tokens_param=max_tokens_param,
            top_p=top_p,
            n=n,
            temperature=temperature,
            arrival_time=request.arrival_time,
            queue=queue,
            log_stats=log_stats,
            stream_interval=stream_interval,
            stream_input=request.resumable,
        )

    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput | PoolingRequestOutput | None:
        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            # Only the final output is required in FINAL_ONLY mode.
            return None

        if self.stream_interval > 1:
            assert self.detokenizer is not None

            # Send output request only when
            # 1. It has finished, or
            # 2. It is the first token, or
            # 3. It has reached the stream interval number of tokens
            if not (
                finished
                or self.sent_tokens_offset == 0
                or self.detokenizer.num_output_tokens() - self.sent_tokens_offset
                >= self.stream_interval
            ):
                return None

            if self.output_kind == RequestOutputKind.DELTA:
                # Send tokens from the offset in DELTA mode, otherwise all
                # tokens are sent.
                new_token_ids = self.detokenizer.output_token_ids[
                    self.sent_tokens_offset :
                ]
                self.sent_tokens_offset = self.detokenizer.num_output_tokens()

        external_req_id = self.external_req_id

        if pooling_output is not None:
            return self._new_request_output(
                external_req_id,
                [self._new_pooling_output(pooling_output)],
                finished,
            )

        output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)

        if self.parent_req is None:
            outputs = [output]
        else:
            outputs, finished = self.parent_req.get_outputs(self.request_id, output)
            if not outputs:
                return None
            external_req_id = self.parent_req.external_req_id

        return self._new_request_output(
            external_req_id,
            outputs,
            finished,
            kv_transfer_params,
            ec_transfer_params,
        )

    def _new_request_output(
        self,
        external_req_id: str,
        outputs: list[CompletionOutput] | list[PoolingOutput],
        finished: bool,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput | PoolingRequestOutput:
        # If prompt embeds were used, put placeholder prompt token ids
        prompt_token_ids = self.prompt_token_ids
        if prompt_token_ids is None and self.prompt_embeds is not None:
            prompt_token_ids = [0] * len(self.prompt_embeds)
        assert prompt_token_ids is not None

        first_output = outputs[0]
        if isinstance(first_output, PoolingOutput):
            assert len(outputs) == 1
            return PoolingRequestOutput(
                request_id=external_req_id,
                outputs=first_output,
                num_cached_tokens=self.num_cached_tokens,
                prompt_token_ids=prompt_token_ids,
                finished=finished,
            )
        assert self.logprobs_processor is not None
        if self.output_kind == RequestOutputKind.DELTA:
            # Side effect: logprobs processor forgets prompt logprobs
            prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()
        else:
            prompt_logprobs = self.logprobs_processor.prompt_logprobs

        return RequestOutput(
            request_id=external_req_id,  # request_id is what was provided externally
            lora_request=self.lora_request,
            prompt=self.prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=prompt_logprobs,
            outputs=cast(list[CompletionOutput], outputs),
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            ec_transfer_params=ec_transfer_params,
            num_cached_tokens=self.num_cached_tokens,
            num_cache_creation_tokens=self.num_cache_creation_tokens,
            metrics=self.stats,
        )

    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
    ) -> CompletionOutput:
        assert self.detokenizer is not None
        assert self.logprobs_processor is not None
        finished = finish_reason is not None
        delta = self.output_kind == RequestOutputKind.DELTA

        # Prepare text and token_ids, based on delta mode
        text = self.detokenizer.get_next_output_text(finished, delta)
        if not delta:
            token_ids = self.detokenizer.output_token_ids

        # Prepare logprobs, based on delta mode
        logprobs = self.logprobs_processor.logprobs
        if delta and logprobs:
            logprobs = logprobs[-len(token_ids) :]

        # Concatenate routed experts on finish
        routed_experts = None
        if finished and self.routed_experts_chunks:
            routed_experts = np.concatenate(self.routed_experts_chunks, axis=0)

        return CompletionOutput(
            index=self.request_index,
            text=text,
            token_ids=token_ids,
            routed_experts=routed_experts,
            logprobs=logprobs,
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,
            finish_reason=str(finish_reason) if finished else None,
            stop_reason=stop_reason if finished else None,
        )

    def _new_pooling_output(self, pooling_output: torch.Tensor) -> PoolingOutput:
        return PoolingOutput(data=pooling_output)


class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        log_stats: bool,
        stream_interval: int = 1,
        tracing_enabled: bool = False,
    ):
        self.log_stats = log_stats
        self.tokenizer = tokenizer
        self.stream_interval = stream_interval
        # DSPARK: think marker ids for the DSML leak guard scope.
        (
            self._dspark_think_start_ids,
            self._dspark_think_end_ids,
            self._dspark_dsml_start_ids,
        ) = (
            _dspark_think_token_ids(tokenizer)
            if _DSPARK_DSML_LEAK_GUARD
            else ([], [], [])
        )
        self._dspark_dialect_ids = (
            _dspark_dialect_token_ids(tokenizer)
            if _DSPARK_DSML_LEAK_GUARD
            else _DSPARK_DSML_DIALECT_TOKEN_IDS
        )
        self._dspark_leak_guard_hits = 0
        self.request_states: dict[str, RequestState] = {}
        self.parent_requests: dict[str, ParentRequest] = {}
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
        self.lora_states = LoRARequestStates(log_stats)
        self.tracing_enabled = tracing_enabled

    def get_num_unfinished_requests(self):
        return len(self.request_states)

    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    def propagate_error(self, e: Exception):
        """Propagate error to all generate() tasks."""

        for _, state in self.request_states.items():
            assert state.queue is not None
            state.queue.put(e)

    def abort_requests(self, request_ids: Iterable[str], internal: bool) -> list[str]:
        """Abort a list of requests.

        The request_ids may be either external request IDs (those passed to
        InputProcessor.process_inputs()) or internal request IDs (those randomly
        generated when creating the EngineCoreRequest).

        If an external request ID is provided, and that external request ID
        was used for multiple requests, all requests associated with that external
        request ID are aborted.

        In the case of parallel sampling, a request ID may be used to identify
        a parent request, in which case the associated child requests are aborted
        also.
        """
        internal_req_ids = []
        for request_id in request_ids:
            if internal:
                # Internal ID - this may be a parent request
                internal_req_ids.append(request_id)

                # Remove internal ID from the external->internal mapping
                if req_state := self.request_states.get(request_id):
                    external_req_id = req_state.external_req_id
                    internal_ids = self.external_req_ids[external_req_id]
                    internal_ids.remove(request_id)
                    if not internal_ids:
                        del self.external_req_ids[external_req_id]
            elif internal_ids := self.external_req_ids.pop(request_id, []):
                # External ID - abort all requests in the external->internal mapping
                internal_req_ids.extend(internal_ids)

        request_ids_to_abort = []
        for request_id in internal_req_ids:
            req_state = self.request_states.pop(request_id, None)
            if req_state is not None:
                self.lora_states.request_finished(request_id, req_state.lora_name)
                request_ids_to_abort.append(request_id)
                # Produce final abort output.
                if req_state.queue is not None and (
                    request_output := req_state.make_request_output(
                        new_token_ids=[],
                        # Set pooling_output is not None to
                        # correctly enter the abort pooling branch
                        pooling_output=EMPTY_CPU_TENSOR
                        if req_state.detokenizer is None
                        else None,
                        finish_reason=FinishReason.ABORT,
                        stop_reason=None,
                        kv_transfer_params=None,
                        ec_transfer_params=None,
                    )
                ):
                    req_state.queue.put(request_output)
            elif parent := self.parent_requests.get(request_id):
                # Abort children prior to removing the parent.
                if parent.child_requests:
                    child_reqs = list(parent.child_requests)
                    child_reqs = self.abort_requests(child_reqs, internal=True)
                    request_ids_to_abort.extend(child_reqs)
                self.parent_requests.pop(request_id, None)
        return request_ids_to_abort

    def _dspark_leak_guard_check(
        self, req_state: RequestState, new_token_ids: list[int]
    ) -> str | None:
        """DSML leak guard (env DSPARK_DSML_LEAK_GUARD=1). Returns a short
        reason when the request must be stopped, else None.

        Scope (DSPARK_DSML_LEAK_GUARD_SCOPE, default "answer"): while the
        output is still inside a <think> section nothing is checked; the
        section is left when a </think> marker shows up (token ids, or text
        as a fallback when the ids are unknown) and only text produced after
        the marker is inspected from then on."""
        det = req_state.detokenizer
        assert det is not None
        ids_to_check: list[int] = list(new_token_ids)
        if req_state.dspark_in_reasoning:
            end_ids = self._dspark_think_end_ids
            dsml_ids = self._dspark_dsml_start_ids
            left = False
            n_new = len(new_token_ids)
            if end_ids:
                # </think>: the answer starts after the marker
                window = list(det.output_token_ids[-(n_new + len(end_ids)) :])
                pos = _dspark_last_index(window, end_ids)
                if pos >= 0:
                    left = True
                    after = pos + len(end_ids) - (len(window) - n_new)
                    ids_to_check = list(new_token_ids[max(0, after) :])
                    cut = det.output_text.rfind(_DSPARK_THINK_END)
                    req_state.dspark_answer_text_start = (
                        cut + len(_DSPARK_THINK_END)
                        if cut >= 0
                        else len(det.output_text)
                    )
            if not left and dsml_ids:
                # tool-call opener without </think> (implicit end): the block
                # itself is inspected, so the answer starts AT the marker
                window = list(det.output_token_ids[-(n_new + len(dsml_ids)) :])
                pos = _dspark_last_index(window, dsml_ids)
                if pos >= 0:
                    left = True
                    after = pos - (len(window) - n_new)
                    ids_to_check = list(new_token_ids[max(0, after) :])
                    cut = det.output_text.rfind(_DSPARK_DSML_TOOL_START)
                    req_state.dspark_answer_text_start = (
                        cut if cut >= 0 else len(det.output_text)
                    )
            if not left and not end_ids:
                tail = det.output_text[-64:]
                if _DSPARK_THINK_END in tail:
                    left = True
                    ids_to_check = []
                    cut = det.output_text.rfind(_DSPARK_THINK_END)
                    req_state.dspark_answer_text_start = cut + len(_DSPARK_THINK_END)
                elif _DSPARK_DSML_TOOL_START in tail:
                    left = True
                    cut = det.output_text.rfind(_DSPARK_DSML_TOOL_START)
                    req_state.dspark_answer_text_start = cut
            if not left:
                return None
            req_state.dspark_in_reasoning = False
        if any(t in self._dspark_dialect_ids for t in ids_to_check):
            return "dsml_dialect_token"
        text = det.output_text
        start = max(getattr(req_state, "dspark_answer_text_start", 0), len(text) - 64)
        tail = text[start:]
        for closer in _DSPARK_DSML_BARE_CLOSERS:
            if closer in tail:
                return f"bare_closer:{closer}"
        return None

    def _dspark_prompt_in_think(
        self, prompt_token_ids: list[int] | None, prompt: str | None
    ) -> bool:
        """DeepSeek V4 chat templates open <think> at the end of the prompt
        when thinking is on: the output then starts inside a reasoning
        section. Token ids first, text as a fallback."""
        if prompt_token_ids and self._dspark_think_start_ids:
            tail = list(prompt_token_ids[-64:])
            ls = _dspark_last_index(tail, self._dspark_think_start_ids)
            le = _dspark_last_index(tail, self._dspark_think_end_ids)
            return ls > le
        if prompt:
            tail = prompt[-128:]
            return tail.rfind(_DSPARK_THINK_START) > tail.rfind(_DSPARK_THINK_END)
        return False

    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        request_id = request.request_id
        req_state = self.request_states.get(request_id)
        if req_state is not None:
            self._update_streaming_request_state(req_state, request, prompt)
            return

        req_state = RequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            log_stats=self.log_stats,
            stream_interval=self.stream_interval,
        )
        self.request_states[request_id] = req_state
        if _DSPARK_DSML_LEAK_GUARD and _DSPARK_DSML_LEAK_GUARD_SCOPE != "request":
            req_state.dspark_in_reasoning = self._dspark_prompt_in_think(
                request.prompt_token_ids, prompt
            )
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req

        # Track the external_req_id -> [internal_req_id, ...] mapping
        self.external_req_ids[req_state.external_req_id].append(request_id)

    def _update_streaming_request_state(
        self, req_state: RequestState, request: EngineCoreRequest, prompt: str | None
    ) -> None:
        """Queue a streaming update instead of immediately applying it."""
        if not request.resumable:
            # Final request - just mark completion, don't add its dummy tokens.
            if req_state.input_chunk_queue is None:
                # Engine already finished - emit final output and clean up.
                self._finish_request(req_state)
                if req_state.queue is not None:
                    # Emit a final output with finished=True
                    # to unblock the generate() loop.
                    req_state.queue.put(STREAM_FINISHED)
            elif req_state.input_chunk_queue:
                req_state.input_chunk_queue[-1].final = True
            else:
                req_state.streaming_input = False
            return

        update = StreamingUpdate(
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            arrival_time=request.arrival_time,
        )

        # Apply request updates now if the last input already completed.
        if req_state.input_chunk_queue is None:
            req_state.apply_streaming_update(update)
            req_state.input_chunk_queue = deque()
        else:
            # Queue the streaming update otherwise.
            req_state.input_chunk_queue.append(update)

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: IterationStats | None = None,
    ) -> OutputProcessorOutput:
        """
        Process the EngineCoreOutputs:
        1) Compute stats for logging
        2) Detokenize
        3) Create and handle RequestOutput objects:
            * If there is a queue (for usage with AsyncLLM),
              put the RequestOutput objects into the queue for
              handling by the per-request generate() tasks.

            * If there is no queue (for usage with LLMEngine),
              return a list of RequestOutput objects.

        NOTE FOR DEVELOPERS

        vLLM V1 minimizes the number of python loops over the full
        batch to ensure system overheads are minimized. This is the
        only function that should loop over EngineCoreOutputs.

        If you need to touch every element of the batch, do it from
        within the loop below.
        """

        request_outputs: list[RequestOutput | PoolingRequestOutput] = []
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # 1) Compute stats for this iteration.
            self._update_stats_from_output(
                req_state, engine_core_output, engine_core_timestamp, iteration_stats
            )

            new_token_ids = engine_core_output.new_token_ids
            pooling_output = engine_core_output.pooling_output
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason
            kv_transfer_params = engine_core_output.kv_transfer_params
            ec_transfer_params = engine_core_output.ec_transfer_params
            if engine_core_output.routed_experts is not None:
                req_state.routed_experts_chunks.append(
                    engine_core_output.routed_experts
                )

            if req_state.is_prefilling:
                if engine_core_output.prefill_stats is not None:
                    req_state.num_cached_tokens = (
                        engine_core_output.prefill_stats.num_cached_tokens
                    )
                    req_state.num_cache_creation_tokens = (
                        engine_core_output.prefill_stats.num_cache_creation_tokens
                    )
                req_state.is_prefilling = False

            if pooling_output is None:
                assert req_state.detokenizer is not None
                assert req_state.logprobs_processor is not None
                # 2) Detokenize the token ids into text and perform stop checks.
                stop_string = req_state.detokenizer.update(
                    new_token_ids, finish_reason == FinishReason.STOP
                )
                if stop_string:
                    finish_reason = FinishReason.STOP
                    stop_reason = stop_string

                # [dspark-dsml-leak-guard-0271] DSML 구조 토큰이 tool-parse
                # 성립 없이 표면화되면(미복구 방언 or 마커탈락 닫기 태그) 요청을
                # stop 처리 → 생 DSML 히스토리 오염 차단.
                if _DSPARK_DSML_LEAK_GUARD and finish_reason is None:
                    _leak = self._dspark_leak_guard_check(req_state, new_token_ids)
                    if _leak:
                        finish_reason = FinishReason.STOP
                        stop_reason = "dspark_dsml_leak_guard"
                        self._dspark_leak_guard_hits += 1
                        logger.info(
                            "DSPARK DSML leak guard stopped request %s (%s) after "
                            "%d output tokens (hits %d total).",
                            req_id,
                            _leak,
                            len(req_state.detokenizer.output_token_ids),
                            self._dspark_leak_guard_hits,
                        )

                # 3) Compute sample and prompt logprobs for request,
                # if required.
                req_state.logprobs_processor.update_from_output(engine_core_output)

            # 4) Create and handle RequestOutput objects.
            if request_output := req_state.make_request_output(
                new_token_ids,
                pooling_output,
                finish_reason,
                stop_reason,
                kv_transfer_params,
                ec_transfer_params,
            ):
                if req_state.streaming_input:
                    request_output.finished = False

                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)

            # Free completed requests.
            if finish_reason is not None:
                if req_state.streaming_input:
                    if req_state.input_chunk_queue:
                        update = req_state.input_chunk_queue.popleft()
                        req_state.apply_streaming_update(update)
                    else:
                        req_state.input_chunk_queue = None
                else:
                    self._finish_request(req_state)
                    if not engine_core_output.finished:
                        # If req not finished in EngineCore, but Detokenizer
                        # detected stop string, abort needed in EngineCore.
                        reqs_to_abort.append(req_id)

                    # Track per-request stats
                    self._update_stats_from_finished(
                        req_state, finish_reason, iteration_stats
                    )
                    if self.tracing_enabled:
                        self.do_tracing(engine_core_output, req_state, iteration_stats)

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )

    def _finish_request(self, req_state: RequestState) -> None:
        req_id = req_state.request_id
        self.request_states.pop(req_id)

        internal_ids = self.external_req_ids[req_state.external_req_id]
        internal_ids.remove(req_id)
        if not internal_ids:
            del self.external_req_ids[req_state.external_req_id]

        # Remove parent request if applicable.
        parent_req = req_state.parent_req
        if parent_req and not parent_req.child_requests:
            self.parent_requests.pop(parent_req.request_id, None)

    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
        self.lora_states.update_scheduler_stats(scheduler_stats)

    def do_tracing(
        self,
        engine_core_output: EngineCoreOutput,
        req_state: RequestState,
        iteration_stats: IterationStats | None,
    ) -> None:
        assert req_state.stats is not None
        assert iteration_stats is not None

        metrics = req_state.stats
        arrival_time_ns = int(metrics.arrival_time * 1e9)
        trace_context = extract_trace_context(engine_core_output.trace_headers)
        prompt_length = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # Calculate timing metrics
        e2e_time = iteration_stats.iteration_timestamp - metrics.arrival_time
        queued_time = metrics.scheduled_ts - metrics.queued_ts
        prefill_time = metrics.first_token_ts - metrics.scheduled_ts
        decode_time = metrics.last_token_ts - metrics.first_token_ts
        inference_time = metrics.last_token_ts - metrics.scheduled_ts

        # Build attributes dict
        attributes: dict[str, Any] = {
            SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN: (
                metrics.first_token_latency
            ),
            SpanAttributes.GEN_AI_LATENCY_E2E: e2e_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE: queued_time,
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: prompt_length,
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: (
                metrics.num_generation_tokens
            ),
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL: prefill_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE: decode_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE: inference_time,
            SpanAttributes.GEN_AI_REQUEST_ID: req_state.external_req_id,
        }

        # Add optional request parameters
        if req_state.top_p:
            attributes[SpanAttributes.GEN_AI_REQUEST_TOP_P] = req_state.top_p
        if req_state.max_tokens_param:
            attributes[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] = (
                req_state.max_tokens_param
            )
        if req_state.temperature:
            attributes[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] = (
                req_state.temperature
            )
        if req_state.n:
            attributes[SpanAttributes.GEN_AI_REQUEST_N] = req_state.n

        instrument_manual(
            span_name="llm_request",
            start_time=arrival_time_ns,
            attributes=attributes,
            context=trace_context,
            kind=SpanKind.SERVER,
        )

    def _update_stats_from_output(
        self,
        req_state: RequestState,
        engine_core_output: EngineCoreOutput,
        engine_core_timestamp: float | None,
        iteration_stats: IterationStats | None,
    ):
        if iteration_stats is None:
            return

        assert engine_core_timestamp is not None
        assert req_state.stats is not None
        iteration_stats.update_from_output(
            engine_core_output,
            engine_core_timestamp,
            req_state.is_prefilling,
            req_state.stats,
            self.lora_states,
            req_state.lora_name,
        )

    def _update_stats_from_finished(
        self,
        req_state: RequestState,
        finish_reason: FinishReason | None,
        iteration_stats: IterationStats | None,
    ):
        if iteration_stats is None:
            return

        assert finish_reason is not None
        assert req_state.stats is not None
        iteration_stats.update_from_finished_request(
            finish_reason=finish_reason,
            request_id=req_state.external_req_id,
            num_prompt_tokens=req_state.prompt_len,
            max_tokens_param=req_state.max_tokens_param,
            req_stats=req_state.stats,
            num_cached_tokens=req_state.num_cached_tokens,
        )
        self.lora_states.request_finished(req_state.request_id, req_state.lora_name)

        ParentRequest.observe_finished_request(
            req_state.parent_req, iteration_stats, req_state.stats.num_generation_tokens
        )
