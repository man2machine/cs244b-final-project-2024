# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:02:33 2023

@author: Shahir
"""

import os
import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Any, Self, final

import ray

from transformers import PreTrainedTokenizerBase, Conversation, TensorType  # type: ignore
from transformers import Conversation

from vllm import SamplingParams, CompletionOutput  # type: ignore
from vllm.engine.async_llm_engine import AsyncEngineArgs, AsyncLLMEngine, _AsyncLLMEngine  # type: ignore

from loguru import logger

from alto_lib.core.async_utils import AsyncConcurrentProcessor


LOG_LLM_OUTPUT = True

PromptRequestInfoT = TypeVar('PromptRequestInfoT')


class _CustomAsyncLLMEngine(_AsyncLLMEngine):
    def get_is_scheduler_full(
        self: Self
    ) -> bool:
        
        self.scheduler_config.max_num_seqs
        self.scheduler_config.max_num_batched_tokens
        # self.scheduler.
        return NotImplemented
    
    def get_num_requests_in_queue(
        self: Self
    ) -> int:

        return len(self.scheduler.waiting) + len(self.scheduler.swapped)

    def get_num_tokens_in_queue(
        self: Self
    ) -> int:

        num_tokens = sum(
            sum(
                sum(
                    len(s.data.prompt_token_ids)
                    for s in g.get_seqs()
                )
                for g in queue
            )
            for queue in (self.scheduler.waiting, self.scheduler.swapped)
        )

        return num_tokens

    def apply_chat_template(
        self: Self,
        conversation: list[dict[str, str]] | Conversation,
        chat_template: str | None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> str | list[int]:

        return self.tokenizer.tokenizer.apply_chat_template(
            conversation=conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            tokenizer_kwargs=tokenizer_kwargs
        )


class CustomAsyncLLMEngine(AsyncLLMEngine):
    _engine_class: type[_CustomAsyncLLMEngine] = _CustomAsyncLLMEngine
    engine: _CustomAsyncLLMEngine | ray.ObjectRef

    @classmethod
    def from_engine_args(
        cls: type[Self],
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True
    ) -> Self:

        return super().from_engine_args(engine_args, start_engine_loop=start_engine_loop)  # type: ignore

    def apply_chat_template(
        self: Self,
        conversation: list[dict[str, str]] | Conversation,
        chat_template: str | None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None
    ) -> str | list[int]:

        if self.engine_use_ray:
            out = self.engine.apply_chat_template.remote(  # type: ignore
                conversation=conversation,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_dict=return_dict,
                tokenizer_kwargs=tokenizer_kwargs
            )
        else:
            out = self.engine.apply_chat_template(  # type: ignore
                conversation=conversation,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_dict=return_dict,
                tokenizer_kwargs=tokenizer_kwargs
            )

        return out


@final
@dataclass(kw_only=True, frozen=True, slots=True)
class VllmPromptRequest(Generic[PromptRequestInfoT]):
    user_request_id: int
    prompt: str
    request_info: PromptRequestInfoT | None = field(default=None)


@final
@dataclass(kw_only=True, frozen=True, slots=True)
class AsyncGenerateCallbackInput(Generic[PromptRequestInfoT]):
    user_request_id: int
    vllm_request_id: int
    stream_finished: bool
    completion_output: CompletionOutput | None
    request_info: PromptRequestInfoT | None = field(default=None)


@final
class LLMGeneratorAsync:
    _engine_async: CustomAsyncLLMEngine
    _next_vllm_request_id: int
    _vllm_request_id_lock: asyncio.Lock
    _task_runner: AsyncConcurrentProcessor
    _tokenizer: PreTrainedTokenizerBase

    def __init__(
        self: Self,
        engine_args: AsyncEngineArgs,
        max_running_requests: int | None = None
    ) -> None:

        self._engine_async = CustomAsyncLLMEngine.from_engine_args(engine_args)
        self._next_vllm_request_id = 0
        self._vllm_request_id_lock = asyncio.Lock()
        self._task_runner = AsyncConcurrentProcessor(max_running_requests)

        internal_engine = self._engine_async.engine
        assert not isinstance(internal_engine, ray.ObjectRef)
        self._tokenizer = internal_engine.tokenizer.tokenizer

    async def _generate_async(
        self: Self,
        system_prompt: str,
        sampling_params: SamplingParams,
        user_request: VllmPromptRequest[PromptRequestInfoT],
        callback_func: (
            Callable[[AsyncGenerateCallbackInput], None] |
            Callable[[AsyncGenerateCallbackInput], Coroutine[Any, Any, None]]
        )
    ) -> None:

        await self._vllm_request_id_lock.acquire()
        vllm_request_id = self._next_vllm_request_id
        self._next_vllm_request_id += 1
        self._vllm_request_id_lock.release()

        message = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_request.prompt
            },
        ]
        user_prompt = self._tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        stream = await self._engine_async.add_request(
            str(vllm_request_id),
            user_prompt,
            sampling_params
        )

        user_request_id = user_request.user_request_id
        request_info = user_request.request_info
        callback_is_coro = asyncio.iscoroutinefunction(callback_func)

        completion_output = None
        async for output in stream:
            completion_output = output.outputs[0]
            callback_input = AsyncGenerateCallbackInput(
                user_request_id=user_request_id,
                vllm_request_id=vllm_request_id,
                stream_finished=stream.finished,
                completion_output=completion_output,
                request_info=request_info
            )
            if callback_is_coro:
                coro = callback_func(callback_input)
                assert coro is not None
                await coro
            else:
                callback_func(callback_input)

        assert stream.finished
        if completion_output and LOG_LLM_OUTPUT:
            logger.trace("Generated output {}".format(repr(completion_output.text)))

    async def add_request(
        self: Self,
        system_prompt: str,
        sampling_params: SamplingParams,
        user_request: VllmPromptRequest[PromptRequestInfoT],
        callback_func: (
            Callable[[AsyncGenerateCallbackInput], None] |
            Callable[[AsyncGenerateCallbackInput], Coroutine[Any, Any, None]]
        )
    ) -> None:

        await self._task_runner.add_task(
            self._generate_async(
                system_prompt=system_prompt,
                sampling_params=sampling_params,
                user_request=user_request,
                callback_func=callback_func
            )
        )


def get_sampling_params(
    temperature: float = 0.5,
    max_tokens: int = 512
) -> SamplingParams:

    return SamplingParams(temperature=temperature, max_tokens=max_tokens)


def get_colbert_tokenizer_fname(
    colbert_data_path: str
) -> str:

    return os.path.join(colbert_data_path, "msmarco.psg.kldR2.nway64.ib__colbert-400000/")
