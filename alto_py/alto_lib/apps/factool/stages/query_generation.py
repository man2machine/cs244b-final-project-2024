# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:55:58 2023

@author: Shahir
"""

import re
import collections
import logging
from dataclasses import dataclass
from typing import Self, final

from loguru import logger

from vllm import AsyncEngineArgs  # type: ignore

from alto_lib.core.manager import (
    StageCommunicatorFactory, QueueRequestMessage, QueueRequestMessageSerializer,
    StageOutputQueueInterface, AsyncQueueEmptyException
)
from alto_lib.core.async_utils import AsyncConcurrentProcessor
from alto_lib.core.app_endpoints import InstanceComputeLatencyLogger
from alto_lib.apps.factool.common.messages import LMClaimQuestion, LMClaim, LMClaimQuestionToken
from alto_lib.apps.factool.common.prompts import QUERY_GENERATION_PROMPT
from alto_lib.apps.factool.common.lm_utils import (
    get_sampling_params, VllmPromptRequest, AsyncGenerateCallbackInput, LLMGeneratorAsync
)
from alto_lib.apps.factool.common import FactoolQueueEndpoint


@dataclass(frozen=True, slots=True)
class RequestInfo:
    claim_id: int


@final
class GenerateProcessor:
    _queue_to_bm25: StageOutputQueueInterface[QueueRequestMessage[LMClaimQuestionToken]]
    _queue_to_colbert_encoder: StageOutputQueueInterface[QueueRequestMessage[LMClaimQuestion]]
    _queue_to_driver: StageOutputQueueInterface[QueueRequestMessage[LMClaimQuestion]]
    _debug_log: InstanceComputeLatencyLogger

    _next_query_ids_bm25: dict[int, int]
    _next_query_token_start_indices: dict[int, int]
    _next_query_ids_colbert: dict[int, int]
    _next_query_start_indices: dict[int, int]

    _sent_token: dict[int, bool]
    _sent_question: dict[int, bool]

    def __init__(
        self: Self,
        queue_to_bm25: StageOutputQueueInterface[QueueRequestMessage[LMClaimQuestionToken]],
        queue_to_colbert_encoder: StageOutputQueueInterface[QueueRequestMessage[LMClaimQuestion]],
        queue_to_driver: StageOutputQueueInterface[QueueRequestMessage[LMClaimQuestion]],
        debug_log: InstanceComputeLatencyLogger
    ) -> None:

        self._queue_to_bm25 = queue_to_bm25
        self._queue_to_colbert_encoder = queue_to_colbert_encoder
        self._queue_to_driver = queue_to_driver
        self._debug_log = debug_log

        self._next_query_ids_bm25 = collections.defaultdict(int)
        self._next_query_token_start_indices = collections.defaultdict(int)
        self._next_query_ids_colbert = collections.defaultdict(int)
        self._next_query_start_indices = collections.defaultdict(int)

        self._sent_token = collections.defaultdict(bool)
        self._sent_question = collections.defaultdict(bool)

    async def generate_callback(
        self: Self,
        data: AsyncGenerateCallbackInput[RequestInfo]
    ) -> None:

        vllm_request_id = data.vllm_request_id
        assert data.completion_output is not None
        assert data.request_info is not None

        all_text = data.completion_output.text
        all_text = all_text.replace("<|im_end|>", "")
        while all_text and ((all_text[-1] == " ") or (all_text[-1] == "\n")):
            all_text = all_text[:-1]

        # region - parse and send query tokens
        if vllm_request_id in self._next_query_token_start_indices:
            new_text = all_text[self._next_query_token_start_indices[vllm_request_id]:]
        else:
            new_text = all_text
            while new_text and (new_text[0] == " "):
                self._next_query_token_start_indices[vllm_request_id] += 1
                new_text = new_text[1:]

        if len(new_text) != 0:
            split_matches = list(re.finditer("[ \n]", new_text))
            split_indices = [m.start() for m in split_matches]
            split_is_newline = [(m.group(0) == "\n") for m in split_matches]

            if data.stream_finished:
                split_indices.append(len(new_text))
                split_is_newline.append(False)

            token_start_index = 0
            for i in range(len(split_indices)):
                last_split = (i == (len(split_indices) - 1))
                last_token_in_query = False
                if last_split:
                    if data.stream_finished:
                        last_token_in_query = True
                    else:
                        # skip as we don't know if the token is the end of a claim or not because
                        # even if there is a space, after the space there could be a newline
                        break
                elif split_is_newline[i + 1]:
                    last_token_in_query = True

                token_text = new_text[token_start_index:split_indices[i]]

                if len(token_text):
                    claim_query_token_msg = LMClaimQuestionToken(
                        word=token_text,
                        claim_id=data.request_info.claim_id,
                        question_id=self._next_query_ids_bm25[vllm_request_id],
                        last_token=last_token_in_query,
                    )
                    queue_token_item = QueueRequestMessage(
                        user_request_id=data.user_request_id,
                        message=claim_query_token_msg
                    )

                    logger.trace("Generated queue message {}".format(queue_token_item))

                    self._debug_log.update_stage_queue_out((data.user_request_id, data.request_info.claim_id))
                    await self._queue_to_bm25.put_async(queue_token_item, block=False)

                    self._sent_token[vllm_request_id] = True
                    if last_token_in_query:
                        self._next_query_ids_bm25[vllm_request_id] += 1

                token_start_index += (split_indices[i] + 1)

            self._next_query_token_start_indices[vllm_request_id] += token_start_index

        # endregion

        # region - parse and send queries
        if vllm_request_id in self._next_query_start_indices:
            new_text = all_text[self._next_query_start_indices[vllm_request_id]:]
        else:
            new_text = all_text
            while new_text and (new_text[0] == " "):
                self._next_query_start_indices[vllm_request_id] += 1
                new_text = new_text[1:]

        if len(new_text) != 0:
            split_indices = [m.start() for m in re.finditer("\n", new_text)]

            if data.stream_finished:
                split_indices.append(len(new_text))

            query_start_index = 0
            for i in range(len(split_indices)):
                last_split = (i == (len(split_indices) - 1))
                last_query = (last_split and data.stream_finished)

                query_text = new_text[query_start_index:split_indices[i]]

                if len(query_text):
                    claim_query_msg = LMClaimQuestion(
                        question=query_text,
                        claim_id=data.request_info.claim_id,
                        question_id=self._next_query_ids_colbert[vllm_request_id],
                        last_question=last_query,
                    )
                    queue_question_item = QueueRequestMessage(
                        user_request_id=data.user_request_id,
                        message=claim_query_msg
                    )

                    logger.trace("Generated queue message {}".format(queue_question_item))

                    self._debug_log.update_stage_queue_out((data.user_request_id, data.request_info.claim_id))
                    await self._queue_to_colbert_encoder.put_async(queue_question_item, block=False)
                    await self._queue_to_driver.put_async(queue_question_item, block=False)

                    self._sent_question[vllm_request_id] = True
                    self._next_query_ids_colbert[vllm_request_id] += 1

                query_start_index += (split_indices[i] + 1)

            self._next_query_start_indices[vllm_request_id] += query_start_index

        # endregion

        if data.stream_finished:
            if not self._sent_token[vllm_request_id]:
                logger.trace("Did not send token, sending blank token now")
                claim_query_token_msg = LMClaimQuestionToken(
                    word="",
                    claim_id=data.request_info.claim_id,
                    question_id=0,
                    last_token=True,
                )
                queue_token_item = QueueRequestMessage(
                    user_request_id=data.user_request_id,
                    message=claim_query_token_msg
                )

                logger.trace("Generated queue message {}".format(queue_token_item))

                self._debug_log.update_stage_queue_out(data.user_request_id)
                await self._queue_to_bm25.put_async(queue_token_item, block=False)

            else:
                self._next_query_ids_bm25.pop(vllm_request_id)
                self._next_query_token_start_indices.pop(vllm_request_id)

            self._sent_token.pop(vllm_request_id)

            if not self._sent_question[vllm_request_id]:
                logger.trace("Did not send query, sending blank query now")
                claim_query_msg = LMClaimQuestion(
                    question="",
                    claim_id=data.request_info.claim_id,
                    question_id=0,
                    last_question=True,
                )
                queue_question_item = QueueRequestMessage(
                    user_request_id=data.user_request_id,
                    message=claim_query_msg
                )

                logger.trace("Generated queue message {}".format(queue_question_item))

                self._debug_log.update_stage_queue_out((data.user_request_id, data.request_info.claim_id))
                await self._queue_to_colbert_encoder.put_async(queue_question_item, block=False)
                await self._queue_to_driver.put_async(queue_question_item, block=False)

            else:
                self._next_query_ids_colbert.pop(vllm_request_id)
                self._next_query_start_indices.pop(vllm_request_id)

            self._sent_question.pop(vllm_request_id)


@final
class EmptyClaimProcessor:
    _queue_to_bm25: StageOutputQueueInterface[QueueRequestMessage[LMClaimQuestionToken]]
    _queue_to_colbert_encoder: StageOutputQueueInterface[QueueRequestMessage[LMClaimQuestion]]
    _queue_to_driver: StageOutputQueueInterface[QueueRequestMessage[LMClaimQuestion]]
    _debug_log: InstanceComputeLatencyLogger

    _task_runner: AsyncConcurrentProcessor

    def __init__(
        self: Self,
        queue_to_bm25: StageOutputQueueInterface[QueueRequestMessage[LMClaimQuestionToken]],
        queue_to_colbert_encoder: StageOutputQueueInterface[QueueRequestMessage[LMClaimQuestion]],
        queue_to_driver: StageOutputQueueInterface[QueueRequestMessage[LMClaimQuestion]],
        debug_log: InstanceComputeLatencyLogger,
        max_running_requests: int | None = None
    ) -> None:

        self._queue_to_bm25 = queue_to_bm25
        self._queue_to_colbert_encoder = queue_to_colbert_encoder
        self._queue_to_driver = queue_to_driver
        self._debug_log = debug_log

        self._task_runner = AsyncConcurrentProcessor(max_running_requests)

    async def _process_empty_claim(
        self: Self,
        item: QueueRequestMessage[LMClaim]
    ) -> None:

        claim_query_token_msg = LMClaimQuestionToken(
            word="",
            claim_id=item.message.claim_id,
            question_id=0,
            last_token=True,
        )
        queue_token_item = QueueRequestMessage(
            user_request_id=item.user_request_id,
            message=claim_query_token_msg
        )

        logger.trace("Generated queue message {}".format(queue_token_item))

        self._debug_log.update_stage_queue_out((item.user_request_id, item.message.claim_id))
        await self._queue_to_bm25.put_async(queue_token_item, block=False)

        claim_query_msg = LMClaimQuestion(
            question="",
            claim_id=item.message.claim_id,
            question_id=0,
            last_question=True,
        )
        queue_question_item = QueueRequestMessage(
            user_request_id=item.user_request_id,
            message=claim_query_msg
        )

        logger.trace("Generated queue message {}".format(queue_question_item))

        self._debug_log.update_stage_queue_out((item.user_request_id, item.message.claim_id))
        await self._queue_to_colbert_encoder.put_async(queue_question_item, block=False)
        await self._queue_to_driver.put_async(queue_question_item, block=False)

    async def add_request(
        self: Self,
        item: QueueRequestMessage[LMClaim]
    ) -> None:

        await self._task_runner.add_task(self._process_empty_claim(item))


async def main(
    comm_params_data: bytes,
    engine_args: AsyncEngineArgs
) -> None:

    logging.disable(logging.INFO)
    logger.debug("Starting query generation stage")

    stage_comm = StageCommunicatorFactory.get_proxy_from_bytes(comm_params_data)
    stage_comm.initialize()

    debug_log = InstanceComputeLatencyLogger(stage_comm.get_instance_name())

    sampling_params = get_sampling_params()

    queue_from_claim_extraction = stage_comm.get_input_queue_interface(
        input_stage_name=FactoolQueueEndpoint.CLAIM_EXTRACTION,
        output_stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        index=0,
        item_type=QueueRequestMessage[LMClaim],
        serializer=QueueRequestMessageSerializer(LMClaim)
    )
    queue_to_bm25 = stage_comm.get_output_queue_interface(
        input_stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        output_stage_name=FactoolQueueEndpoint.BM25,
        index=0,
        item_type=QueueRequestMessage[LMClaimQuestionToken],
        serializer=QueueRequestMessageSerializer(LMClaimQuestionToken)
    )
    queue_to_colbert_encoder = stage_comm.get_output_queue_interface(
        input_stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        output_stage_name=FactoolQueueEndpoint.COLBERT_ENCODER,
        index=0,
        item_type=QueueRequestMessage[LMClaimQuestion],
        serializer=QueueRequestMessageSerializer(LMClaimQuestion)
    )
    queue_to_driver = stage_comm.get_output_queue_interface(
        input_stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        output_stage_name=FactoolQueueEndpoint.DRIVER,
        index=0,
        item_type=QueueRequestMessage[LMClaimQuestion],
        serializer=QueueRequestMessageSerializer(LMClaimQuestion)
    )

    generate_processor = GenerateProcessor(
        queue_to_bm25,
        queue_to_colbert_encoder,
        queue_to_driver,
        debug_log
    )
    empty_claim_processor = EmptyClaimProcessor(
        queue_to_bm25,
        queue_to_colbert_encoder,
        queue_to_driver,
        debug_log
    )
    generator = LLMGeneratorAsync(engine_args, max_running_requests=None)

    logger.debug("Query generation stage ready")
    stage_comm.signal_instance_ready()

    while True:
        try:
            await generator._task_runner.wait_for_open_task_slot()
            items = await queue_from_claim_extraction.get_batch_async(
                min_num_items=1,
                all_items=True,
                block=True,
                timeout=5
            )

            user_requests: list[VllmPromptRequest] = []
            for item in items:
                debug_log.update_stage_queue_in((item.user_request_id, item.message.claim_id))
                if item.message.claim == "":
                    await empty_claim_processor.add_request(item)
                else:
                    request = VllmPromptRequest(
                        user_request_id=item.user_request_id,
                        prompt="\n".join(
                            ["[claim]:", item.message.claim, "[response]:"]
                        ),
                        request_info=RequestInfo(item.message.claim_id)
                    )
                    user_requests.append(request)

            for user_request in user_requests:
                await generator.add_request(
                    QUERY_GENERATION_PROMPT,
                    sampling_params,
                    user_request,
                    generate_processor.generate_callback
                )

        except AsyncQueueEmptyException:
            pass

        if stage_comm.should_stop():
            break

        debug_log.update_log()
