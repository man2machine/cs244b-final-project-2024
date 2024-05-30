# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:33:35 2023

@author: Shahir
"""

import re
import logging
import collections
from typing import Self, final

from loguru import logger

from vllm import AsyncEngineArgs  # type: ignore

from alto_lib.core.manager import (
    StageCommunicatorFactory, QueueRequestMessage, QueueRequestMessageSerializer,
    StageOutputQueueInterface, AsyncQueueEmptyException
)
from alto_lib.core.app_endpoints import InstanceComputeLatencyLogger
from alto_lib.apps.factool.common.messages import LMClaim, LMGeneration
from alto_lib.apps.factool.common.prompts import CLAIM_EXTRACTION_PROMPT
from alto_lib.apps.factool.common.lm_utils import (
    get_sampling_params, VllmPromptRequest, AsyncGenerateCallbackInput, LLMGeneratorAsync
)
from alto_lib.apps.factool.common import FactoolQueueEndpoint


@final
class GenerateProcessor:
    _queue_to_query_generation: StageOutputQueueInterface[QueueRequestMessage[LMClaim]]
    _queue_to_driver: StageOutputQueueInterface[QueueRequestMessage[LMClaim]]
    _debug_log: InstanceComputeLatencyLogger

    _next_claim_ids: dict[int, int]
    _next_claim_start_indices: dict[int, int]

    _sent_claim: dict[int, bool]

    def __init__(
        self: Self,
        queue_to_query_generation: StageOutputQueueInterface[QueueRequestMessage[LMClaim]],
        queue_to_driver: StageOutputQueueInterface[QueueRequestMessage[LMClaim]],
        debug_log: InstanceComputeLatencyLogger
    ) -> None:

        self._queue_to_query_generation = queue_to_query_generation
        self._queue_to_driver = queue_to_driver
        self._debug_log = debug_log

        self._next_claim_ids = collections.defaultdict(int)
        self._next_claim_start_indices = collections.defaultdict(int)

        self._sent_claim = collections.defaultdict(bool)

    async def generate_callback(
        self: Self,
        data: AsyncGenerateCallbackInput
    ) -> None:

        vllm_request_id = data.vllm_request_id
        assert data.completion_output is not None

        all_text = data.completion_output.text
        all_text = all_text.replace("<|im_end|>", "")

        # region - parse and send claims
        if vllm_request_id in self._next_claim_start_indices:
            new_text = all_text[self._next_claim_start_indices[vllm_request_id]:]
        else:
            new_text = all_text
            while new_text and (new_text[0] == " "):
                self._next_claim_start_indices[vllm_request_id] += 1
                new_text = new_text[1:]

        if len(new_text) != 0:
            split_indices = [m.start() for m in re.finditer("\n", new_text)]

            if data.stream_finished:
                split_indices.append(len(new_text))

            claim_start_index = 0
            for i in range(len(split_indices)):
                last_split = (i == (len(split_indices) - 1))
                last_claim = (last_split and data.stream_finished)

                claim_text = new_text[claim_start_index:split_indices[i]]

                if len(claim_text):
                    claim_msg = LMClaim(
                        claim=claim_text,
                        claim_id=self._next_claim_ids[vllm_request_id],
                        last_claim=last_claim
                    )
                    queue_item = QueueRequestMessage(
                        user_request_id=data.user_request_id,
                        message=claim_msg
                    )

                    logger.trace("Generated queue message {}".format(queue_item))

                    self._debug_log.update_stage_queue_out(data.user_request_id)
                    await self._queue_to_query_generation.put_async(queue_item, block=False)
                    await self._queue_to_driver.put_async(queue_item, block=False)

                    self._sent_claim[vllm_request_id] = True
                    self._next_claim_ids[vllm_request_id] += 1

                claim_start_index += (split_indices[i] + 1)

            self._next_claim_start_indices[vllm_request_id] += claim_start_index

        # endregion

        if data.stream_finished:
            if not self._sent_claim[vllm_request_id]:
                logger.trace("Did not send token, sending blank token now")
                claim_msg = LMClaim(
                    claim="",
                    claim_id=0,
                    last_claim=True
                )
                queue_item = QueueRequestMessage(
                    user_request_id=data.user_request_id,
                    message=claim_msg
                )

                logger.trace("Generated queue message {}".format(queue_item))

                self._debug_log.update_stage_queue_out(data.user_request_id)
                await self._queue_to_query_generation.put_async(queue_item, block=False)
                await self._queue_to_driver.put_async(queue_item, block=False)

            else:
                self._next_claim_ids.pop(vllm_request_id)
                self._next_claim_start_indices.pop(vllm_request_id)

            self._sent_claim.pop(vllm_request_id)


async def main(
    comm_params_data: bytes,
    engine_args: AsyncEngineArgs
) -> None:

    logging.disable(logging.INFO)
    logger.debug("Starting claim extraction stage")

    stage_comm = StageCommunicatorFactory.get_proxy_from_bytes(comm_params_data)
    stage_comm.initialize()

    debug_log = InstanceComputeLatencyLogger(stage_comm.get_instance_name())

    sampling_params = get_sampling_params()

    queue_from_question_answering = stage_comm.get_input_queue_interface(
        input_stage_name=FactoolQueueEndpoint.QUESTION_ANSWERING,
        output_stage_name=FactoolQueueEndpoint.CLAIM_EXTRACTION,
        index=0,
        item_type=QueueRequestMessage[LMGeneration],
        serializer=QueueRequestMessageSerializer(LMGeneration)
    )
    queue_to_query_generation = stage_comm.get_output_queue_interface(
        input_stage_name=FactoolQueueEndpoint.CLAIM_EXTRACTION,
        output_stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        index=0,
        item_type=QueueRequestMessage[LMClaim],
        serializer=QueueRequestMessageSerializer(LMClaim)
    )
    queue_to_driver = stage_comm.get_output_queue_interface(
        input_stage_name=FactoolQueueEndpoint.CLAIM_EXTRACTION,
        output_stage_name=FactoolQueueEndpoint.DRIVER,
        index=0,
        item_type=QueueRequestMessage[LMClaim],
        serializer=QueueRequestMessageSerializer(LMClaim)
    )

    generate_processor = GenerateProcessor(
        queue_to_query_generation,
        queue_to_driver,
        debug_log
    )
    generator = LLMGeneratorAsync(engine_args, max_running_requests=None)

    logger.debug("Claim extraction stage ready")
    stage_comm.signal_instance_ready()

    while True:
        try:
            await generator._task_runner.wait_for_open_task_slot()
            items = await queue_from_question_answering.get_batch_async(
                min_num_items=1,
                all_items=True,
                block=True,
                timeout=5
            )

            user_requests: list[VllmPromptRequest] = [
                VllmPromptRequest(
                    user_request_id=item.user_request_id,
                    prompt="\n".join(
                        ["[text]:", item.message.generation, "[response]:"]
                    )
                ) for item in items
            ]

            for user_request in user_requests:
                debug_log.update_stage_queue_in(user_request.user_request_id)
                await generator.add_request(
                    CLAIM_EXTRACTION_PROMPT,
                    sampling_params,
                    user_request,
                    generate_processor.generate_callback
                )

        except AsyncQueueEmptyException:
            pass

        if stage_comm.should_stop():
            break

        debug_log.update_log()
