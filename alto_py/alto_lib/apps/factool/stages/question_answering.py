# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:29:05 2023

@author: Shahir
"""

import logging
from typing import Self, final

from loguru import logger

from vllm import AsyncEngineArgs  # type: ignore

from alto_lib.core.manager import (
    StageCommunicatorFactory, QueueRequestMessage, QueueRequestMessageSerializer,
    StageInputQueueInterface, StageOutputQueueInterface, AsyncQueueEmptyException
)
from alto_lib.core.app_endpoints import InstanceComputeLatencyLogger
from alto_lib.apps.factool.common.messages import LMGeneration, FactoolRequest
from alto_lib.apps.factool.common.prompts import QUESTION_ANSWERING_PROMPT
from alto_lib.apps.factool.common.lm_utils import (
    get_sampling_params, VllmPromptRequest, AsyncGenerateCallbackInput, LLMGeneratorAsync
)
from alto_lib.apps.factool.common import FactoolQueueEndpoint


@final
class GenerateProcessor:
    _queue_to_claim_extraction: StageOutputQueueInterface[QueueRequestMessage[LMGeneration]]
    _queue_to_driver: StageOutputQueueInterface[QueueRequestMessage[LMGeneration]]
    _queue_from_driver: StageInputQueueInterface[QueueRequestMessage[FactoolRequest]]
    _debug_log: InstanceComputeLatencyLogger

    def __init__(
        self: Self,
        queue_to_claim_extraction: StageOutputQueueInterface[QueueRequestMessage[LMGeneration]],
        queue_to_driver: StageOutputQueueInterface[QueueRequestMessage[LMGeneration]],
        queue_from_driver: StageInputQueueInterface[QueueRequestMessage[FactoolRequest]],
        debug_log: InstanceComputeLatencyLogger
    ) -> None:

        self._queue_to_claim_extraction = queue_to_claim_extraction
        self._queue_to_driver = queue_to_driver
        self._queue_from_driver = queue_from_driver
        self._debug_log = debug_log

    async def generate_callback(
        self: Self,
        data: AsyncGenerateCallbackInput
    ) -> None:

        if data.stream_finished:
            assert data.completion_output is not None
            text = data.completion_output.text.strip().replace("<|im_end|>", "")

            generation_msg = LMGeneration(generation=text)
            queue_item = QueueRequestMessage(
                user_request_id=data.user_request_id,
                message=generation_msg
            )

            logger.trace("Generated queue message {}".format(queue_item))

            self._queue_from_driver.mark_item_finished()

            self._debug_log.update_stage_queue_out(data.user_request_id)
            await self._queue_to_claim_extraction.put_async(queue_item, block=False)
            await self._queue_to_driver.put_async(queue_item, block=False)


async def main(
    comm_params_data: bytes,
    engine_args: AsyncEngineArgs
) -> None:

    logging.disable(logging.INFO)
    logger.debug("Starting question answering stage")

    stage_comm = StageCommunicatorFactory.get_proxy_from_bytes(comm_params_data)
    stage_comm.initialize()

    debug_log = InstanceComputeLatencyLogger(stage_comm.get_instance_name())

    sampling_params = get_sampling_params()

    queue_from_driver = stage_comm.get_input_queue_interface(
        input_stage_name=FactoolQueueEndpoint.DRIVER,
        output_stage_name=FactoolQueueEndpoint.QUESTION_ANSWERING,
        index=0,
        item_type=QueueRequestMessage[FactoolRequest],
        serializer=QueueRequestMessageSerializer(FactoolRequest)
    )
    queue_to_claim_extraction = stage_comm.get_output_queue_interface(
        input_stage_name=FactoolQueueEndpoint.QUESTION_ANSWERING,
        output_stage_name=FactoolQueueEndpoint.CLAIM_EXTRACTION,
        index=0,
        item_type=QueueRequestMessage[LMGeneration],
        serializer=QueueRequestMessageSerializer(LMGeneration)
    )
    queue_to_driver = stage_comm.get_output_queue_interface(
        input_stage_name=FactoolQueueEndpoint.QUESTION_ANSWERING,
        output_stage_name=FactoolQueueEndpoint.DRIVER,
        index=0,
        item_type=QueueRequestMessage[LMGeneration],
        serializer=QueueRequestMessageSerializer(LMGeneration)
    )

    generate_processor = GenerateProcessor(
        queue_to_claim_extraction,
        queue_to_driver,
        queue_from_driver,
        debug_log
    )
    generator = LLMGeneratorAsync(engine_args, max_running_requests=None)

    logger.debug("Question answering stage ready")
    stage_comm.signal_instance_ready()

    while True:
        try:
            await generator._task_runner.wait_for_open_task_slot()
            items = await queue_from_driver.get_batch_async(
                min_num_items=1,
                all_items=True,
                block=True,
                timeout=5
            )

            user_requests: list[VllmPromptRequest] = [
                VllmPromptRequest(
                    user_request_id=item.user_request_id,
                    prompt=item.message.query
                ) for item in items
            ]

            for user_request in user_requests:
                debug_log.update_stage_queue_in(user_request.user_request_id)
                await generator.add_request(
                    QUESTION_ANSWERING_PROMPT,
                    sampling_params,
                    user_request,
                    generate_processor.generate_callback
                )

        except AsyncQueueEmptyException:
            pass

        if stage_comm.should_stop():
            break

        debug_log.update_log()
