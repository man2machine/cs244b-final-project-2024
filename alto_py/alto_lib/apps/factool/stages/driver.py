# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 22:11:03 2023

@author: Shahir
"""

import asyncio
import logging
from typing import Self, NamedTuple

from loguru import logger

from alto_lib.core.manager import (
    StageCommunicator, StageCommunicatorFactory, QueueRequestMessage, QueueRequestMessageSerializer,
    AsyncQueue, AsyncQueueEmptyException
)
from alto_lib.core.app_endpoints import (
    AppControlServerNode, AppRequestHandlerServerNode, AppResponseHandlerServerNode,
    AppRequest, AppResponse, InstanceComputeLatencyLogger, IPv4Address
)
from alto_lib.apps.factool.common.messages import (
    FactoolRequest, LMGeneration, LMClaim, LMClaimQuestion, RerankerClaimQuestionOutput, FactoolResponse,
    VerifiedClaim, QuestionResult, DocumentId
)
from alto_lib.apps.factool.common import FactoolQueueEndpoint


STOP_POLL_INTERVAL: float = 5


class FactoolResponseAggregator:
    class _ClaimKey(NamedTuple):
        user_request_id: int
        claim_id: int

    class _QueryKey(NamedTuple):
        user_request_id: int
        claim_id: int
        question_id: int

    _generations: dict[int, str]
    _claims: dict[_ClaimKey, str]
    _queries: dict[_QueryKey, str]
    _reranker_output: dict[_QueryKey, list[DocumentId]]
    _num_claims: dict[int, int]
    _num_queries: dict[_ClaimKey, int]

    def __init__(
        self: Self
    ) -> None:

        self._generations = {}
        self._claims = {}
        self._queries = {}
        self._reranker_output = {}
        self._num_claims = {}
        self._num_queries = {}

    def append_answer(
        self: Self,
        user_request_id: int,
        answer: LMGeneration
    ) -> None:

        self._generations[user_request_id] = answer.generation

    def append_claim(
        self: Self,
        user_request_id: int,
        claim: LMClaim
    ) -> None:

        self._claims[self._ClaimKey(user_request_id, claim.claim_id)] = claim.claim
        if claim.last_claim:
            self._num_claims[user_request_id] = claim.claim_id + 1

    def append_query(
        self: Self,
        user_request_id: int,
        query: LMClaimQuestion
    ) -> None:

        self._queries[self._QueryKey(user_request_id, query.claim_id, query.question_id)] = query.question
        if query.last_question:
            self._num_queries[self._ClaimKey(user_request_id, query.claim_id)] = query.question_id + 1

    def append_reranker_output(
        self: Self,
        user_request_id: int,
        output: RerankerClaimQuestionOutput
    ) -> None:

        self._reranker_output[self._QueryKey(user_request_id, output.claim_id, output.question_id)] = output.doc

    def aggregate_outputs(
        self: Self,
        user_request_id: int
    ) -> FactoolResponse | None:

        if user_request_id not in self._generations:
            return None

        if user_request_id not in self._num_claims:
            return None

        for claim_id in range(self._num_claims[user_request_id]):
            if self._ClaimKey(user_request_id, claim_id) not in self._claims:
                return None

            if self._ClaimKey(user_request_id, claim_id) not in self._num_queries:
                return None

            for question_id in range(self._num_queries[self._ClaimKey(user_request_id, claim_id)]):
                if self._QueryKey(user_request_id, claim_id, question_id) not in self._queries:
                    return None

                if self._QueryKey(user_request_id, claim_id, question_id) not in self._reranker_output:
                    return None

        factool_response_claims = []
        for claim_id in range(self._num_claims[user_request_id]):
            verified_claim_questions = []
            for question_id in range(self._num_queries[self._ClaimKey(user_request_id, claim_id)]):
                question_result = QuestionResult(
                    question=self._queries.pop(self._QueryKey(user_request_id, claim_id, question_id)),
                    doc=self._reranker_output.pop(self._QueryKey(user_request_id, claim_id, question_id))
                )
                verified_claim_questions.append(question_result)

            self._num_queries.pop(self._ClaimKey(user_request_id, claim_id))

            verified_claim = VerifiedClaim(
                claim=self._claims.pop(self._ClaimKey(user_request_id, claim_id)),
                questions=verified_claim_questions
            )
            factool_response_claims.append(verified_claim)

        self._num_claims.pop(user_request_id)

        response = FactoolResponse(
            generation=self._generations.pop(user_request_id),
            claims=factool_response_claims
        )

        return response


async def _source_loop(
    stage_comm: StageCommunicator,
    queue_source_to_driver_local: AsyncQueue[QueueRequestMessage[FactoolRequest]],
    start_barrier: asyncio.Barrier
) -> None:

    queue_to_question_answering = stage_comm.get_output_queue_interface(
        input_stage_name=FactoolQueueEndpoint.DRIVER,
        output_stage_name=FactoolQueueEndpoint.QUESTION_ANSWERING,
        index=0,
        item_type=QueueRequestMessage[FactoolRequest],
        serializer=QueueRequestMessageSerializer(FactoolRequest)
    )

    await start_barrier.wait()

    logger.debug("Driver stage ready")
    stage_comm.signal_instance_ready()

    while True:
        try:
            items = await queue_source_to_driver_local.get_batch_async(
                min_num_items=1,
                all_items=True,
                timeout=STOP_POLL_INTERVAL
            )

            for item in items:
                logger.trace("Received {}".format(item))

            await queue_to_question_answering.put_batch_async(items, block=True, timeout=5)

        except AsyncQueueEmptyException:
            pass

        if stage_comm.should_stop():
            break


async def _answer_loop(
    stage_comm: StageCommunicator,
    response_aggregator: FactoolResponseAggregator,
    start_barrier: asyncio.Barrier
) -> None:

    queue_from_question_answering = stage_comm.get_input_queue_interface(
        input_stage_name=FactoolQueueEndpoint.QUESTION_ANSWERING,
        output_stage_name=FactoolQueueEndpoint.DRIVER,
        index=0,
        item_type=QueueRequestMessage[LMGeneration],
        serializer=QueueRequestMessageSerializer(LMGeneration)
    )

    await start_barrier.wait()

    while True:
        try:
            items = await queue_from_question_answering.get_batch_async(
                min_num_items=1,
                all_items=True,
                block=True,
                timeout=STOP_POLL_INTERVAL
            )
            for item in items:
                response_aggregator.append_answer(
                    user_request_id=item.user_request_id,
                    answer=item.message
                )

                queue_from_question_answering.mark_item_finished()

        except AsyncQueueEmptyException:
            pass

        if stage_comm.should_stop():
            break


async def _claim_loop(
    stage_comm: StageCommunicator,
    response_aggregator: FactoolResponseAggregator,
    start_barrier: asyncio.Barrier
) -> None:

    queue_from_claim_extraction = stage_comm.get_input_queue_interface(
        input_stage_name=FactoolQueueEndpoint.CLAIM_EXTRACTION,
        output_stage_name=FactoolQueueEndpoint.DRIVER,
        index=0,
        item_type=QueueRequestMessage[LMClaim],
        serializer=QueueRequestMessageSerializer(LMClaim)
    )

    await start_barrier.wait()

    while True:
        try:
            items = await queue_from_claim_extraction.get_batch_async(
                min_num_items=1,
                all_items=True,
                block=True,
                timeout=STOP_POLL_INTERVAL
            )
            for item in items:
                response_aggregator.append_claim(
                    user_request_id=item.user_request_id,
                    claim=item.message
                )

                queue_from_claim_extraction.mark_item_finished()

        except AsyncQueueEmptyException:
            pass

        if stage_comm.should_stop():
            break


async def _query_loop(
    stage_comm: StageCommunicator,
    response_aggregator: FactoolResponseAggregator,
    start_barrier: asyncio.Barrier
) -> None:

    queue_from_query_generation = stage_comm.get_input_queue_interface(
        input_stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        output_stage_name=FactoolQueueEndpoint.DRIVER,
        index=0,
        item_type=QueueRequestMessage[LMClaimQuestion],
        serializer=QueueRequestMessageSerializer(LMClaimQuestion)
    )

    await start_barrier.wait()

    while True:
        try:
            items = await queue_from_query_generation.get_batch_async(
                min_num_items=1,
                all_items=True,
                block=True,
                timeout=STOP_POLL_INTERVAL
            )
            for item in items:
                response_aggregator.append_query(
                    user_request_id=item.user_request_id,
                    query=item.message
                )

                queue_from_query_generation.mark_item_finished()

        except AsyncQueueEmptyException:
            pass

        if stage_comm.should_stop():
            break


async def _reranker_loop(
    stage_comm: StageCommunicator,
    response_aggregator: FactoolResponseAggregator,
    response_node: AppResponseHandlerServerNode,
    start_barrier: asyncio.Barrier,
    debug_log: InstanceComputeLatencyLogger
) -> None:

    queue_from_reranker = stage_comm.get_input_queue_interface(
        input_stage_name=FactoolQueueEndpoint.RERANKER,
        output_stage_name=FactoolQueueEndpoint.DRIVER,
        index=0,
        item_type=QueueRequestMessage[RerankerClaimQuestionOutput],
        serializer=QueueRequestMessageSerializer(RerankerClaimQuestionOutput)
    )

    await start_barrier.wait()

    while True:
        try:
            items = await queue_from_reranker.get_batch_async(
                min_num_items=1,
                all_items=True,
                block=True,
                timeout=STOP_POLL_INTERVAL
            )
            for item in items:
                response_aggregator.append_reranker_output(
                    user_request_id=item.user_request_id,
                    output=item.message
                )

                response = response_aggregator.aggregate_outputs(item.user_request_id)
                if response:
                    debug_log.update_stage_queue_out(item.user_request_id)
                    await response_node.add_response(
                        AppResponse(
                            user_request_id=item.user_request_id,
                            message=response.to_bytes()
                        )
                    )

                    logger.trace("Generated response {}".format(response))

                queue_from_reranker.mark_item_finished()

        except AsyncQueueEmptyException:
            pass

        if stage_comm.should_stop():
            break
        
        debug_log.update_log()


async def _request_loop(
    request_node: AppRequestHandlerServerNode,
    start_barrier: asyncio.Barrier
) -> None:

    await start_barrier.wait()
    await request_node.run()


async def _response_loop(
    response_node: AppResponseHandlerServerNode,
    start_barrier: asyncio.Barrier
) -> None:

    await start_barrier.wait()
    await response_node.run()


async def main(
    comm_params_data: bytes,
    control_address: IPv4Address,
    request_recv_address: IPv4Address,
    response_send_address: IPv4Address
) -> None:

    logging.disable(logging.INFO)
    logger.debug("Starting driver stage")

    stage_comm = StageCommunicatorFactory.get_proxy_from_bytes(comm_params_data)
    stage_comm.initialize()
    
    debug_log = InstanceComputeLatencyLogger(stage_comm.get_instance_name())

    response_aggregator = FactoolResponseAggregator()
    queue_source_to_driver_local: AsyncQueue[QueueRequestMessage[FactoolRequest]] = AsyncQueue()

    def request_recv_callback(
        request: AppRequest
    ) -> None:

        item = QueueRequestMessage(
            user_request_id=request.user_request_id,
            message=FactoolRequest.from_bytes(request.message)
        )
        debug_log.update_stage_queue_in(item.user_request_id)
        queue_source_to_driver_local.put_nowait(item)

    control_node = AppControlServerNode(
        control_address=control_address,
        name="Factool Control Node Driver Stage",
        stage_comm=stage_comm
    )
    request_node = AppRequestHandlerServerNode(
        request_recv_address=request_recv_address,
        name="Factool Request Handler Driver Stage",
        request_recv_callback=request_recv_callback,
        stage_comm=stage_comm
    )
    response_node = AppResponseHandlerServerNode(
        response_send_address=response_send_address,
        name="Factool Response Handler Driver Stage",
        stage_comm=stage_comm
    )

    tasks = [
        asyncio.create_task(control_node.initialize()),
        asyncio.create_task(request_node.initialize()),
        asyncio.create_task(response_node.initialize())
    ]

    await asyncio.wait(tasks)

    start_barrier = asyncio.Barrier(7)

    tasks = [
        asyncio.create_task(control_node.run()),  # not part of the barrier, should start immediately
        asyncio.create_task(_source_loop(stage_comm, queue_source_to_driver_local, start_barrier)),
        asyncio.create_task(_answer_loop(stage_comm, response_aggregator, start_barrier)),
        asyncio.create_task(_claim_loop(stage_comm, response_aggregator, start_barrier)),
        asyncio.create_task(_query_loop(stage_comm, response_aggregator, start_barrier)),
        asyncio.create_task(_reranker_loop(stage_comm, response_aggregator, response_node, start_barrier, debug_log)),
        asyncio.create_task(_request_loop(request_node, start_barrier)),
        asyncio.create_task(_response_loop(response_node, start_barrier))
    ]

    await asyncio.wait(tasks)
