# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:56:38 2023

@author: Shahir
"""

import os
import asyncio
import logging
import threading
from typing import Self, NamedTuple, final

from scipy import sparse  # type: ignore

from loguru import logger

from bm25_index import Bm25Index

from alto_lib.core.manager import (
    StageCommunicatorFactory, QueueRequestMessage, QueueRequestMessageSerializer,
    StageOutputQueueInterface, AsyncQueueEmptyException
)
from alto_lib.core.async_utils import AsyncConcurrentProcessor
from alto_lib.core.app_endpoints import InstanceComputeLatencyLogger
from alto_lib.apps.factool.common.messages import LMClaimQuestionToken, BM25ClaimQuestionResult, DocumentId
from alto_lib.apps.factool.common import FactoolQueueEndpoint


@final
class BM25Processor:
    class _QueryKey(NamedTuple):
        user_request_id: int
        claim_id: int
        query_id: int

    _bm25_index: Bm25Index
    _queue_to_reranker: StageOutputQueueInterface[QueueRequestMessage[BM25ClaimQuestionResult]]
    _debug_log: InstanceComputeLatencyLogger

    _scores_per_query_id: dict[_QueryKey, sparse.csr_array]
    _locks_per_query_id: dict[_QueryKey, threading.Lock]
    _tasks_per_query_id: dict[_QueryKey, set[asyncio.Task]]
    _outer_lock: threading.Lock

    task_runner: AsyncConcurrentProcessor

    def __init__(
        self: Self,
        bm25_index: Bm25Index,
        queue_to_reranker: StageOutputQueueInterface[QueueRequestMessage[BM25ClaimQuestionResult]],
        debug_log: InstanceComputeLatencyLogger,
        max_running_requests: int | None = None
    ) -> None:

        self._bm25_index = bm25_index
        self._queue_to_reranker = queue_to_reranker
        self._debug_log = debug_log

        self._scores_per_query_id = {}
        self._locks_per_query_id = {}
        self._tasks_per_query_id = {}
        self._outer_lock = threading.Lock()

        self.task_runner = AsyncConcurrentProcessor(max_running_requests)

    def _process_claim_question(
        self: Self,
        item: QueueRequestMessage[LMClaimQuestionToken]
    ) -> None:

        query_token = item.message
        search_query_id = self._QueryKey(item.user_request_id, item.message.claim_id, item.message.question_id)
        preproc_tokens = self._bm25_index.preprocess(
            query_token.word,
            self._bm25_index.stopwords
        )

        if len(preproc_tokens) > 1:
            raise ValueError(
                f"BM25 returned multiple tokens for {query_token.word}"
            )

        if len(preproc_tokens) == 1:
            self._outer_lock.acquire()
            if search_query_id not in self._locks_per_query_id:
                self._locks_per_query_id[search_query_id] = threading.Lock()
            self._outer_lock.release()
            
            token_scores = self._bm25_index.get_token_scores(preproc_tokens[0])

            self._locks_per_query_id[search_query_id].acquire()
            if token_scores is not None:
                if search_query_id not in self._scores_per_query_id:
                    self._scores_per_query_id[search_query_id] = token_scores
                else:
                    self._scores_per_query_id[search_query_id] += token_scores
            self._locks_per_query_id[search_query_id].release()

        if query_token.last_token:
            if search_query_id not in self._scores_per_query_id:
                pids = []
                scores = []
            else:
                pids, scores = self._bm25_index.get_topk(
                    self._scores_per_query_id[search_query_id],
                    k=200
                )

            docs = []
            for rank, (pid, score) in enumerate(zip(pids, scores)):
                docs.append(
                    DocumentId(
                        rank=rank,
                        doc_id=pid,
                        score=score
                    )
                )

            bm25_result_msg = BM25ClaimQuestionResult(
                doc=docs,
                claim_id=query_token.claim_id,
                question_id=query_token.question_id,
            )
            queue_item = QueueRequestMessage(
                user_request_id=item.user_request_id,
                message=bm25_result_msg
            )

            logger.trace("Generated queue message {}".format(queue_item))

            self._debug_log.update_stage_queue_out((item.user_request_id, item.message.claim_id, item.message.question_id))
            routing_key = (item.user_request_id, item.message.claim_id, item.message.question_id)
            self._queue_to_reranker.put(queue_item, block=False, key=routing_key)
            
            self._outer_lock.acquire()
            self._scores_per_query_id.pop(search_query_id, None)
            self._locks_per_query_id.pop(search_query_id, None)
            self._outer_lock.release()

    async def add_request(
        self: Self,
        item: QueueRequestMessage[LMClaimQuestionToken]
    ) -> None:

        search_query_id = self._QueryKey(item.user_request_id, item.message.claim_id, item.message.question_id)
        new_query_id = (search_query_id not in self._tasks_per_query_id)
        if not item.message.last_token:
            task = await self.task_runner.add_task(asyncio.to_thread(self._process_claim_question, item))
            if new_query_id:
                self._tasks_per_query_id[search_query_id] = set()
            self._tasks_per_query_id[search_query_id].add(task)
        else:
            if not new_query_id:
                await asyncio.gather(*self._tasks_per_query_id.pop(search_query_id))
            await self.task_runner.add_task(asyncio.to_thread(self._process_claim_question, item))


async def main(
    comm_params_data: bytes,
    model_serving_data_path: str
) -> None:

    logging.disable(logging.INFO)
    logger.debug("Starting BM25 stage")

    stage_comm = StageCommunicatorFactory.get_proxy_from_bytes(comm_params_data)
    stage_comm.initialize()

    debug_log = InstanceComputeLatencyLogger(stage_comm.get_instance_name())

    bm25_index = Bm25Index(
        collection_path=None,
        index_root=os.path.join(model_serving_data_path, "bm25_indexes"),
        index_name="wiki23"
    )

    queue_from_query_generation = stage_comm.get_input_queue_interface(
        input_stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        output_stage_name=FactoolQueueEndpoint.BM25,
        index=0,
        item_type=QueueRequestMessage[LMClaimQuestionToken],
        serializer=QueueRequestMessageSerializer(LMClaimQuestionToken)
    )
    queue_to_reranker = stage_comm.get_output_queue_interface(
        input_stage_name=FactoolQueueEndpoint.BM25,
        output_stage_name=FactoolQueueEndpoint.RERANKER,
        index=0,
        item_type=QueueRequestMessage[BM25ClaimQuestionResult],
        serializer=QueueRequestMessageSerializer(BM25ClaimQuestionResult)
    )

    bm25_processor = BM25Processor(
        bm25_index=bm25_index,
        queue_to_reranker=queue_to_reranker,
        debug_log=debug_log,
        max_running_requests=4
    )

    logger.debug("BM25 stage ready")
    stage_comm.signal_instance_ready()

    while True:
        try:
            await bm25_processor.task_runner.wait_for_open_task_slot()
            items = await queue_from_query_generation.get_batch_async(
                min_num_items=1,
                max_num_items=bm25_processor.task_runner.get_num_open_task_slots(),
                block=True,
                timeout=5
            )

            for item in items:
                if item.message.last_token:
                    debug_log.update_stage_queue_in((item.user_request_id, item.message.claim_id, item.message.question_id))
                await bm25_processor.add_request(item)

        except AsyncQueueEmptyException:
            pass

        if stage_comm.should_stop():
            break

        debug_log.update_log()
