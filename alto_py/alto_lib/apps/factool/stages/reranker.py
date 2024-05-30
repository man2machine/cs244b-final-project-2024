# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:27:25 2023

@author: Shahir
"""

import os
import asyncio
import logging
from typing import Self, NamedTuple

import torch
import numpy as np
from loguru import logger
import portalocker as pl

from colbert.infra import ColBERTConfig
from colbert.searcher import Searcher as ColBERTSearcher

from alto_lib.core.manager import (
    StageCommunicator, StageCommunicatorFactory, QueueRequestMessage, QueueRequestMessageSerializer,
    StageInputQueueInterface, StageOutputQueueInterface, AsyncQueueEmptyException
)
from alto_lib.core.async_utils import AsyncConcurrentProcessor
from alto_lib.core.app_endpoints import InstanceComputeLatencyLogger
from alto_lib.apps.factool.common.messages import (
    BM25ClaimQuestionResult, ColbertClaimQuestionTensor, RerankerClaimQuestionOutput, DocumentId
)
from alto_lib.utils import get_rel_pkg_path
from alto_lib.apps.factool.common import FactoolQueueEndpoint


class RerankerProcessor:
    class _QueryKey(NamedTuple):
        user_request_id: int
        claim_id: int
        question_id: int

    _stage_comm: StageCommunicator
    _start_barrier: asyncio.Barrier

    _queue_from_bm25: StageInputQueueInterface[QueueRequestMessage[BM25ClaimQuestionResult]]
    _queue_from_colbert_encoder: StageInputQueueInterface[QueueRequestMessage[ColbertClaimQuestionTensor]]
    _queue_to_driver: StageOutputQueueInterface[QueueRequestMessage[RerankerClaimQuestionOutput]]
    _debug_log: InstanceComputeLatencyLogger

    _colbert_searcher: ColBERTSearcher
    _in_flight_queries_bm25: dict[_QueryKey, torch.Tensor]
    _in_flight_queries_colbert: dict[_QueryKey, torch.Tensor]
    _query_lock: asyncio.Lock

    task_runner: AsyncConcurrentProcessor

    def __init__(
        self: Self,
        stage_comm: StageCommunicator,
        start_barrier: asyncio.Barrier,
        queue_from_bm25: StageInputQueueInterface[QueueRequestMessage[BM25ClaimQuestionResult]],
        queue_from_colbert_encoder: StageInputQueueInterface[QueueRequestMessage[ColbertClaimQuestionTensor]],
        queue_to_driver: StageOutputQueueInterface[QueueRequestMessage[RerankerClaimQuestionOutput]],
        colbert_searcher: ColBERTSearcher,
        debug_log: InstanceComputeLatencyLogger,
        max_running_requests: int | None = None
    ) -> None:

        self._stage_comm = stage_comm
        self._start_barrier = start_barrier
        self._queue_from_bm25 = queue_from_bm25
        self._queue_from_colbert_encoder = queue_from_colbert_encoder
        self._queue_to_driver = queue_to_driver
        self._colbert_searcher = colbert_searcher
        self._debug_log = debug_log

        self._in_flight_queries_bm25 = {}
        self._in_flight_queries_colbert = {}
        self._query_lock = asyncio.Lock()

        self.task_runner = AsyncConcurrentProcessor(max_running_requests)

    def _colbert_search(
        self: Self,
        embeddings: torch.Tensor,
        pids_to_search: torch.Tensor,
        k: int = 5
    ) -> tuple[list[int], list[int], list[float]]:

        pids_searched, ranks, scores = self._colbert_searcher.dense_search(
            embeddings,
            k=k,
            pids=(pids_to_search if len(pids_to_search) != 0 else None)
        )

        return pids_searched, ranks, scores

    def _process_completed_query(
        self: Self,
        search_query_id: _QueryKey,
        pids_to_search: torch.Tensor,
        embeddings: torch.Tensor
    ) -> None:

        pids_searched, ranks, scores = self._colbert_search(embeddings, pids_to_search)

        doc = []
        for pid, rank, score in zip(pids_searched, ranks, scores):
            doc.append(
                DocumentId(
                    rank=rank,
                    doc_id=pid,
                    score=score
                )
            )

        reranker_output_msg = RerankerClaimQuestionOutput(
            doc=doc,
            claim_id=search_query_id.claim_id,
            question_id=search_query_id.question_id,
        )
        queue_item = QueueRequestMessage(
            user_request_id=search_query_id.user_request_id,
            message=reranker_output_msg
        )

        logger.trace("Generated queue message {}".format(queue_item))

        self._debug_log.update_stage_queue_out((search_query_id.user_request_id, search_query_id.claim_id, search_query_id.question_id, 0))
        self._debug_log.update_stage_queue_out((search_query_id.user_request_id, search_query_id.claim_id, search_query_id.question_id, 1))

        self._queue_to_driver.put(queue_item, block=False)
        routing_key = (
            search_query_id.user_request_id,
            search_query_id.claim_id,
            search_query_id.question_id
        )
        self._queue_from_bm25.mark_key_finished(routing_key)
        self._queue_from_colbert_encoder.mark_key_finished(routing_key)

    async def _check_and_process_query(
        self: Self,
        search_query_id: _QueryKey
    ) -> None:

        await self._query_lock.acquire()
        if (search_query_id in self._in_flight_queries_bm25) and (search_query_id in self._in_flight_queries_colbert):
            pids_to_search = self._in_flight_queries_bm25.pop(search_query_id)
            embeddings = self._in_flight_queries_colbert.pop(search_query_id)
            self._query_lock.release()
            await self.task_runner.add_task(
                asyncio.to_thread(
                    self._process_completed_query,
                    search_query_id,
                    pids_to_search,
                    embeddings
                )
            )
        else:
            self._query_lock.release()

    async def bm25_data_loop(
        self: Self
    ) -> None:

        await self._start_barrier.wait()

        while True:
            try:
                items = await self._queue_from_bm25.get_batch_async(
                    min_num_items=1,
                    max_num_items=self.task_runner.get_num_open_task_slots(),
                    block=True,
                    timeout=5
                )

                for item in items:
                    self._debug_log.update_stage_queue_in((item.user_request_id, item.message.claim_id, item.message.question_id, 0))
                    msg = item.message
                    search_query_id = self._QueryKey(item.user_request_id, msg.claim_id, msg.question_id)
                    pids = torch.tensor([doc.doc_id for doc in msg.doc], dtype=torch.int32)
                    self._in_flight_queries_bm25[search_query_id] = pids

                    await self._check_and_process_query(search_query_id)

            except AsyncQueueEmptyException:
                pass

            if self._stage_comm.should_stop():
                break

            self._debug_log.update_log()

    async def colbert_data_loop(
        self: Self
    ) -> None:

        await self._start_barrier.wait()

        while True:
            try:
                await self.task_runner.wait_for_open_task_slot()
                items = await self._queue_from_colbert_encoder.get_batch_async(
                    min_num_items=1,
                    max_num_items=self.task_runner.get_num_open_task_slots(),
                    block=True,
                    timeout=5
                )

                for item in items:
                    self._debug_log.update_stage_queue_in((item.user_request_id, item.message.claim_id, item.message.question_id, 1))
                    msg = item.message
                    search_query_id = self._QueryKey(item.user_request_id, msg.claim_id, msg.question_id)
                    shape = (1, msg.num_query_tokens, msg.embedding_dim)
                    embeddings = torch.tensor(np.frombuffer(msg.data, dtype=np.float32).reshape(shape))
                    self._in_flight_queries_colbert[search_query_id] = embeddings

                    await self._check_and_process_query(search_query_id)

            except AsyncQueueEmptyException:
                pass

            if self._stage_comm.should_stop():
                break

            self._debug_log.update_log()


async def main(
    comm_params_data: bytes,
    model_serving_data_path: str
) -> None:

    logging.disable(logging.INFO)
    logger.debug("Starting reranker stage")

    stage_comm = StageCommunicatorFactory.get_proxy_from_bytes(comm_params_data)
    stage_comm.initialize()

    debug_log = InstanceComputeLatencyLogger(stage_comm.get_instance_name())

    queue_from_bm25 = stage_comm.get_input_queue_interface(
        input_stage_name=FactoolQueueEndpoint.BM25,
        output_stage_name=FactoolQueueEndpoint.RERANKER,
        index=0,
        item_type=QueueRequestMessage[BM25ClaimQuestionResult],
        serializer=QueueRequestMessageSerializer(BM25ClaimQuestionResult)
    )
    queue_from_colbert_encoder = stage_comm.get_input_queue_interface(
        input_stage_name=FactoolQueueEndpoint.COLBERT_ENCODER,
        output_stage_name=FactoolQueueEndpoint.RERANKER,
        index=0,
        item_type=QueueRequestMessage[ColbertClaimQuestionTensor],
        serializer=QueueRequestMessageSerializer(ColbertClaimQuestionTensor)
    )
    queue_to_driver = stage_comm.get_output_queue_interface(
        input_stage_name=FactoolQueueEndpoint.RERANKER,
        output_stage_name=FactoolQueueEndpoint.DRIVER,
        index=0,
        item_type=QueueRequestMessage[RerankerClaimQuestionOutput],
        serializer=QueueRequestMessageSerializer(RerankerClaimQuestionOutput)
    )

    with pl.Lock(
        get_rel_pkg_path("scripts/factool/reranker_torch_compile.lock"),
        mode='a',
        flags=pl.LockFlags.EXCLUSIVE
    ):
        colbert_tokenizer_fname = os.path.join(model_serving_data_path, "msmarco.psg.kldR2.nway64.ib__colbert-400000/")
        colbert_experiment = 'wiki23.nbits=2'
        colbert_search_config = ColBERTConfig(
            index_root=os.path.join(model_serving_data_path, "experiments/default/indexes"),
            experiment=colbert_experiment,
            load_collection_with_mmap=True,
            load_index_with_mmap=False,
            gpus=0
        )
        colbert_searcher = ColBERTSearcher(
            index=f"{colbert_experiment}.latest",
            checkpoint=colbert_tokenizer_fname,
            config=colbert_search_config
        )

    start_barrier = asyncio.Barrier(3)

    reranker_processor = RerankerProcessor(
        stage_comm=stage_comm,
        start_barrier=start_barrier,
        queue_from_bm25=queue_from_bm25,
        queue_from_colbert_encoder=queue_from_colbert_encoder,
        queue_to_driver=queue_to_driver,
        colbert_searcher=colbert_searcher,
        debug_log=debug_log,
        max_running_requests=4
    )

    tasks = [
        asyncio.create_task(reranker_processor.bm25_data_loop()),
        asyncio.create_task(reranker_processor.colbert_data_loop())
    ]

    await start_barrier.wait()

    logger.debug("Reranker stage ready")
    stage_comm.signal_instance_ready()

    await asyncio.wait(tasks)
