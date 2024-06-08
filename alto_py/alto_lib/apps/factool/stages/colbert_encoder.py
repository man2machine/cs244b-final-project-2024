# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 04:05:26 2023

@author: Shahir
"""

import os
import asyncio
import logging
from dataclasses import dataclass
from collections.abc import Callable
from typing import Self, final

import numpy as np

import torch
import torch.nn as nn

from transformers import AutoTokenizer, PreTrainedTokenizerBase  # type: ignore
from transformers.utils import PaddingStrategy, TensorType  # type: ignore

from loguru import logger

from colbert.modeling.base_colbert import BaseColBERT

from alto_lib.core.manager import (
    StageCommunicatorFactory, QueueRequestMessage, QueueRequestMessageSerializer,
    StageOutputQueueInterface, AsyncQueueEmptyException
)
from alto_lib.core.async_utils import AsyncConcurrentProcessor
from alto_lib.core.app_endpoints import InstanceComputeLatencyLogger
from alto_lib.apps.factool.common.messages import LMClaimQuestion, ColbertClaimQuestionTensor
from alto_lib.apps.factool.common import FactoolQueueEndpoint


torch.set_float32_matmul_precision('high')


ColBERTCallable = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(kw_only=True, frozen=True, slots=True)
class ColbertQueryEncoderConfig:
    max_length: int
    q_marker_token_id: int
    mask_token_id: int
    pad_token_id: int
    device: str


class ColBERT(nn.Module):
    lm: nn.Module
    linear: nn.Module

    def __init__(
        self: Self,
        lm: nn.Module,
        linear: nn.Module
    ) -> None:

        super().__init__()

        self.lm = lm
        self.linear = linear

    __call__: ColBERTCallable

    def forward(
        self: Self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:

        return self.linear(self.lm(input_ids, attention_mask)[0])


@final
class ColbertProcessor:
    _colbert_tokenizer: PreTrainedTokenizerBase
    _colbert_query_encoder: ColBERTCallable
    _colbert_query_encoder_config: ColbertQueryEncoderConfig
    _queue_to_reranker: StageOutputQueueInterface[QueueRequestMessage[ColbertClaimQuestionTensor]]
    _debug_log: InstanceComputeLatencyLogger

    task_runner: AsyncConcurrentProcessor

    def __init__(
        self: Self,
        colbert_tokenizer: PreTrainedTokenizerBase,
        colbert_query_encoder: ColBERTCallable,
        colbert_query_encoder_config: ColbertQueryEncoderConfig,
        queue_to_reranker: StageOutputQueueInterface[QueueRequestMessage[ColbertClaimQuestionTensor]],
        debug_log: InstanceComputeLatencyLogger,
        max_running_requests: int | None = None
    ) -> None:

        self._colbert_tokenizer = colbert_tokenizer
        self._colbert_query_encoder = colbert_query_encoder
        self._colbert_query_encoder_config = colbert_query_encoder_config
        self._queue_to_reranker = queue_to_reranker
        self._debug_log = debug_log

        self.task_runner = AsyncConcurrentProcessor(max_running_requests)

    def _colbert_encode(
        self: Self,
        queries: list[str]
    ) -> torch.Tensor:

        config = self._colbert_query_encoder_config

        with torch.no_grad():
            inputs = [". " + query for query in queries]
            tokens = self._colbert_tokenizer(
                inputs,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=True,
                return_tensors=TensorType.PYTORCH,
                max_length=self._colbert_query_encoder_config.max_length,
            )

            ids: torch.Tensor = tokens['input_ids']
            mask: torch.Tensor = tokens['attention_mask']

            ids[:, 1] = config.q_marker_token_id
            ids[ids == config.pad_token_id] = config.mask_token_id

            embeddings = self._colbert_query_encoder(ids.to(config.device), mask.to(config.device))

        return embeddings

    def _process_claim_question(
        self: Self,
        items: list[QueueRequestMessage[LMClaimQuestion]]
    ) -> None:

        if len(items) == 0:
            return

        embeddings = self._colbert_encode([item.message.question for item in items]).cpu()

        for i, item in enumerate(items):
            data_slice: np.ndarray = embeddings[i].numpy()
            colbert_result_msg = ColbertClaimQuestionTensor(
                data=data_slice.tobytes(),
                num_query_tokens=embeddings.shape[1],
                embedding_dim=embeddings.shape[2],
                claim_id=item.message.claim_id,
                question_id=item.message.question_id,
            )
            queue_item = QueueRequestMessage(
                user_request_id=item.user_request_id,
                message=colbert_result_msg
            )

            logger.trace("Generated queue message {}".format(queue_item))

            self._debug_log.update_stage_queue_out(
                (item.user_request_id, item.message.claim_id, item.message.question_id)
            )
            routing_key = (item.user_request_id, item.message.claim_id, item.message.question_id)
            self._queue_to_reranker.put(queue_item, block=False, key=routing_key)

    async def add_request(
        self: Self,
        item: QueueRequestMessage[LMClaimQuestion]
    ) -> None:

        await self.task_runner.add_task(asyncio.to_thread(self._process_claim_question, [item]))

    async def add_requests(
        self: Self,
        items: list[QueueRequestMessage[LMClaimQuestion]]
    ) -> None:

        await self.task_runner.add_task(asyncio.to_thread(self._process_claim_question, items))

    def warm_up(
        self: Self
    ) -> None:

        logger.debug("Warming up ColBERT query encoder model")

        warmup_query = "The quick brown fox jumped over the lazy dog"
        for _ in range(10):
            self._colbert_encode([warmup_query])

        logger.debug("Finished warming up ColBERT query encoder model")


async def main(
    comm_params_data: bytes,
    model_serving_data_path: str
) -> None:

    logging.disable(logging.INFO)
    logger.debug("Starting ColBERT encoder stage")

    stage_comm = StageCommunicatorFactory.get_proxy_from_bytes(comm_params_data)
    stage_comm.initialize()

    debug_log = InstanceComputeLatencyLogger(stage_comm.get_instance_name())

    queue_from_query_generation = stage_comm.get_input_queue_interface(
        input_stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        output_stage_name=FactoolQueueEndpoint.COLBERT_ENCODER,
        index=0,
        item_type=QueueRequestMessage[LMClaimQuestion],
        serializer=QueueRequestMessageSerializer(LMClaimQuestion)
    )
    queue_to_reranker = stage_comm.get_output_queue_interface(
        input_stage_name=FactoolQueueEndpoint.COLBERT_ENCODER,
        output_stage_name=FactoolQueueEndpoint.RERANKER,
        index=0,
        item_type=QueueRequestMessage[ColbertClaimQuestionTensor],
        serializer=QueueRequestMessageSerializer(ColbertClaimQuestionTensor)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    colbert_query_encoder_config = ColbertQueryEncoderConfig(
        max_length=32,
        q_marker_token_id=1,
        mask_token_id=103,
        pad_token_id=0,
        device=device
    )

    colbert_tokenizer_fname = os.path.join(model_serving_data_path, "msmarco.psg.kldR2.nway64.ib__colbert-400000/")
    colbert_tokenizer = AutoTokenizer.from_pretrained(colbert_tokenizer_fname)

    base_model = BaseColBERT(colbert_tokenizer_fname)
    colbert = ColBERT(base_model.bert, base_model.linear).eval()

    colbert = colbert.to(colbert_query_encoder_config.device)
    if colbert_query_encoder_config.device == 'cpu':
        colbert = colbert.float()

    colbert_compiled = torch.compile(colbert, mode="reduce-overhead")

    colbert_processor = ColbertProcessor(
        colbert_tokenizer=colbert_tokenizer,
        colbert_query_encoder=colbert_compiled,
        colbert_query_encoder_config=colbert_query_encoder_config,
        queue_to_reranker=queue_to_reranker,
        debug_log=debug_log,
        max_running_requests=None
    )

    colbert_processor.warm_up()

    logger.debug("ColBERT encoder stage ready")
    stage_comm.signal_instance_ready()

    while True:
        try:
            await colbert_processor.task_runner.wait_for_open_task_slot()
            items = await queue_from_query_generation.get_batch_async(
                min_num_items=1,
                max_num_items=colbert_processor.task_runner.get_num_open_task_slots(),
                block=True,
                timeout=5
            )

            for item in items:
                debug_log.update_stage_queue_in((item.user_request_id, item.message.claim_id, item.message.question_id))
            await colbert_processor.add_requests(items)

        except AsyncQueueEmptyException:
            pass

        if stage_comm.should_stop():
            break

        debug_log.update_log()
