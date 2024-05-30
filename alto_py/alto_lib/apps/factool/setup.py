# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:59:12 2023

@author: Shahir
"""

from vllm import AsyncEngineArgs  # type: ignore

from alto_lib.core.manager import StageManager, QueueRequestMessage
from alto_lib.core.app_endpoints import IPv4Address
from alto_lib.apps.factool.common import FactoolQueueEndpoint
from alto_lib.apps.factool.common.messages import (
    FactoolRequest, LMGeneration, LMClaim, LMClaimQuestion, LMClaimQuestionToken, BM25ClaimQuestionResult,
    ColbertClaimQuestionTensor, RerankerClaimQuestionOutput
)
from alto_lib.apps.factool.stages import (
    driver, question_answering, claim_extraction, query_generation, bm25, colbert_encoder, reranker
)


def setup_factool_app(
    *,
    manager: StageManager,
    model_serving_data_path: str,
    control_address: IPv4Address,
    request_recv_address: IPv4Address,
    response_send_address: IPv4Address
) -> None:

    # pipeline stages
    manager.add_stages(list(FactoolQueueEndpoint))

    # driver queues
    manager.add_queue(
        input_stage_name=FactoolQueueEndpoint.DRIVER,
        output_stage_name=FactoolQueueEndpoint.QUESTION_ANSWERING,
        item_type=QueueRequestMessage[FactoolRequest]
    )

    # question answering queues
    manager.add_queue(
        input_stage_name=FactoolQueueEndpoint.QUESTION_ANSWERING,
        output_stage_name=FactoolQueueEndpoint.CLAIM_EXTRACTION,
        item_type=QueueRequestMessage[LMGeneration]
    )
    manager.add_queue(
        input_stage_name=FactoolQueueEndpoint.QUESTION_ANSWERING,
        output_stage_name=FactoolQueueEndpoint.DRIVER,
        item_type=QueueRequestMessage[LMGeneration]
    )

    # claim extraction queues
    manager.add_queue(
        input_stage_name=FactoolQueueEndpoint.CLAIM_EXTRACTION,
        output_stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        item_type=QueueRequestMessage[LMClaim]
    )
    manager.add_queue(
        input_stage_name=FactoolQueueEndpoint.CLAIM_EXTRACTION,
        output_stage_name=FactoolQueueEndpoint.DRIVER,
        item_type=QueueRequestMessage[LMClaim]
    )

    # query generation queues
    manager.add_queue(
        input_stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        output_stage_name=FactoolQueueEndpoint.BM25,
        item_type=QueueRequestMessage[LMClaimQuestionToken]
    )
    manager.add_queue(
        input_stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        output_stage_name=FactoolQueueEndpoint.COLBERT_ENCODER,
        item_type=QueueRequestMessage[LMClaimQuestion]
    )
    manager.add_queue(
        input_stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        output_stage_name=FactoolQueueEndpoint.DRIVER,
        item_type=QueueRequestMessage[LMClaimQuestion]
    )

    # bm25 queues
    manager.add_queue(
        input_stage_name=FactoolQueueEndpoint.BM25,
        output_stage_name=FactoolQueueEndpoint.RERANKER,
        item_type=QueueRequestMessage[BM25ClaimQuestionResult]
    )

    # colbert encoder queues
    manager.add_queue(
        input_stage_name=FactoolQueueEndpoint.COLBERT_ENCODER,
        output_stage_name=FactoolQueueEndpoint.RERANKER,
        item_type=QueueRequestMessage[ColbertClaimQuestionTensor]
    )

    # reranker queues
    manager.add_queue(
        input_stage_name=FactoolQueueEndpoint.RERANKER,
        output_stage_name=FactoolQueueEndpoint.DRIVER,
        item_type=QueueRequestMessage[RerankerClaimQuestionOutput]
    )

    # stages
    engine_args = AsyncEngineArgs(
        model="open-orca/mistral-7b-openorca",
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        worker_use_ray=False
    )

    comm_params_data = manager.get_comm_params_data()

    manager.add_stage_args(
        stage_name=FactoolQueueEndpoint.DRIVER,
        run_func=driver.main,
        comm_params_data=comm_params_data,
        control_address=control_address,
        request_recv_address=request_recv_address,
        response_send_address=response_send_address
    )
    manager.add_stage_args(
        stage_name=FactoolQueueEndpoint.QUESTION_ANSWERING,
        run_func=question_answering.main,
        comm_params_data=comm_params_data,
        engine_args=engine_args
    )
    manager.add_stage_args(
        stage_name=FactoolQueueEndpoint.CLAIM_EXTRACTION,
        run_func=claim_extraction.main,
        comm_params_data=comm_params_data,
        engine_args=engine_args
    )
    manager.add_stage_args(
        stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        run_func=query_generation.main,
        comm_params_data=comm_params_data,
        engine_args=engine_args
    )
    manager.add_stage_args(
        stage_name=FactoolQueueEndpoint.BM25,
        run_func=bm25.main,
        comm_params_data=comm_params_data,
        model_serving_data_path=model_serving_data_path
    )
    manager.add_stage_args(
        stage_name=FactoolQueueEndpoint.COLBERT_ENCODER,
        run_func=colbert_encoder.main,
        comm_params_data=comm_params_data,
        model_serving_data_path=model_serving_data_path
    )
    manager.add_stage_args(
        stage_name=FactoolQueueEndpoint.RERANKER,
        run_func=reranker.main,
        comm_params_data=comm_params_data,
        model_serving_data_path=model_serving_data_path
    )

    manager.add_instances(
        stage_name=FactoolQueueEndpoint.DRIVER,
        num_cpus=1,
        num_gpus=0,
        num_llm_gpus=0
    )
    manager.add_instances(
        stage_name=FactoolQueueEndpoint.QUESTION_ANSWERING,
        num_cpus=1,
        num_gpus=0,
        num_llm_gpus=1,
        num_instances=2
    )
    manager.add_instances(
        stage_name=FactoolQueueEndpoint.CLAIM_EXTRACTION,
        num_cpus=1,
        num_gpus=0,
        num_llm_gpus=1,
        num_instances=2
    )
    manager.add_instances(
        stage_name=FactoolQueueEndpoint.QUERY_GENERATION,
        num_cpus=1,
        num_gpus=0,
        num_llm_gpus=1,
        num_instances=3
    )
    manager.add_instances(
        stage_name=FactoolQueueEndpoint.BM25,
        num_cpus=1,
        num_gpus=0,
        num_llm_gpus=0,
        num_instances=4
    )
    manager.add_instances(
        stage_name=FactoolQueueEndpoint.COLBERT_ENCODER,
        num_cpus=1,
        num_gpus=1,
        num_llm_gpus=0,
        num_instances=1
    )
    manager.add_instances(
        stage_name=FactoolQueueEndpoint.RERANKER,
        num_cpus=1,
        num_gpus=0,
        num_llm_gpus=0,
        num_instances=4
    )
