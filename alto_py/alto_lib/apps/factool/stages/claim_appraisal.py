# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:15:36 2024

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
from alto_lib.apps.factool.common.messages import LMClaim, VerifiedClaim
from alto_lib.apps.factool.common.prompts import CLAIM_APPRAISAL_PROMPT, CLAIM_APPRAISAL_PATTERNS
from alto_lib.apps.factool.common.lm_utils import (
    get_sampling_params, VllmPromptRequest, AsyncGenerateCallbackInput, LLMGeneratorAsync
)
from alto_lib.apps.factool.common import FactoolQueueEndpoint


def get_default_verified_claim():
    return {
        "reasoning": "",
        "error": "",
        "correction": "",
        "factuality": False,
    }



async def send_verified_claim(
    claim, verification, queue_id, claim_id, output_queue_group, tag
):
    verified_claim = VerifiedClaim()
    verified_claim.claim_id = claim_id
    verified_claim.claim = claim
    verified_claim.reasoning = verification["reasoning"]
    verified_claim.correction = verification["correction"]
    verified_claim.error = verification["error"]
    verified_claim.factuality = verification["factuality"]

    output_queue_group.log_send_msg(
        "claim_appraisal_lm_to_driver", queue_id, claim_id=claim_id
    )

    await output_queue_group.write(
        "claim_appraisal_lm_to_driver", verified_claim, queue_id
    )
    await output_queue_group.record_write_complete(
        "claim_appraisal_lm_to_driver", queue_id, tag
    )


async def generate(
    engine,
    tokenizer,
    claim,
    prompt,
    sampling_params,
    queue_id,
    claim_id,
    tag,
    output_queue_group,
    logger,
):
    messages = [
        {
            "role": "system",
            "content": CLAIM_APPRAISAL_PROMPT,
        },
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    prev_newline_idx = 0
    num_claims = 0
    buffer = None

    logger.debug(f"Adding prompt to engine:\n{prompt}")

    stream = await engine.add_request(prompt, sampling_params)

    start = time.time()
    async for output in stream:
        pass
    end = time.time()
    generation_time = end - start

    completion = output
    verification = get_default_verified_claim()
    for key, pattern in CLAIM_APPRAISAL_PATTERNS.items():
        match = re.search(pattern, completion)
        if match is not None:
            val = match.group(1)
            if key == "factuality":
                verification[key] = True if "true" in val.lower() else False
                if not isinstance(verification[key], bool):
                    raise ValueError(f"verification['factuality'] is not a bool!")
            else:
                verification[key] = val

    logger.debug(
        f"Took {generation_time} seconds to run claim appraisal:\n{json.dumps(verification, indent=4)}"
    )

    await send_verified_claim(
        claim, verification, queue_id, claim_id, output_queue_group, tag
    )


async def main(args):
    set_logging_config()
    logger = logging.getLogger(f"CLAIM APPRAISAL # {args.replica_id}")
    logger.setLevel(get_logging_level(args))

    engine_args = LmEngineFactory.get_engine_args_from_cli_args(args.lm_engine, args)
    engine = LmEngineFactory.create_engine_from_engine_args(
        args.lm_engine,
        engine_args,
        port=args.base_lm_engine_port,
        additional_ports=[args.base_lm_engine_port + i for i in range(1, 6)],
    )
    tokenizer = engine.get_tokenizer()
    set_logging_level(args)
    sampling_params = LmEngineFactory.create_sampling_params(
        args.lm_engine, temperature=0.5, max_tokens=512
    )

    input_queues, output_queues, replica_id = await setup_async_queues(args)

    loop = asyncio.get_running_loop()
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        loop.add_signal_handler(sig, signal_handler, input_queues, output_queues)

    for iq in input_queues:
        if iq.queue_name == "claim_extraction_lm_to_claim_appraisal_lm":
            iq.proto_class = LMClaim
        elif iq.queue_name == "question_generation_lm_to_claim_appraisal_lm":
            iq.proto_class = LMClaimQuestionCount
        elif iq.queue_name == "reranker_to_claim_appraisal_lm":
            iq.proto_class = RerankerClaimQuestionOutput
        else:
            logger.warn("Unknown iq name: {}".format(iq.queue_name))
            exit(1)
    output_queue_group = AsyncOutputQueueGroup(output_queues, logger)
    input_queue_group = AsyncInputQueueGroup(input_queues, output_queue_group, logger)

    input_queue_group.register_close(
        "claim_extraction_lm_to_claim_appraisal_lm", "claim_appraisal_lm_to_driver"
    )
    input_queue_group.register_close(
        "question_generation_lm_to_claim_appraisal_lm", "claim_appraisal_lm_to_driver"
    )
    input_queue_group.register_close(
        "reranker_to_claim_appraisal_lm", "claim_appraisal_lm_to_driver"
    )

    logger.info("Loading collection...")
    collection = {}
    with open(os.path.join(os.environ["DATA_PATH"], "wiki150.tsv"), "r") as f:
        next(f)
        for line in f:
            (doc_id, doc_text, _) = line.strip().split("\t")
            collection[int(doc_id)] = doc_text
    logger.info(f"Finished loading collection")

    # register dummy protos on two output queues
    logger.info("Finished initializing")
    await output_queue_group.write(
        "claim_appraisal_lm_to_driver",
        VerifiedClaim(),
        0,
    )

    claims = defaultdict(dict)

    generate_tasks = set()
    send_tasks = set()
    async for objects_dict in input_queue_group:
        for queue_name, queue_batch in objects_dict.items():
            for queue_object in queue_batch:
                queue_id = queue_object.identifier
                claim_id = queue_object.data.claim_id
                claim_key = (queue_id, claim_id)

                if queue_name == "claim_extraction_lm_to_claim_appraisal_lm":
                    input_queue_group.log_recv_msg(
                        "claim_extraction_lm_to_claim_appraisal_lm",
                        queue_id,
                        claim_id=claim_id,
                    )
                    claims[claim_key]["claim"] = queue_object.data.claim
                elif queue_name == "question_generation_lm_to_claim_appraisal_lm":
                    num_questions = queue_object.data.num_questions
                    input_queue_group.log_recv_msg(
                        "question_generation_lm_to_claim_appraisal_lm",
                        queue_id,
                        claim_id=claim_id,
                        num_questions=num_questions,
                    )
                    claims[claim_key]["num_questions"] = num_questions

                elif queue_name == "reranker_to_claim_appraisal_lm":
                    question_id = queue_object.data.question_id
                    input_queue_group.log_recv_msg(
                        "reranker_to_claim_appraisal_lm",
                        queue_id,
                        claim_id=claim_id,
                        question_id=question_id,
                    )
                    if "evidence" not in claims[claim_key]:
                        claims[claim_key]["evidence"] = {}
                    claims[claim_key]["evidence"][question_id] = queue_object.data.doc
                else:
                    logger.warn("Unknown queue name")
                    exit(1)

                if (
                    "claim" in claims[claim_key]
                    and "evidence" in claims[claim_key]
                    and "num_questions" in claims[claim_key]
                    and len(claims[claim_key]["evidence"])
                    == claims[claim_key]["num_questions"]
                ):
                    tag = output_queue_group.generate_tag(
                        "claim_appraisal_lm_to_driver"
                    )
                    await output_queue_group.record_write_start(
                        "claim_appraisal_lm_to_driver", queue_id, tag
                    )

                    claim = claims[claim_key]["claim"]
                    evidence = defaultdict(float)
                    for question_id in claims[claim_key]["evidence"]:
                        for doc in claims[claim_key]["evidence"][question_id]:
                            evidence[doc.doc_id] += 1.0 / doc.rank

                    k = 3
                    docs = heapq.nlargest(k, list(evidence.items()), key=lambda x: x[1])
                    docs_text = [collection[doc[0]] for doc in docs]

                    if len(docs_text) > 0:
                        prompt = ["The following is the given text [text]:"]
                        prompt.append(claim)
                        prompt.append(
                            "The following is the provided evidences [evidences]:"
                        )
                        for evidence in docs_text:
                            prompt.append(evidence)
                        prompt = "\n".join(prompt)

                        logger.debug(
                            f"Adding generate task for queue object {queue_id}, claim id {claim_id}"
                        )
                        task = asyncio.create_task(
                            generate(
                                engine,
                                tokenizer,
                                claim,
                                prompt,
                                sampling_params,
                                queue_id,
                                claim_id,
                                tag,
                                output_queue_group,
                                logger,
                            )
                        )

                        generate_tasks.add(task)
                        task.add_done_callback(generate_tasks.discard)

                        logger.debug(
                            f"{len(generate_tasks)} generate tasks in the queue"
                        )
                    else:
                        verification = get_default_verified_claim()
                        task = asyncio.create_task(
                            send_verified_claim(
                                claim,
                                verification,
                                queue_id,
                                claim_id,
                                output_queue_group,
                                tag,
                            )
                        )

                        send_tasks.add(task)
                        task.add_done_callback(send_tasks.discard)

    tasks = generate_tasks.union(send_tasks)
    if len(tasks) > 0:
        done, pending = await asyncio.wait(tasks)
        if len(pending) > 0:
            raise ValueError(
                f"Still have the following waiting tasks in claim appraisal: {pending}"
            )

    logger.info(f"Cleaning up")
    await cleanup(input_queues, output_queues)
    del engine
    logger.info(f"Finished cleaning up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claim appraisal")
    parser = add_queue_args(parser)
    args, _ = parser.parse_known_args()
    parser = LmEngineFactory.add_cli_args_to_parser(args.lm_engine, parser)
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
