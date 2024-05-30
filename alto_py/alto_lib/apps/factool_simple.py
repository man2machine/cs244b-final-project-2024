# -*- coding: utf-8 -*-
"""
Created on Thu May  2 08:14:12 2024

@author: Shahir
"""

import asyncio
from typing import Self
from collections.abc import AsyncIterator

from scipy import sparse  # type: ignore

from bm25_index import Bm25Index

import alto_lib.core.interface as alto
from alto_lib.apps.factool.common.prompts import (
    QUESTION_ANSWERING_PROMPT, CLAIM_EXTRACTION_PROMPT, QUERY_GENERATION_PROMPT, CLAIM_APPRAISAL_PROMPT
)


class FactoolRequest(alto.Message):
    question: str


class ScoredDocument(alto.Message):
    rank: int
    doc_id: int
    score: float


class BM25ClaimQueryOutput(alto.Message):
    doc: list[ScoredDocument]


class ColbertClaimQueryTensor(alto.Message):
    data: bytes
    num_query_tokens: int
    embedding_dim: int


class RerankerClaimQueryOutput(alto.Message):
    doc: list[ScoredDocument]


class VerifiedClaim(alto.Message):
    claim: str
    reasoning: str
    correction: str
    error: str
    factuality: bool


class FactoolResponse(alto.Message):
    generation: str
    verified_claims: list[VerifiedClaim]


class FactoolClient(alto.Stage):
    @alto.generator
    async def get_requests(
        self: Self
    ) -> AsyncIterator[FactoolRequest]:

        yield NotImplemented


class QuestionAnswering(alto.Stage):
    @alto.processor
    async def get_answer(
        self: Self,
        request: FactoolRequest
    ) -> alto.LMTextPipe[alto.FullText]:

        out = self.lm_generate(
            system_prompt=QUESTION_ANSWERING_PROMPT,
            user_prompt=request.question
        )

        return out


class ClaimExtraction(alto.Stage):
    @alto.processor
    async def get_claims(
        self: Self,
        qa_output: alto.LMTextPipe[alto.FullText]
    ) -> AsyncIterator[alto.LMTextPipe[alto.Line]]:

        answer = await qa_output.to_str()
        out = self.lm_generate(
            system_prompt=CLAIM_EXTRACTION_PROMPT,
            user_prompt=answer
        ).split(alto.Line)

        async for claim in out:
            yield claim


class QueryGeneration(alto.Stage):
    @alto.processor
    async def get_queries(
        self: Self,
        ce_output: alto.LMTextPipe[alto.Line]
    ) -> AsyncIterator[alto.LMTextPipe[alto.Line]]:

        claim = await ce_output.to_str()
        out = self.lm_generate(
            system_prompt=QUERY_GENERATION_PROMPT,
            user_prompt=claim
        ).split(alto.Line)

        async for query in out:
            yield query


class BM25(alto.Stage):
    _bm25_index: Bm25Index

    class _QueryWorker:
        _bm25_index: Bm25Index
        _lock: asyncio.Lock
        _stage: alto.Stage
        _query_scores: sparse.csr_array | None

        def __init__(
            self: Self,
            bm25_index: Bm25Index
        ) -> None:

            self._bm25_index = bm25_index
            self._query_scores = None

        def _get_token_scores(
            self: Self,
            word: str
        ) -> sparse.csr_array:

            return NotImplemented

        def _get_topk(
            self: Self,
            scores: sparse.csr_array
        ) -> BM25ClaimQueryOutput:

            return NotImplemented

        async def _process_token(
            self: Self,
            word: alto.LMTextPipe[alto.Word],
        ) -> None:

            token_scores = self._get_token_scores(await word.to_str())
            await self._lock.acquire()
            if self._query_scores is None:
                self._query_scores = token_scores
            else:
                self._query_scores += token_scores
            self._lock.release()

        async def process_query(
            self: Self,
            tokens: alto.Stream[alto.LMTextPipe[alto.Word]],
        ) -> BM25ClaimQueryOutput:

            async for _ in self._stage.spawn_thread_iter(tokens, self._process_token, preserve_output_order=False):
                pass
            assert self._query_scores is not None
            docs = self._get_topk(self._query_scores)

            return docs

    @alto.processor
    async def get_doc_set(
        self: Self,
        qg_output: alto.Stream[alto.LMTextPipe[alto.Word]],
    ) -> BM25ClaimQueryOutput:

        worker = self._QueryWorker(self._bm25_index)
        docs = await worker.process_query(qg_output)

        return docs


class ColbertEncoder(alto.Stage):
    def _get_encodings(
        self: Self,
        qg_outpus: list[alto.LMTextPipe[alto.Line]]
    ) -> list[ColbertClaimQueryTensor]:

        return NotImplemented

    @alto.processor(batched_input=True)
    async def encode_query(
        self: Self,
        qg_outputs: list[alto.LMTextPipe[alto.Line]]
    ) -> list[ColbertClaimQueryTensor]:

        encodings = self._get_encodings(qg_outputs)

        return encodings


class Reranker(alto.Stage):
    def _rerank(
        self: Self,
        docs: BM25ClaimQueryOutput,
        embedding: ColbertClaimQueryTensor
    ) -> RerankerClaimQueryOutput:

        return NotImplemented

    @alto.processor
    async def rerank_doc_set_and_emb(
        self: Self,
        docs_and_encoding: alto.Joined[BM25ClaimQueryOutput, ColbertClaimQueryTensor],
    ) -> RerankerClaimQueryOutput:

        doc, embedding = await docs_and_encoding[0], await docs_and_encoding[1]
        out = self._rerank(doc, embedding)

        return out


class ClaimAppraisal(alto.Stage):
    def _combine_docs(
        self: Self,
        docs_per_query: list[RerankerClaimQueryOutput]
    ) -> set[ScoredDocument]:

        return NotImplemented

    def _get_user_prompt(
        self: Self,
        claim: alto.LMTextPipe,
        docs: set[ScoredDocument]
    ) -> str:

        return NotImplemented

    def _get_appraisal(
        self: Self,
        lm_out: alto.LMTextPipe[alto.FullText],
    ) -> VerifiedClaim:

        return NotImplemented

    @alto.processor
    async def appraise_claim_and_doc_set(
        self: Self,
        claims_and_docs: alto.Joined[alto.LMTextPipe, alto.Stream[RerankerClaimQueryOutput]],
    ) -> VerifiedClaim:

        claim, docs_per_query = await claims_and_docs[0], await claims_and_docs[1]

        docs = self._combine_docs(await docs_per_query.to_list())
        user_prompt = self._get_user_prompt(claim, docs)
        lm_out = self.lm_generate(
            system_prompt=CLAIM_APPRAISAL_PROMPT,
            user_prompt=user_prompt
        )
        appraisal_out = self._get_appraisal(lm_out)

        return appraisal_out


def build_factool(
) -> alto.AppPipeline[alto.Stream[alto.Joined[FactoolRequest, alto.Stream[VerifiedClaim]]]]:

    client_stage = FactoolClient.create_stage()
    qa_stage = QuestionAnswering.create_stage()
    ce_stage = ClaimExtraction.create_stage()
    qg_stage = QueryGeneration.create_stage()
    bm25_stage = BM25.create_stage()
    colbert_stage = ColbertEncoder.create_stage()
    reranker_stage = Reranker.create_stage()
    appraisal_stage = ClaimAppraisal.create_stage()

    def process_query(
        query: alto.StageOutput[alto.LMTextPipe[alto.Line]]
    ) -> alto.StageOutput[RerankerClaimQueryOutput]:

        query_tokens = alto.split_text(query, alto.Word)
        doc = bm25_stage.get_doc_set(query_tokens)
        embeddings = colbert_stage.encode_query(query)
        doc_and_embedding = alto.join(doc, embeddings)
        reranked_docs = reranker_stage.rerank_doc_set_and_emb(doc_and_embedding)

        return reranked_docs

    def process_claim(
        claim: alto.StageOutput[alto.LMTextPipe[alto.Line]]
    ) -> alto.StageOutput[VerifiedClaim]:

        queries = qg_stage.get_queries(claim)
        reranked_docs = alto.pmap(process_query, queries)
        claim_and_docs = alto.join(claim, reranked_docs)
        verified_claim = appraisal_stage.appraise_claim_and_doc_set(claim_and_docs)

        return verified_claim

    def process_request(
        request: alto.StageOutput[FactoolRequest]
    ) -> alto.StageOutput[alto.Joined[FactoolRequest, alto.Stream[VerifiedClaim]]]:

        answer = qa_stage.get_answer(request)
        claims = ce_stage.get_claims(answer)
        response = alto.pmap(process_claim, claims)
        output = alto.join(request, response)

        return output

    requests = client_stage.get_requests()
    output = alto.pmap(process_request, requests)
    factool = output.build_pipeline()

    return factool


async def run_factool(
    factool: alto.AppPipeline[alto.Stream[alto.Joined[FactoolRequest, alto.Stream[VerifiedClaim]]]]
) -> None:

    factool.start()
    async for pair in factool.output:
        request = await pair[0]
        response = await pair[1]
        print("=" * 40)
        print(f"Request: {request.question}")
        verified_claims = await response.to_list()
        for claim in verified_claims:
            print(f"Claim: {claim.claim}")
            print(f"Reasoning: {claim.reasoning}")
            print(f"Correction: {claim.correction}")
            print(f"Error: {claim.error}")
            print(f"Factuality: {claim.factuality}")
            print("\n")
        print("\n")


if __name__ == '__main__':
    factool = build_factool()
    asyncio.run(run_factool(factool))
