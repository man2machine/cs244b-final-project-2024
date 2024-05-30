# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 19:52:28 2023

@author: Shahir
"""

from typing import Self, final
from dataclasses import dataclass

import msgpack  # type: ignore

from alto_lib.core.manager import QueueMessage
from alto_lib.utils import DEFAULT_ENCODING

# TODO: consider using msgspec or ujson instead of msgpack

HIDE_TEXT = False


@final
@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class FactoolRequest(QueueMessage):
    query: str

    def to_bytes(
        self: Self
    ) -> bytes:

        return self.query.encode(DEFAULT_ENCODING)

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        return cls(query=buf.decode(DEFAULT_ENCODING))

    def __str__(
        self: Self
    ) -> str:

        if HIDE_TEXT:
            return "FactoolRequest(query=<len {}>)".format(len(self.query))
        else:
            return "FactoolRequest(query={})".format(repr(self.query))


@final
@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class LMGeneration(QueueMessage):
    generation: str

    def to_bytes(
        self: Self
    ) -> bytes:

        return self.generation.encode(DEFAULT_ENCODING)

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        return cls(generation=buf.decode(DEFAULT_ENCODING))

    def __str__(
        self: Self
    ) -> str:

        if HIDE_TEXT:
            return "LMGeneration(generation=<len {}>)".format(len(self.generation))
        else:
            return "LMGeneration(generation={})".format(repr(self.generation))


@final
@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class LMClaim(QueueMessage):
    claim: str
    claim_id: int
    last_claim: bool

    def to_bytes(
        self: Self
    ) -> bytes:

        return msgpack.packb((self.claim_id, self.last_claim, self.claim))

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        claim_id, last_claim, claim = msgpack.unpackb(buf)
        out = cls(
            claim=claim,
            claim_id=claim_id,
            last_claim=last_claim
        )

        return out

    def __str__(
        self: Self
    ) -> str:

        if HIDE_TEXT:
            return "LMClaim(claim=<len {}>, claim_id={}, last_claim={})".format(
                len(self.claim),
                self.claim_id,
                self.last_claim
            )
        else:
            return "LMClaim(claim={}, claim_id={}, last_claim={})".format(
                repr(self.claim),
                self.claim_id,
                self.last_claim
            )


@final
@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class LMClaimQuestion(QueueMessage):
    question: str
    claim_id: int
    question_id: int
    last_question: bool

    def to_bytes(
        self: Self
    ) -> bytes:

        return msgpack.packb((self.claim_id, self.question_id, self.last_question, self.question))

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        claim_id, question_id, last_question, question = msgpack.unpackb(buf)
        out = cls(
            question=question,
            claim_id=claim_id,
            question_id=question_id,
            last_question=last_question
        )

        return out

    def __str__(
        self: Self
    ) -> str:

        if HIDE_TEXT:
            return "LMClaimQuestion(question=<len {}>, claim_id={}, question_id={}, last_question={})".format(
                len(self.question),
                self.claim_id,
                self.question_id,
                self.last_question
            )
        else:
            return "LMClaimQuestion(question={}, claim_id={}, question_id={}, last_question={})".format(
                repr(self.question),
                self.claim_id,
                self.question_id,
                self.last_question
            )


@final
@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class LMClaimQuestionToken(QueueMessage):
    word: str
    claim_id: int
    question_id: int
    last_token: bool

    def to_bytes(
        self: Self
    ) -> bytes:

        return msgpack.packb((self.claim_id, self.question_id, self.last_token, self.word))

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        claim_id, question_id, last_token, word = msgpack.unpackb(buf)
        out = cls(
            word=word,
            claim_id=claim_id,
            question_id=question_id,
            last_token=last_token
        )

        return out

    def __str__(
        self: Self
    ) -> str:

        return "LMClaimQuestionToken(word={}, claim_id={}, question_id={}, last_token={})".format(
            self.word,
            self.claim_id,
            self.question_id,
            self.last_token
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class DocumentId(QueueMessage):
    rank: int
    doc_id: int
    score: float

    def to_bytes(
        self: Self
    ) -> bytes:

        return msgpack.packb((self.rank, self.doc_id, self.score))

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        rank, doc_id, score = msgpack.unpackb(buf)
        out = cls(
            rank=rank,
            doc_id=doc_id,
            score=score
        )

        return out

    def __str__(
        self: Self
    ) -> str:

        return "DocumentId(rank={}, doc_id={}, score={:.3f})".format(
            self.rank,
            self.doc_id,
            self.score
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class BM25ClaimQuestionResult(QueueMessage):
    doc: list[DocumentId]
    claim_id: int
    question_id: int

    def to_bytes(
        self: Self
    ) -> bytes:

        return msgpack.packb((self.claim_id, self.question_id, [n.to_bytes() for n in self.doc]))

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        claim_id, question_id, doc_bytes = msgpack.unpackb(buf)
        doc = [DocumentId.from_bytes(n) for n in doc_bytes]

        out = cls(
            doc=doc,
            claim_id=claim_id,
            question_id=question_id
        )

        return out

    def __str__(
        self: Self
    ) -> str:

        return "BM25ClaimQuestionResult(doc=<len {}>, claim_id={}, question_id={})".format(
            len(self.doc),
            self.claim_id,
            self.question_id
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ColbertClaimQuestionTensor(QueueMessage):
    data: bytes
    num_query_tokens: int
    embedding_dim: int
    claim_id: int
    question_id: int

    def to_bytes(
        self: Self
    ) -> bytes:

        return msgpack.packb((self.num_query_tokens, self.embedding_dim, self.claim_id, self.question_id, self.data))

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        num_query_tokens, embedding_dim, claim_id, question_id, data = msgpack.unpackb(buf)
        out = cls(
            data=data,
            num_query_tokens=num_query_tokens,
            embedding_dim=embedding_dim,
            claim_id=claim_id,
            question_id=question_id
        )

        return out

    def __str__(
        self: Self
    ) -> str:

        return (
            "ColbertClaimQuestionTensor("
            "data=<len {}>, num_query_tokens={}, embedding_dim={}, claim_id={}, question_id={})"
        ).format(
            len(self.data),
            self.num_query_tokens,
            self.embedding_dim,
            self.claim_id,
            self.question_id
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class RerankerClaimQuestionOutput(QueueMessage):
    doc: list[DocumentId]
    claim_id: int
    question_id: int

    def to_bytes(
        self: Self
    ) -> bytes:

        return msgpack.packb((self.claim_id, self.question_id, [n.to_bytes() for n in self.doc]))

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        claim_id, question_id, doc_bytes = msgpack.unpackb(buf)
        doc = [DocumentId.from_bytes(n) for n in doc_bytes]

        out = cls(
            doc=doc,
            claim_id=claim_id,
            question_id=question_id
        )

        return out

    def __str__(
        self: Self
    ) -> str:

        return "RerankerClaimQuestionOutput(doc=<len {}>, claim_id={}, question_id={})".format(
            len(self.doc),
            self.claim_id,
            self.question_id
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class QuestionResult(QueueMessage):
    question: str
    doc: list[DocumentId]

    def to_bytes(
        self: Self
    ) -> bytes:

        return msgpack.packb((self.question, [n.to_bytes() for n in self.doc]))

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        question, doc_bytes = msgpack.unpackb(buf)
        doc = [DocumentId.from_bytes(n) for n in doc_bytes]

        out = cls(
            question=question,
            doc=doc
        )

        return out

    def __str__(
        self: Self
    ) -> str:

        if HIDE_TEXT:
            return "QuestionResult(question=<len {}>, doc=<len {}>)".format(
                len(self.question),
                len(self.doc)
            )
        else:
            return "QuestionResult(question={}, doc={})".format(
                repr(self.question),
                self.doc
            )


@final
@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class VerifiedClaim(QueueMessage):
    claim: str
    questions: list[QuestionResult]

    def to_bytes(
        self: Self
    ) -> bytes:

        return msgpack.packb((self.claim, [n.to_bytes() for n in self.questions]))

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        claim, question_bytes = msgpack.unpackb(buf)
        questions = [QuestionResult.from_bytes(n) for n in question_bytes]

        out = cls(
            claim=claim,
            questions=questions
        )

        return out

    def __str__(
        self: Self
    ) -> str:

        if HIDE_TEXT:
            return "VerifiedClaim(claim=<len {}>, questions=<len {}>)".format(
                len(self.claim),
                len(self.questions)
            )
        else:
            return "VerifiedClaim(claim={}, questions={})".format(
                repr(self.claim),
                self.questions
            )


@final
@dataclass(frozen=True, kw_only=True, repr=False)
class FactoolResponse(QueueMessage):
    generation: str
    claims: list[VerifiedClaim]

    def to_bytes(
        self: Self
    ) -> bytes:

        return msgpack.packb((self.generation, [n.to_bytes() for n in self.claims]))

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        generation, claim_bytes = msgpack.unpackb(buf)
        claims = [VerifiedClaim.from_bytes(n) for n in claim_bytes]

        out = cls(
            generation=generation,
            claims=claims
        )

        return out

    def __str__(
        self: Self
    ) -> str:

        if HIDE_TEXT:
            return "FactoolResponse(generation=<len {}>, claims=<len {}>)".format(
                len(self.generation),
                len(self.claims)
            )
        else:
            return "FactoolResponse(generation={}, claims={})".format(
                repr(self.generation),
                self.claims
            )
