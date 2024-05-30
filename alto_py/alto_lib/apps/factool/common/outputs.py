# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 03:53:26 2023

@author: Shahir
"""

from dataclasses import dataclass, field
from typing import Self


@dataclass(kw_only=True)
class Passage:
    passage_id: int  # unique identifier for the passage
    passage_text: str  # the content of the passage

    def __repr__(
        self: Self
    ) -> str:

        return f"|============> Passage {self.passage_id}: {self.passage_text}"


@dataclass(kw_only=True)
class SearchQuery:
    query_text: str  # the actual search query text
    relevant_passages: list[Passage] = field(default_factory=list)  # list of passages relevant to the query

    def add_passage(
        self: Self,
        passage: Passage
    ) -> None:

        self.relevant_passages.append(passage)

    def __repr__(
        self: Self
    ) -> str:

        passages = "\n".join(str(p) for p in self.relevant_passages)

        return f"|========> SearchQuery \"{self.query_text}\" supported by the following passages:\n{passages}"


@dataclass(kw_only=True)
class Claim:
    claim_text: str  # the content of the claim
    search_queries: list[SearchQuery] = field(default_factory=list)  # list of search queries used to verify the claim

    def add_search_query(
        self: Self,
        search_query: SearchQuery
    ) -> None:

        self.search_queries.append(search_query)

    def __repr__(
        self: Self
    ) -> str:

        queries = "\n".join(str(q) for q in self.search_queries)

        return f"|====> Claim \"{self.claim_text}\" verified with the following queries:\n{queries}"


@dataclass(kw_only=True)
class QuestionAnswer:
    user_request_id: int
    question_text: str
    answer_text: str
    claims: list[Claim] = field(default_factory=list)  # list of claims made in the answer

    def add_claim(
        self: Self,
        claim: Claim
    ) -> None:

        self.claims.append(claim)

    def __repr__(
        self: Self
    ) -> str:

        claims = "\n".join(str(c) for c in self.claims)

        return f"[ID: {self.user_request_id}]\nQ: {self.question_text}\nA: {self.answer_text}\nClaims:\n{claims}"
