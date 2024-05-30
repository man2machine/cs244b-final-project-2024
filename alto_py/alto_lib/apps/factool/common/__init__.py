# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:02:33 2023

@author: Shahir
"""

from enum import ReprEnum
from typing import final


@final
class FactoolQueueEndpoint(str, ReprEnum):
    DRIVER = 'driver'
    QUESTION_ANSWERING = 'question_answering'
    CLAIM_EXTRACTION = 'claim_extraction'
    QUERY_GENERATION = 'query_generation'
    COLBERT_ENCODER = 'colbert_encoder'
    BM25 = 'bm25'
    RERANKER = 'reranker'
