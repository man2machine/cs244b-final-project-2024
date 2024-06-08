# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:35:24 2023

@author: Shahir
"""

import os
import collections
from typing import Self, final

from alto_lib.core.app_endpoints import AppRequest, PossionSourceClientNode, IPv4Address
from alto_lib.apps.factool.common.messages import FactoolRequest


@final
class FactoolClientNode(PossionSourceClientNode):
    _requests: collections.deque[FactoolRequest]
    _next_user_request_id: int

    def __init__(
        self: Self,
        *,
        model_serving_data_path: str,
        control_address: IPv4Address,
        request_send_address: IPv4Address,
        response_recv_address: IPv4Address,
        name: str,
        log_fname: str,
        rate: float = 0.25,
        log_interval: float = 1,
        send_length: float = 720
    ) -> None:

        queries: list[str] = []
        with open(os.path.join(model_serving_data_path, "squad.train.tsv"), "r") as f:
            for line in f:
                _, query = line.strip().split("\t")
                queries.append(query)

        requests: list[FactoolRequest] = []
        for query in queries:
            request = FactoolRequest(query=query)
            requests.append(request)

        self._requests = collections.deque(requests)
        self._next_user_request_id = 0

        super().__init__(
            control_address=control_address,
            request_send_address=request_send_address,
            response_recv_address=response_recv_address,
            name=name,
            log_fname=log_fname,
            rate=rate,
            log_interval=log_interval,
            send_length=send_length
        )

    def _get_next_request(
        self: Self
    ) -> AppRequest:

        if not len(self._requests):
            raise ValueError()

        out = AppRequest(
            user_request_id=self._next_user_request_id,
            message=self._requests.popleft().to_bytes()
        )

        self._next_user_request_id += 1
        if len(self._requests) == 0:
            self.set_requests_finished()

        return out
