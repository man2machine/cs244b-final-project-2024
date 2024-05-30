# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:09:59 2024

@author: Shahir
"""

import abc
from enum import ReprEnum
from collections.abc import Hashable
from typing import Self

from alto_lib.core.manager._proto import PipelinePlacementSpec, PipelineQueuesSpec, QueueSpec


class RoutingType(int, ReprEnum):
    FIXED_HASHING = 1
    SIMPLE_ESTIMATION = 2
    ADVANCED_ESTIMATION = 2


class StageRouter(metaclass=abc.ABCMeta):
    _out_stage_name: str
    _in_stage_names: list[str]
    _num_instances_per_stage: dict[str, int]
    _queue_specs: dict[str, QueueSpec]
    _max_size_per_queue: dict[str, int]

    def __init__(
        self: Self,
        *,
        output_stage_name: str,
        placement_spec: PipelinePlacementSpec,
        queues_spec: PipelineQueuesSpec,
        max_size_per_queue: dict[str, int]
    ) -> None:

        self._out_stage_name = output_stage_name

        self._in_stage_names = []
        self._queue_specs = {}
        for queue_name, queue_spec in queues_spec.queues.items():
            if self.get_router_stage_from_queue_spec(queue_spec) == output_stage_name:
                self._in_stage_names.append(queue_spec.input_stage_name)
                self._queue_specs[queue_name] = queue_spec

        self._num_instances_per_stage = {}
        for stage_name in (self._in_stage_names + [self._out_stage_name]):
            num_instances = len(placement_spec.stages[stage_name].instances)
            self._num_instances_per_stage[stage_name] = num_instances

        self._max_size_per_queue = max_size_per_queue

    @staticmethod
    def get_router_stage_from_queue_spec(
        queue_spec: QueueSpec
    ) -> str:

        return queue_spec.output_stage_name

    @classmethod
    @abc.abstractmethod
    def get_config_bytes(
        cls: type[Self],
        *args,
        **kwargs
    ) -> bytes:

        pass

    @classmethod
    @abc.abstractmethod
    def from_bytes(
        cls: type[Self],
        data: bytes
    ) -> Self:

        pass

    @abc.abstractmethod
    def max_size(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> int:

        pass

    @abc.abstractmethod
    def size(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> int:

        pass

    @abc.abstractmethod
    def remaining(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> int:

        pass

    @abc.abstractmethod
    def is_empty(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> bool:

        pass

    @abc.abstractmethod
    def is_full(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> bool:

        pass

    @abc.abstractmethod
    def put_nowait(
        self: Self,
        queue_name: str,
        in_instance_index: int,
        item: bytes,
        key: Hashable | None = None
    ) -> None:

        pass

    @abc.abstractmethod
    def get_nowait(
        self: Self,
        queue_name: str,
        out_instance_index: int
    ) -> bytes:

        pass

    @abc.abstractmethod
    async def put_async(
        self: Self,
        queue_name: str,
        in_instance_index: int,
        item: bytes,
        timeout: float | None = None,
        key: Hashable | None = None
    ) -> None:

        pass

    @abc.abstractmethod
    async def get_async(
        self: Self,
        queue_name: str,
        out_instance_index: int,
        timeout: float | None = None
    ) -> bytes:

        pass

    @abc.abstractmethod
    def put_nowait_batch(
        self: Self,
        queue_name: str,
        in_instance_index: int,
        items: list[bytes],
        keys: list[Hashable] | None = None
    ) -> None:

        pass

    @abc.abstractmethod
    def get_nowait_batch(
        self: Self,
        queue_name: str,
        out_instance_index: int,
        num_items: int | None = None,
        max_num_items: int | None = None,
        all_items: bool = False
    ) -> list[bytes]:

        pass

    @abc.abstractmethod
    async def put_batch_async(
        self: Self,
        queue_name: str,
        in_instance_index: int,
        items: list[bytes],
        timeout: float | None = None,
        keys: list[Hashable] | None = None
    ) -> None:

        pass

    @abc.abstractmethod
    async def get_batch_async(
        self: Self,
        queue_name: str,
        out_instance_index: int,
        num_items: int | None = None,
        max_num_items: int | None = None,
        all_items: bool = False,
        min_num_items: int | None = None,
        timeout: float | None = None
    ) -> list[bytes]:

        pass

    @abc.abstractmethod
    def mark_key_finished(
        self: Self,
        queue_name: str,
        out_instance_index: int,
        key: Hashable
    ) -> None:

        pass

    @abc.abstractmethod
    def mark_item_finished(
        self: Self,
        queue_name: str,
        out_instance_index: int
    ) -> None:

        pass
