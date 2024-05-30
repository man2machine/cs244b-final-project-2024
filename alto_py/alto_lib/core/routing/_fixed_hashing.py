# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 19:32:30 2024

@author: Shahir
"""

import asyncio
from collections.abc import Hashable
from enum import ReprEnum
from typing import Self, final
import threading

import msgpack  # type: ignore

from alto_lib.core.manager import AsyncQueue, PipelinePlacementSpec, PipelineQueuesSpec
from alto_lib.core.routing._base import StageRouter


from loguru import logger


@final
class RoutingType(int, ReprEnum):
    FIXED_HASHING = 1
    SIMPLE_THROUGHPUT_ESTIMATION = 2


@final
class HashingRouter(StageRouter):
    _num_instances_out: int
    _out_queues_per_edge_per_instance: dict[str, dict[int, AsyncQueue[bytes]]]
    _fallback_out_instance_indices: dict[str, int]
    _key_lock: threading.Lock

    def __init__(
        self: Self,
        *,
        out_stage_name: str,
        placement_spec: PipelinePlacementSpec,
        queues_spec: PipelineQueuesSpec,
        max_size_per_queue: dict[str, int]
    ) -> None:

        super().__init__(
            output_stage_name=out_stage_name,
            placement_spec=placement_spec,
            queues_spec=queues_spec,
            max_size_per_queue=max_size_per_queue
        )

        self._num_instances_out = self._num_instances_per_stage[self._out_stage_name]
        self._out_queues_per_edge_per_instance = {
            queue_name: {
                i: AsyncQueue(self._max_size_per_queue[queue_name])
                for i in range(self._num_instances_out)
            }
            for queue_name in self._queue_specs.keys()
        }
        self._fallback_out_instance_indices = {queue_name: 0 for queue_name in self._queue_specs.keys()}
        self._key_lock = threading.Lock()

    @classmethod
    def get_config_bytes(
        cls: type[Self],
        output_stage_name: str,
        placement_spec: PipelinePlacementSpec,
        queues_spec: PipelineQueuesSpec,
        max_size_per_queue: dict[str, int]
    ) -> bytes:

        return msgpack.packb(
            (output_stage_name, placement_spec.to_bytes(), queues_spec.to_bytes(), max_size_per_queue)
        )

    @classmethod
    def from_bytes(
        cls: type[Self],
        data: bytes
    ) -> Self:

        out_stage_name, placement_spec_bytes, queues_spec_bytes, max_size_per_queue = msgpack.unpackb(data)

        return cls(
            out_stage_name=out_stage_name,
            placement_spec=PipelinePlacementSpec.from_bytes(placement_spec_bytes),
            queues_spec=PipelineQueuesSpec.from_bytes(queues_spec_bytes),
            max_size_per_queue=max_size_per_queue
        )

    def _get_out_instance_index(
        self: Self,
        queue_name: str,
        key: Hashable | None
    ) -> int:

        self._key_lock.acquire()

        if key is None:
            out_queue_index = self._fallback_out_instance_indices[queue_name]
            self._fallback_out_instance_indices[queue_name] = (out_queue_index + 1) % self._num_instances_out
        else:
            out_queue_index = hash(key) % self._num_instances_out

        self._key_lock.release()

        return out_queue_index

    def _arrange_put_batch(
        self: Self,
        queue_name: str,
        items: list[bytes],
        keys: list[Hashable] | None = None
    ) -> dict[int, list[bytes]]:

        items_per_out_index: dict[int, list[bytes]] = {i: [] for i in range(self._num_instances_out)}
        for i, item in enumerate(items):
            key = None if (keys is None) else keys[i]
            out_queue_index = self._get_out_instance_index(queue_name, key)
            items_per_out_index[out_queue_index].append(item)

        return items_per_out_index

    def max_size(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> int:

        return self._max_size_per_queue[queue_name]

    def size(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> int:

        out_queues = self._out_queues_per_edge_per_instance[queue_name]
        if not in_queue:
            return out_queues[instance_index].size()
        else:
            return sum(out_queues[i].size() for i in range(self._num_instances_out))

    def remaining(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> int:

        out_queues = self._out_queues_per_edge_per_instance[queue_name]
        if not in_queue:
            return out_queues[instance_index].remaining()
        else:
            return -1

    def is_empty(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> bool:

        out_queues = self._out_queues_per_edge_per_instance[queue_name]
        if not in_queue:
            return out_queues[instance_index].is_empty()
        else:
            return all(out_queues[i].is_empty() for i in range(self._num_instances_out))

    def is_full(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> bool:

        out_queues = self._out_queues_per_edge_per_instance[queue_name]
        if not in_queue:
            return out_queues[instance_index].is_full()
        else:
            return all(out_queues[i].is_full() for i in range(self._num_instances_out))

    def put_nowait(
        self: Self,
        queue_name: str,
        in_instance_index: int,
        item: bytes,
        key: Hashable | None = None
    ) -> None:

        out_queues = self._out_queues_per_edge_per_instance[queue_name]
        out_queue_index = self._get_out_instance_index(queue_name, key)
        out_queues[out_queue_index].put_nowait(item)

    def get_nowait(
        self: Self,
        queue_name: str,
        out_instance_index: int
    ) -> bytes:

        out_queues = self._out_queues_per_edge_per_instance[queue_name]
        return out_queues[out_instance_index].get_nowait()

    async def put_async(
        self: Self,
        queue_name: str,
        in_instance_index: int,
        item: bytes,
        timeout: float | None = None,
        key: Hashable | None = None
    ) -> None:

        out_queues = self._out_queues_per_edge_per_instance[queue_name]
        out_queue_index = self._get_out_instance_index(queue_name, key)
        await out_queues[out_queue_index].put_async(item, timeout=timeout)

    async def get_async(
        self: Self,
        queue_name: str,
        out_instance_index: int,
        timeout: float | None = None
    ) -> bytes:

        out_queues = self._out_queues_per_edge_per_instance[queue_name]
        return await out_queues[out_instance_index].get_async(timeout=timeout)

    def put_nowait_batch(
        self: Self,
        queue_name: str,
        in_instance_index: int,
        items: list[bytes],
        keys: list[Hashable] | None = None
    ) -> None:

        out_queues = self._out_queues_per_edge_per_instance[queue_name]
        items_per_out_index = self._arrange_put_batch(queue_name, items, keys)
        for out_index, instance_items in items_per_out_index.items():
            if instance_items:
                out_queues[out_index].put_nowait_batch(instance_items)

    def get_nowait_batch(
        self: Self,
        queue_name: str,
        out_instance_index: int,
        num_items: int | None = None,
        max_num_items: int | None = None,
        all_items: bool = False
    ) -> list[bytes]:

        out_queues = self._out_queues_per_edge_per_instance[queue_name]
        return out_queues[out_instance_index].get_nowait_batch(
            num_items=num_items,
            max_num_items=max_num_items,
            all_items=all_items
        )

    async def put_batch_async(
        self: Self,
        queue_name: str,
        in_instance_index: int,
        items: list[bytes],
        timeout: float | None = None,
        keys: list[Hashable] | None = None
    ) -> None:

        out_queues = self._out_queues_per_edge_per_instance[queue_name]
        items_per_out_index = self._arrange_put_batch(queue_name, items, keys)
        coros = [
            out_queues[out_index].put_batch_async(instance_items, timeout=timeout)
            for out_index, instance_items in items_per_out_index.items()
            if instance_items
        ]

        if coros:
            await asyncio.gather(*coros)

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

        out_queues = self._out_queues_per_edge_per_instance[queue_name]
        return await out_queues[out_instance_index].get_batch_async(
            num_items=num_items,
            max_num_items=max_num_items,
            all_items=all_items,
            min_num_items=min_num_items,
            timeout=timeout
        )

    def mark_key_finished(
        self: Self,
        queue_name: str,
        out_instance_index: int,
        key: Hashable
    ) -> None:

        pass

    def mark_item_finished(
        self: Self,
        queue_name: str,
        out_instance_index: int
    ) -> None:

        pass
