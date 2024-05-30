# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 19:32:30 2024

@author: Shahir
"""

import time
import asyncio
import threading
from enum import ReprEnum
from dataclasses import dataclass
from collections.abc import Hashable
from typing import Self, final

from loguru import logger

import msgpack  # type: ignore

from alto_lib.core.manager import AsyncQueue, PipelinePlacementSpec, PipelineQueuesSpec
from alto_lib.core.routing._base import StageRouter


@final
@dataclass(kw_only=True, frozen=True)
class AugmentedQueueItem:
    item: bytes
    weight: float


@final
@dataclass(kw_only=True, frozen=False)
class ThroughputInfo:
    start_time: float
    num_items_processed: float


@final
class WeightEstimationStrategy(int, ReprEnum):
    EQUAL = 1
    BYTE_LENGTH = 2
    MANUAL = 3


@final
class SimpleBalancedRouter(StageRouter):
    _num_instances_out: int
    _out_queues_per_edge_per_instance: dict[str, dict[int, AsyncQueue[bytes]]]

    _estimate_throughput: bool
    _weight_strategy: WeightEstimationStrategy
    _key_lock: threading.Lock
    _key_to_out_instance: dict[Hashable, int]
    _unfinished_per_out_instance: dict[str, dict[int, float]]
    _throughput_per_out_instance: dict[str, dict[int, ThroughputInfo]]

    def __init__(
        self: Self,
        *,
        out_stage_name: str,
        placement_spec: PipelinePlacementSpec,
        queues_spec: PipelineQueuesSpec,
        max_size_per_queue: dict[str, int],
        estimate_throughput: bool = True,
        weight_strategy: WeightEstimationStrategy = WeightEstimationStrategy.EQUAL
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

        self._estimate_throughput = estimate_throughput
        self._weight_strategy = weight_strategy
        self._key_lock = threading.Lock()
        self._key_to_out_instance = {}
        self._unfinished_per_out_instance = {
            queue_name: {
                i: 0
                for i in range(self._num_instances_out)
            }
            for queue_name in self._queue_specs.keys()
        }
        start_time = time.monotonic()
        self._throughput_per_out_instance = {
            queue_name: {
                i: ThroughputInfo(start_time=start_time, num_items_processed=0)
                for i in range(self._num_instances_out)
            } for queue_name in self._queue_specs.keys()
        }

    @classmethod
    def get_config_bytes(
        cls: type[Self],
        output_stage_name: str,
        placement_spec: PipelinePlacementSpec,
        queues_spec: PipelineQueuesSpec,
        max_size_per_queue: dict[str, int],
        estimate_throughput: bool = True,
        weight_strategy: WeightEstimationStrategy = WeightEstimationStrategy.EQUAL
    ) -> bytes:

        return msgpack.packb((
            output_stage_name,
            placement_spec.to_bytes(),
            queues_spec.to_bytes(),
            max_size_per_queue,
            estimate_throughput,
            int(weight_strategy)
        ))

    @classmethod
    def from_bytes(
        cls: type[Self],
        data: bytes
    ) -> Self:

        (
            out_stage_name,
            placement_spec_bytes,
            queues_spec_bytes,
            max_size_per_queue,
            estimate_throughput,
            weight_strategy
        ) = msgpack.unpackb(data)

        return cls(
            out_stage_name=out_stage_name,
            placement_spec=PipelinePlacementSpec.from_bytes(placement_spec_bytes),
            queues_spec=PipelineQueuesSpec.from_bytes(queues_spec_bytes),
            max_size_per_queue=max_size_per_queue,
            estimate_throughput=estimate_throughput,
            weight_strategy=weight_strategy
        )

    def _get_instance_throughput(
        self: Self,
        queue_name: str,
        instance_index: int
    ) -> float:

        if self._estimate_throughput:
            info = self._throughput_per_out_instance[queue_name][instance_index]
            diff = time.monotonic() - info.start_time

            if info.num_items_processed == 0:
                return 0
            else:
                return diff / info.num_items_processed

        else:
            return 1

    def _get_item_info(
        self: Self,
        queue_name: str,
        item: bytes
    ) -> AugmentedQueueItem:

        match self._weight_strategy:
            case WeightEstimationStrategy.EQUAL:
                weight = 1
            case _:
                raise ValueError()

        item_info = AugmentedQueueItem(
            item=item,
            weight=weight
        )

        return item_info

    def _get_out_instance_index(
        self: Self,
        queue_name: str,
        item: bytes,
        key: Hashable | None
    ) -> int:

        self._key_lock.acquire()

        item_info = self._get_item_info(queue_name, item)
        preset_instance_index = self._key_to_out_instance.get(key, None)
        if (key is not None) and (preset_instance_index is not None):
            best_instance_index = preset_instance_index
        else:
            args_per_instance = {
                instance_index: (
                    (self._get_instance_throughput(queue_name, instance_index) * item_info.weight) +
                    self._unfinished_per_out_instance[queue_name][instance_index]
                )
                for instance_index in range(self._num_instances_out)
            }
            best_instance_index = min(args_per_instance.keys(), key=lambda i: args_per_instance[i])
            if key is not None:
                self._key_to_out_instance[key] = best_instance_index

        self._unfinished_per_out_instance[queue_name][best_instance_index] += item_info.weight

        self._key_lock.release()

        return best_instance_index

    def _arrange_put_batch(
        self: Self,
        queue_name: str,
        items: list[bytes],
        keys: list[Hashable] | None = None
    ) -> dict[int, list[bytes]]:

        items_per_out_index: dict[int, list[bytes]] = {i: [] for i in range(self._num_instances_out)}
        for i, item in enumerate(items):
            key = None if (keys is None) else keys[i]
            out_queue_index = self._get_out_instance_index(queue_name, item, key)
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
        out_queue_index = self._get_out_instance_index(queue_name, item, key)
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
        out_queue_index = self._get_out_instance_index(queue_name, item, key)
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

        self._key_to_out_instance.pop(key, None)

    def mark_item_finished(
        self: Self,
        queue_name: str,
        out_instance_index: int
    ) -> None:

        if self._estimate_throughput:
            self._throughput_per_out_instance[queue_name][out_instance_index].num_items_processed += 1
        self._unfinished_per_out_instance[queue_name][out_instance_index] -= 1
