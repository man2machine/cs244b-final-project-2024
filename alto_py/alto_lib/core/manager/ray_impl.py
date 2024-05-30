# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 04:07:16 2023

@author: Shahir

Inspired from ray.util.queue
"""

import asyncio
from typing import Self, Any, Generic, final, cast, get_type_hints
from collections.abc import Callable, Coroutine, Hashable

import ray
import ray.actor
import ray.runtime_context

from loguru import logger
import ujson as json  # type: ignore
import msgpack  # type: ignore

from alto_lib.core.manager._base import (
    StageManager, StageCommunicatorFactoryParams, StageCommunicator, StageCommunicatorType, StageFuncParamsT,
    StageInputQueueInterface, StageOutputQueueInterface, QueueItemSerializer, QueueItemT,
    get_queue_name, get_instance_name, get_stage_name_from_instance_name, get_instance_index_from_instance_name
)
from alto_lib.core.manager._proto import PipelineQueuesSpec
from alto_lib.core.routing import RoutingType, StageRouter, HashingRouter, SimpleBalancedRouter
from alto_lib.core.app_endpoints import InstanceQueueLatencyLogger


TIMEOUT_NEGATIVE_ERROR_MSG: str = "'timeout' must be a non-negative number"
TRACK_QUEUE_LATENCY: bool = True


@final
class _RayStageRoutingActor:
    _routers_per_stage: dict[str, StageRouter]
    _routers_per_queue: dict[str, StageRouter]

    def __init__(
        self: Self,
        queues_spec: PipelineQueuesSpec,
        routing_type: RoutingType,
        router_configs_per_stage: dict[str, bytes]
    ) -> None:

        router_class: type[StageRouter]
        match routing_type:
            case RoutingType.FIXED_HASHING:
                router_class = HashingRouter
            case RoutingType.SIMPLE_ESTIMATION:
                router_class = SimpleBalancedRouter
            case _:
                raise ValueError()

        self._routers_per_stage = {
            stage_name: router_class.from_bytes(config_bytes)
            for stage_name, config_bytes in router_configs_per_stage.items()
        }
        self._routers_per_queue = {
            queue_spec.queue_name: self._routers_per_stage[queue_spec.output_stage_name]
            for queue_spec in queues_spec.queues.values()
            if StageRouter.get_router_stage_from_queue_spec(queue_spec) in router_configs_per_stage.keys()
        }

    def max_size(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> int:

        return self._routers_per_queue[queue_name].max_size(
            queue_name=queue_name,
            instance_index=instance_index,
            in_queue=in_queue
        )

    def size(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> int:

        return self._routers_per_queue[queue_name].size(
            queue_name=queue_name,
            instance_index=instance_index,
            in_queue=in_queue
        )

    def remaining(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> int:

        return self._routers_per_queue[queue_name].remaining(
            queue_name=queue_name,
            instance_index=instance_index,
            in_queue=in_queue
        )

    def is_empty(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> bool:

        return self._routers_per_queue[queue_name].is_empty(
            queue_name=queue_name,
            instance_index=instance_index,
            in_queue=in_queue
        )

    def is_full(
        self: Self,
        queue_name: str,
        instance_index: int,
        in_queue: bool = True
    ) -> bool:

        return self._routers_per_queue[queue_name].is_full(
            queue_name=queue_name,
            instance_index=instance_index,
            in_queue=in_queue
        )

    def put_nowait(
        self: Self,
        queue_name: str,
        in_instance_index: int,
        item: bytes,
        key: Hashable | None = None
    ) -> None:

        self._routers_per_queue[queue_name].put_nowait(
            queue_name=queue_name,
            in_instance_index=in_instance_index,
            item=item,
            key=key
        )

    def get_nowait(
        self: Self,
        queue_name: str,
        out_instance_index: int
    ) -> bytes:

        return self._routers_per_queue[queue_name].get_nowait(
            queue_name=queue_name,
            out_instance_index=out_instance_index
        )

    async def put_async(
        self: Self,
        queue_name: str,
        in_instance_index: int,
        item: bytes,
        timeout: float | None = None,
        key: Hashable | None = None
    ) -> None:

        await self._routers_per_queue[queue_name].put_async(
            queue_name=queue_name,
            in_instance_index=in_instance_index,
            item=item,
            timeout=timeout,
            key=key
        )

    async def get_async(
        self: Self,
        queue_name: str,
        out_instance_index: int,
        timeout: float | None = None
    ) -> bytes:

        return await self._routers_per_queue[queue_name].get_async(
            queue_name=queue_name,
            out_instance_index=out_instance_index,
            timeout=timeout
        )

    def put_nowait_batch(
        self: Self,
        queue_name: str,
        in_instance_index: int,
        items: list[bytes],
        keys: list[Hashable] | None = None
    ) -> None:

        self._routers_per_queue[queue_name].put_nowait_batch(
            queue_name=queue_name,
            in_instance_index=in_instance_index,
            items=items,
            keys=keys
        )

    def get_nowait_batch(
        self: Self,
        queue_name: str,
        out_instance_index: int,
        num_items: int | None = None,
        max_num_items: int | None = None,
        all_items: bool = False
    ) -> list[bytes]:

        return self._routers_per_queue[queue_name].get_nowait_batch(
            queue_name=queue_name,
            out_instance_index=out_instance_index,
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

        await self._routers_per_queue[queue_name].put_batch_async(
            queue_name=queue_name,
            in_instance_index=in_instance_index,
            items=items,
            timeout=timeout,
            keys=keys
        )

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

        return await self._routers_per_queue[queue_name].get_batch_async(
            queue_name=queue_name,
            out_instance_index=out_instance_index,
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

        return self._routers_per_queue[queue_name].mark_key_finished(
            queue_name=queue_name,
            out_instance_index=out_instance_index,
            key=key
        )

    def mark_item_finished(
        self: Self,
        queue_name: str,
        out_instance_index: int
    ) -> None:

        return self._routers_per_queue[queue_name].mark_item_finished(
            queue_name=queue_name,
            out_instance_index=out_instance_index
        )


@final
class _RayStageControlActor:
    _instance_names: list[str]
    _ready_per_instance: dict[str, asyncio.Event]
    _stop: bool

    def __init__(
        self: Self,
        instance_names: list[str]
    ) -> None:

        self._instance_names = instance_names
        self._ready_per_instance = {n: asyncio.Event() for n in self._instance_names}
        self._stop = False

    def set_instance_ready(
        self: Self,
        instance_name: str
    ) -> None:

        self._ready_per_instance[instance_name].set()

    async def wait_all_instances_ready(
        self: Self
    ) -> None:

        for e in self._ready_per_instance.values():
            await e.wait()

    def set_global_stop(
        self: Self
    ) -> None:

        self._stop = True

    def should_stop(
        self: Self
    ) -> bool:

        return self._stop


@final
class _RayInstanceExecutorActor(Generic[StageFuncParamsT]):
    _main_func: Callable[StageFuncParamsT, None] | Callable[StageFuncParamsT, Coroutine[Any, Any, None]]
    _args: tuple[Any, ...]
    _kwargs: dict[str, Any]

    def __init__(
        self: Self,
        main_func: Callable[StageFuncParamsT, None] | Callable[StageFuncParamsT, Coroutine[Any, Any, None]],
        *args: StageFuncParamsT.args,
        **kwargs: StageFuncParamsT.kwargs
    ) -> None:

        self._main_func = main_func
        self._args = args
        self._kwargs = kwargs

    async def run(
        self: Self
    ) -> None:

        if asyncio.iscoroutinefunction(self._main_func):
            await self._main_func(*self._args, **self._kwargs)
        else:
            self._main_func(*self._args, **self._kwargs)


@final
class _RayInputQueueInterface(Generic[QueueItemT], StageInputQueueInterface[QueueItemT]):
    _queue_name: str
    _instance_index: int
    _routing_actor: ray.actor.ActorClass | _RayStageRoutingActor  # IDE highlighting
    _serializer: QueueItemSerializer[QueueItemT] | None

    def __init__(
        self: Self,
        queue_name: str,
        instance_name: str,
        routing_actor: ray.actor.ActorClass,
        serializer: QueueItemSerializer[QueueItemT] | None = None
    ) -> None:

        self._queue_name = queue_name
        self._instance_index = get_instance_index_from_instance_name(instance_name)
        self._routing_actor = routing_actor
        self._serializer = serializer

        max_size = cast(int, ray.get(
            self._routing_actor.max_size.remote(  # type: ignore
                queue_name=queue_name,
                instance_index=self._instance_index,
                in_queue=False
            )
        ))
        super().__init__(queue_name=queue_name, instance_name=instance_name, max_size=max_size)

    def _deserialize_item(
        self: Self,
        item: Any
    ) -> QueueItemT:

        if self._serializer:
            return self._serializer.from_bytes(item)
        else:
            return item

    def _deserialize_items(
        self: Self,
        items: list[Any]
    ) -> list[QueueItemT]:

        if self._serializer:
            return [self._serializer.from_bytes(item) for item in items]
        else:
            return items

    def size(
        self: Self,
    ) -> int:

        return ray.get(
            self._routing_actor.size.remote(  # type: ignore
                queue_name=self._queue_name,
                instance_index=self._instance_index,
                in_queue=False
            )
        )

    def remaining(
        self: Self
    ) -> int:

        return ray.get(
            self._routing_actor.remaining.remote(  # type: ignore
                queue_name=self._queue_name,
                instance_index=self._instance_index,
                in_queue=False
            )
        )

    def is_empty(
        self: Self,
    ) -> bool:

        return ray.get(
            self._routing_actor.is_empty.remote(  # type: ignore
                queue_name=self._queue_name,
                instance_index=self._instance_index,
                in_queue=False
            )
        )

    def is_full(
        self: Self,
    ) -> bool:

        return ray.get(
            self._routing_actor.is_full.remote(  # type: ignore
                queue_name=self._queue_name,
                instance_index=self._instance_index,
                in_queue=False
            )
        )

    def get(
        self,
        block: bool = True,
        timeout: float | None = None
    ) -> QueueItemT:

        if not block:
            item = ray.get(
                self._routing_actor.get_nowait.remote(  # type: ignore
                    queue_name=self._queue_name,
                    out_instance_index=self._instance_index
                )
            )
            return self._deserialize_item(item)
        else:
            if timeout is not None and timeout < 0:
                raise ValueError(TIMEOUT_NEGATIVE_ERROR_MSG)
            else:
                item = ray.get(
                    self._routing_actor.get_async.remote(  # type: ignore
                        queue_name=self._queue_name,
                        out_instance_index=self._instance_index,
                        timeout=timeout
                    )
                )
                return self._deserialize_item(item)

    async def get_async(
        self: Self,
        block: bool = True,
        timeout: float | None = None
    ) -> QueueItemT:

        if not block:
            item = await self._routing_actor.get_nowait.remote(  # type: ignore
                queue_name=self._queue_name,
                out_instance_index=self._instance_index
            )
            return self._deserialize_item(item)
        else:
            if timeout is not None and timeout < 0:
                raise ValueError(TIMEOUT_NEGATIVE_ERROR_MSG)
            else:
                item = await self._routing_actor.get_async.remote(  # type: ignore
                    queue_name=self._queue_name,
                    out_instance_index=self._instance_index,
                    timeout=timeout
                )
                return self._deserialize_item(item)

    def get_batch(
        self: Self,
        num_items: int | None = None,
        max_num_items: int | None = None,
        all_items: bool = False,
        min_num_items: int | None = None,
        block: bool = True,
        timeout: float | None = None
    ) -> list[QueueItemT]:

        if not block:
            items = ray.get(
                self._routing_actor.get_nowait_batch.remote(  # type: ignore
                    queue_name=self._queue_name,
                    out_instance_index=self._instance_index,
                    num_items=num_items,
                    max_num_items=max_num_items,
                    all_items=all_items
                )
            )
            return self._deserialize_items(items)
        else:
            if timeout is not None and timeout < 0:
                raise ValueError(TIMEOUT_NEGATIVE_ERROR_MSG)
            else:
                items = ray.get(
                    self._routing_actor.get_batch_async.remote(  # type: ignore
                        queue_name=self._queue_name,
                        out_instance_index=self._instance_index,
                        num_items=num_items,
                        max_num_items=max_num_items,
                        all_items=all_items,
                        min_num_items=min_num_items,
                        timeout=timeout
                    )
                )
                return self._deserialize_items(items)

    async def get_batch_async(
        self: Self,
        num_items: int | None = None,
        max_num_items: int | None = None,
        all_items: bool = False,
        min_num_items: int | None = None,
        block: bool = True,
        timeout: float | None = None
    ) -> list[QueueItemT]:

        if not block:
            items = await self._routing_actor.get_nowait_batch.remote(  # type: ignore
                queue_name=self._queue_name,
                out_instance_index=self._instance_index,
                num_items=num_items,
                max_num_items=max_num_items,
                all_items=all_items
            )
            return self._deserialize_items(items)
        else:
            if timeout is not None and timeout < 0:
                raise ValueError(TIMEOUT_NEGATIVE_ERROR_MSG)
            else:
                items = await self._routing_actor.get_batch_async.remote(  # type: ignore
                    queue_name=self._queue_name,
                    out_instance_index=self._instance_index,
                    num_items=num_items,
                    max_num_items=max_num_items,
                    all_items=all_items,
                    min_num_items=min_num_items,
                    timeout=timeout
                )
                return self._deserialize_items(items)

    def mark_key_finished(
        self: Self,
        key: Hashable
    ) -> None:

        ray.get(
            self._routing_actor.mark_key_finished.remote(  # type: ignore
                queue_name=self._queue_name,
                out_instance_index=self._instance_index,
                key=key
            )
        )

    def mark_item_finished(
        self: Self
    ) -> None:

        ray.get(
            self._routing_actor.mark_item_finished.remote(  # type: ignore
                queue_name=self._queue_name,
                out_instance_index=self._instance_index
            )
        )


@final
class _RayOutputQueueInterface(Generic[QueueItemT], StageOutputQueueInterface[QueueItemT]):
    _queue_name: str
    _instance_index: int
    _routing_actor: ray.actor.ActorClass | _RayStageRoutingActor  # IDE highlighting
    _serializer: QueueItemSerializer[QueueItemT] | None

    def __init__(
        self: Self,
        queue_name: str,
        instance_name: str,
        routing_actor: ray.actor.ActorClass,
        serializer: QueueItemSerializer[QueueItemT] | None = None
    ) -> None:

        self._queue_name = queue_name
        self._instance_index = get_instance_index_from_instance_name(instance_name)
        self._routing_actor = routing_actor
        self._serializer = serializer

        max_size = cast(int, ray.get(
            self._routing_actor.max_size.remote(  # type: ignore
                queue_name=queue_name,
                instance_index=self._instance_index,
                in_queue=False
            )
        ))
        super().__init__(queue_name=queue_name, instance_name=instance_name, max_size=max_size)

    def _serialize_item(
        self: Self,
        item: QueueItemT
    ) -> bytes | QueueItemT:

        if self._serializer:
            return self._serializer.to_bytes(item)
        else:
            return item

    def _serialize_items(
        self: Self,
        items: list[QueueItemT]
    ) -> list[bytes] | list[QueueItemT]:

        if self._serializer:
            return [self._serializer.to_bytes(item) for item in items]
        else:
            return items

    def size(
        self: Self,
    ) -> int:

        return ray.get(
            self._routing_actor.size.remote(  # type: ignore
                queue_name=self._queue_name,
                instance_index=self._instance_index,
                in_queue=True
            )
        )

    def remaining(
        self: Self
    ) -> int:

        return ray.get(
            self._routing_actor.remaining.remote(  # type: ignore
                queue_name=self._queue_name,
                instance_index=self._instance_index,
                in_queue=True
            )
        )

    def is_empty(
        self: Self,
    ) -> bool:

        return ray.get(
            self._routing_actor.is_empty.remote(  # type: ignore
                queue_name=self._queue_name,
                instance_index=self._instance_index,
                in_queue=True
            )
        )

    def is_full(
        self: Self,
    ) -> bool:

        return ray.get(
            self._routing_actor.is_full.remote(  # type: ignore
                queue_name=self._queue_name,
                instance_index=self._instance_index,
                in_queue=True
            )
        )

    def put(
        self: Self,
        item: QueueItemT,
        block: bool = True,
        timeout: float | None = None,
        key: Hashable | None = None
    ) -> None:

        put_item = self._serialize_item(item)

        if not block:
            ray.get(
                self._routing_actor.put_nowait.remote(  # type: ignore
                    queue_name=self._queue_name,
                    in_instance_index=self._instance_index,
                    item=put_item,
                    key=key
                )
            )
        else:
            if timeout is not None and timeout < 0:
                raise ValueError(TIMEOUT_NEGATIVE_ERROR_MSG)
            else:
                ray.get(
                    self._routing_actor.put_async.remote(  # type: ignore
                        queue_name=self._queue_name,
                        in_instance_index=self._instance_index,
                        item=put_item,
                        timeout=timeout,
                        key=key
                    )
                )

    async def put_async(
        self: Self,
        item: QueueItemT,
        block: bool = True,
        timeout: float | None = None,
        key: Hashable | None = None
    ) -> None:

        put_item = self._serialize_item(item)

        if not block:
            await self._routing_actor.put_nowait.remote(  # type: ignore
                queue_name=self._queue_name,
                in_instance_index=self._instance_index,
                item=put_item,
                key=key
            )
        else:
            if timeout is not None and timeout < 0:
                raise ValueError(TIMEOUT_NEGATIVE_ERROR_MSG)
            else:
                await self._routing_actor.put_async.remote(  # type: ignore
                    queue_name=self._queue_name,
                    in_instance_index=self._instance_index,
                    item=put_item,
                    timeout=timeout
                )

    def put_batch(
        self: Self,
        items: list[QueueItemT],
        block: bool = True,
        timeout: float | None = None,
        keys: list[Hashable] | None = None
    ) -> None:

        put_items = self._serialize_items(items)

        if not block:
            ray.get(
                self._routing_actor.put_nowait_batch.remote(  # type: ignore
                    queue_name=self._queue_name,
                    in_instance_index=self._instance_index,
                    items=put_items,
                    keys=keys
                )
            )
        else:
            if timeout is not None and timeout < 0:
                raise ValueError(TIMEOUT_NEGATIVE_ERROR_MSG)
            else:
                ray.get(
                    self._routing_actor.put_batch_async.remote(  # type: ignore
                        queue_name=self._queue_name,
                        in_instance_index=self._instance_index,
                        items=put_items,
                        timeout=timeout,
                        keys=keys
                    )
                )

    async def put_batch_async(
        self: Self,
        items: list[QueueItemT],
        block: bool = True,
        timeout: float | None = None,
        keys: list[Hashable] | None = None
    ) -> None:

        put_items = self._serialize_items(items)

        if not block:
            await self._routing_actor.put_nowait_batch.remote(  # type: ignore
                queue_name=self._queue_name,
                in_instance_index=self._instance_index,
                items=put_items,
                keys=keys
            )
        else:
            if timeout is not None and timeout < 0:
                raise ValueError(TIMEOUT_NEGATIVE_ERROR_MSG)
            else:
                await self._routing_actor.put_batch_async.remote(  # type: ignore
                    queue_name=self._queue_name,
                    in_instance_index=self._instance_index,
                    items=put_items,
                    timeout=timeout,
                    keys=keys
                )


class LoggingSerializer(QueueItemSerializer[QueueItemT]):
    _serializer: QueueItemSerializer[QueueItemT]
    _instance_name: str
    _log: InstanceQueueLatencyLogger

    def __init__(
        self: Self,
        serializer: QueueItemSerializer[QueueItemT],
        instance_name: str
    ) -> None:

        self._serializer = serializer
        self._instance_name = instance_name

        from alto_lib.core.manager._utils import QueueRequestMessageSerializer
        s = cast(QueueRequestMessageSerializer, serializer)
        item_type = s.message_type

        self._log = InstanceQueueLatencyLogger(
            instance_name,
            item_type.__name__
        )
        self._log.log_metadata()

    def to_bytes(
        self: Self,
        item: QueueItemT
    ) -> bytes:

        buf = self._serializer.to_bytes(item)
        buf = self._log.send_message(buf)
        self._log.update_log()

        return buf

    def from_bytes(
        self: Self,
        buf: bytes
    ) -> QueueItemT:

        buf = self._log.recv_messsage(buf)
        item = self._serializer.from_bytes(buf)
        self._log.update_log()

        return item


@final
class RayStageCommunicator(StageCommunicator):
    _queue_to_routing_actor_name: dict[str, str]
    _control_actor: ray.actor.ActorClass | _RayStageControlActor  # IDE highlighting
    _control_actor_name: str
    _instance_name: str
    _stage_name: str

    def __init__(
        self: Self,
        queue_to_routing_actor_name: dict[str, str],
        control_actor_name: str
    ) -> None:

        super().__init__()
        self._queue_to_routing_actor_name = queue_to_routing_actor_name
        self._control_actor_name = control_actor_name

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        queue_to_routing_actor_name, control_actor_name = msgpack.unpackb(buf)

        return cls(queue_to_routing_actor_name, control_actor_name)

    def initialize(
        self: Self
    ) -> None:

        self._control_actor = ray.get_actor(self._control_actor_name)

        context: ray.runtime_context.RuntimeContext = ray.get_runtime_context()
        actor_name = context.get_actor_name()
        assert actor_name is not None

        self._instance_name = actor_name
        self._stage_name = get_stage_name_from_instance_name(self._instance_name)

        super().initialize()

    def get_instance_name(
        self: Self
    ) -> str:

        context: ray.runtime_context.RuntimeContext = ray.get_runtime_context()
        actor_name = context.get_actor_name()
        assert actor_name is not None

        return actor_name

    def get_input_queue_interface(
        self: Self,
        input_stage_name: str,
        output_stage_name: str,
        index: int,
        item_type: type[QueueItemT],
        serializer: QueueItemSerializer[QueueItemT]
    ) -> StageInputQueueInterface[QueueItemT]:

        assert output_stage_name == self._stage_name

        queue_name = get_queue_name(
            input_stage_name,
            output_stage_name,
            index
        )
        routing_actor = ray.get_actor(self._queue_to_routing_actor_name[queue_name])

        serializer = LoggingSerializer(serializer, self.get_instance_name())

        return _RayInputQueueInterface(
            queue_name=queue_name,
            instance_name=self._instance_name,
            routing_actor=routing_actor,
            serializer=serializer
        )

    def get_output_queue_interface(
        self: Self,
        input_stage_name: str,
        output_stage_name: str,
        index: int,
        item_type: type[QueueItemT],
        serializer: QueueItemSerializer[QueueItemT]
    ) -> StageOutputQueueInterface[QueueItemT]:

        assert input_stage_name == self._stage_name

        queue_name = get_queue_name(
            input_stage_name,
            output_stage_name,
            index
        )
        routing_actor = ray.get_actor(self._queue_to_routing_actor_name[queue_name])

        serializer = LoggingSerializer(serializer, self.get_instance_name())

        return _RayOutputQueueInterface(
            queue_name=queue_name,
            instance_name=self._instance_name,
            routing_actor=routing_actor,
            serializer=serializer
        )

    def signal_instance_ready(
        self: Self
    ) -> None:

        ray.get(self._control_actor.set_instance_ready.remote(self._instance_name))  # type: ignore

    async def wait_all_instances_ready(
        self: Self
    ) -> None:

        await self._control_actor.wait_all_instances_ready.remote()  # type: ignore

    def signal_global_stop(
        self: Self
    ) -> None:

        ray.get(self._control_actor.set_global_stop.remote())  # type: ignore

    def should_stop(
        self: Self
    ) -> bool:

        return ray.get(self._control_actor.should_stop.remote())  # type: ignore


@final
class RayStageManager(StageManager):
    _ROUTING_ACTOR_NAME_PREFIX: str = "routing_actor"
    _CONTROL_ACTOR_NAME: str = "control_actor"

    _routing_type: RoutingType
    _routing_actors: dict[str, ray.actor.ActorClass]
    _control_actor: ray.actor.ActorClass
    _instance_actors: dict[str, ray.actor.ActorClass]

    def __init__(
        self: Self,
        num_cpus: int = 1,
        num_gpus: int = 1,
        routing_type: RoutingType = RoutingType.FIXED_HASHING,
        tmp_dir: str = "/tmp/spill"
    ) -> None:

        super().__init__()

        self._routing_type = routing_type

        # from https://docs.ray.io/en/latest/ray-core/objects/object-spilling.htm
        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            ignore_reinit_error=True,
            _system_config={
                "object_spilling_config": json.dumps(
                    {"type": "filesystem", "params": {"directory_path": tmp_dir}},
                )
            }
        )

    @classmethod
    def _get_routing_actor_name(
        cls: type[Self],
        stage_name: str
    ) -> str:

        return "{}-{}".format(cls._ROUTING_ACTOR_NAME_PREFIX, stage_name)

    def _get_proxy_factory_params(
        self: Self
    ) -> StageCommunicatorFactoryParams:

        queues_spec = self.get_queues_spec()

        queue_to_routing_actor_name: dict[str, str] = {}
        for queue_name, queue_spec in queues_spec.queues.items():
            stage_name = StageRouter.get_router_stage_from_queue_spec(queue_spec)
            queue_to_routing_actor_name[queue_name] = self._get_routing_actor_name(stage_name)

        params = StageCommunicatorFactoryParams(
            communicator_type=StageCommunicatorType.RAY,
            config_data=msgpack.packb((queue_to_routing_actor_name, self._CONTROL_ACTOR_NAME))
        )

        return params

    def initialize(
        self: Self
    ) -> None:

        placement_spec = self.get_placement_spec()
        queues_spec = self.get_queues_spec()

        # control actor
        logger.debug("Starting stage control actor")

        instance_names = []
        for stage_name, stage_params in self._stage_name_to_params.items():
            for i in range(len(placement_spec.stages[stage_name].instances)):
                instance_names.append(get_instance_name(stage_name, i))

        self._control_actor = ray.remote(_RayStageControlActor).options(  # type: ignore
            name=self._CONTROL_ACTOR_NAME,
            num_cpus=1
        ).remote(
            instance_names=instance_names
        )

        # routing actors
        logger.debug("Starting routing actors")

        max_size_per_queue: dict[str, int] = {}
        for queue_name, queue_params in self._queue_name_to_params.items():
            max_size_per_queue[queue_name] = queue_params.max_size

        router_configs_per_actor_per_stage: dict[str, dict[str, bytes]] = {}
        for stage_name in placement_spec.stages.keys():
            match self._routing_type:
                case RoutingType.FIXED_HASHING:
                    config_bytes = HashingRouter.get_config_bytes(
                        output_stage_name=stage_name,
                        placement_spec=placement_spec,
                        queues_spec=queues_spec,
                        max_size_per_queue=max_size_per_queue
                    )
                case RoutingType.SIMPLE_ESTIMATION:
                    config_bytes = SimpleBalancedRouter.get_config_bytes(
                        output_stage_name=stage_name,
                        placement_spec=placement_spec,
                        queues_spec=queues_spec,
                        max_size_per_queue=max_size_per_queue
                    )
                case _:
                    raise ValueError()

            routing_actor_name = self._get_routing_actor_name(stage_name)
            router_configs_per_actor_per_stage.setdefault(routing_actor_name, {})
            router_configs_per_actor_per_stage[routing_actor_name] = {stage_name: config_bytes}

        self._routing_actors = {}
        for actor_name, router_configs_per_stage in router_configs_per_actor_per_stage.items():
            self._routing_actors[actor_name] = ray.remote(_RayStageRoutingActor).options(  # type: ignore
                name=actor_name,
                num_cpus=1
            ).remote(
                queues_spec=queues_spec,
                routing_type=self._routing_type,
                router_configs_per_stage=router_configs_per_stage
            )

        self._instance_actors = {}
        for stage_name, stage_params in self._stage_name_to_params.items():
            logger.debug("Starting {} actors".format(stage_name))
            for i, instance_spec in enumerate(placement_spec.stages[stage_name].instances):
                instance_name = get_instance_name(stage_name, i)
                instance_actor = ray.remote(_RayInstanceExecutorActor).options(  # type: ignore
                    name=instance_name,
                    num_cpus=instance_spec.num_cpus,
                    num_gpus=(instance_spec.num_gpus + instance_spec.num_llm_gpus)
                ).remote(
                    stage_params.run_func,
                    *stage_params.run_func_params[0],
                    **stage_params.run_func_params[1]
                )
                self._instance_actors[instance_name] = instance_actor

        super().initialize()

    def run_all_stages(
        self: Self
    ) -> None:

        instance_actor_outs = []
        for stage_name, instance_actor in self._instance_actors.items():
            logger.debug("Starting stage {}".format(stage_name))
            obj_ref = instance_actor.run.remote()  # type: ignore
            instance_actor_outs.append(obj_ref)

        ray.get(instance_actor_outs)
        print("Completed all stages")
