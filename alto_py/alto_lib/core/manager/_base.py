# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 04:07:16 2023

@author: Shahir
"""

import re
import abc
from collections.abc import Callable, Coroutine, Hashable
from dataclasses import dataclass
from enum import ReprEnum
from typing import Self, TypeVar, ParamSpec, Generic, Any, final

import msgpack  # type: ignore

from alto_lib.core.manager._proto import (
    PipelinePlacementSpec, StagePlacementSpec, InstancePlacementSpec, PipelineQueuesSpec, QueueSpec
)
from alto_lib.utils import BytesSerializable


StageFuncParamsT = ParamSpec('StageFuncParamsT')
StageCommunicatorParamsT = ParamSpec('StageCommunicatorParamsT')
QueueItem = Any
QueueItemT = TypeVar('QueueItemT', bound=QueueItem)


def get_queue_name(
    input_stage_name: str,
    output_stage_name: str,
    edge_index: int
) -> str:

    assert "-" not in input_stage_name
    assert "-" not in output_stage_name
    assert "->" not in input_stage_name
    assert "->" not in output_stage_name

    return "({} -> {})-{}".format(input_stage_name, output_stage_name, edge_index)


def get_instance_name(
    stage_name: str,
    instance_index: int
) -> str:

    assert "-" not in stage_name

    return "{}-{}".format(stage_name, instance_index)


def get_stage_names_from_queue_name(
    queue_name: str
) -> tuple[str, str]:

    match = re.search(r"^\(([\w]+) -> ([\w]+)\)-[\d]+$", queue_name)
    assert match is not None
    input_stage_name = match.group(1)
    output_stage_name = match.group(2)

    return input_stage_name, output_stage_name


def get_stage_name_from_instance_name(
    instance_name: str
) -> str:

    return instance_name.rpartition("-")[0]


def get_instance_index_from_instance_name(
    instance_name: str
) -> int:

    return int(instance_name.rpartition("-")[2])


class QueueItemSerializer(Generic[QueueItemT], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_bytes(
        self: Self,
        item: QueueItemT
    ) -> bytes:

        pass

    @abc.abstractmethod
    def from_bytes(
        self: Self,
        data: bytes
    ) -> QueueItemT:

        pass


class StageInputQueueInterface(Generic[QueueItemT], metaclass=abc.ABCMeta):
    max_size: int

    def __init__(
        self: Self,
        queue_name: str,
        instance_name: str,
        max_size: int
    ) -> None:

        stage_name = get_stage_name_from_instance_name(instance_name)
        _, output_stage_name = get_stage_names_from_queue_name(queue_name)
        assert stage_name == output_stage_name

        self.max_size = max_size

    @abc.abstractmethod
    def size(
        self: Self
    ) -> int:

        pass

    @abc.abstractmethod
    def remaining(
        self: Self
    ) -> int:

        pass

    @abc.abstractmethod
    def is_empty(
        self: Self
    ) -> bool:

        pass

    @abc.abstractmethod
    def is_full(
        self: Self
    ) -> bool:

        pass

    @abc.abstractmethod
    def get(
        self: Self,
        block: bool = True,
        timeout: float | None = None
    ) -> QueueItemT:

        pass

    @abc.abstractmethod
    async def get_async(
        self: Self,
        block: bool = True,
        timeout: float | None = None
    ) -> QueueItemT:

        pass

    @abc.abstractmethod
    def get_batch(
        self: Self,
        num_items: int | None = None,
        max_num_items: int | None = None,
        all_items: bool = False,
        min_num_items: int | None = None,
        block: bool = True,
        timeout: float | None = None
    ) -> list[QueueItemT]:

        pass

    @abc.abstractmethod
    async def get_batch_async(
        self: Self,
        num_items: int | None = None,
        max_num_items: int | None = None,
        all_items: bool = False,
        min_num_items: int | None = None,
        block: bool = True,
        timeout: float | None = None
    ) -> list[QueueItemT]:

        pass

    @abc.abstractmethod
    def mark_key_finished(
        self: Self,
        key: Hashable
    ) -> None:

        pass

    @abc.abstractmethod
    def mark_item_finished(
        self: Self
    ) -> None:

        pass


class StageOutputQueueInterface(Generic[QueueItemT], metaclass=abc.ABCMeta):
    max_size: int

    def __init__(
        self: Self,
        queue_name: str,
        instance_name: str,
        max_size: int
    ) -> None:

        stage_name = get_stage_name_from_instance_name(instance_name)
        input_stage_name, _ = get_stage_names_from_queue_name(queue_name)
        assert stage_name == input_stage_name

        self.max_size = max_size

    @abc.abstractmethod
    def size(
        self: Self
    ) -> int:

        pass

    @abc.abstractmethod
    def remaining(
        self: Self
    ) -> int:

        pass

    @abc.abstractmethod
    def is_empty(
        self: Self
    ) -> bool:

        pass

    @abc.abstractmethod
    def is_full(
        self: Self
    ) -> bool:

        pass

    @abc.abstractmethod
    def put(
        self: Self,
        item: QueueItemT,
        block: bool = True,
        timeout: float | None = None,
        key: Hashable | None = None
    ) -> None:

        pass

    @abc.abstractmethod
    async def put_async(
        self: Self,
        item: QueueItemT,
        block: bool = True,
        timeout: float | None = None,
        key: Hashable | None = None
    ) -> None:

        pass

    @abc.abstractmethod
    def put_batch(
        self: Self,
        item: list[QueueItemT],
        block: bool = True,
        timeout: float | None = None,
        keys: list[Hashable] | None = None
    ) -> None:

        pass

    @abc.abstractmethod
    async def put_batch_async(
        self: Self,
        items: list[QueueItemT],
        block: bool = True,
        timeout: float | None = None,
        keys: list[Hashable] | None = None
    ) -> None:

        pass


class StageCommunicator(metaclass=abc.ABCMeta):
    _comm_ready: bool

    def __init__(
        self: Self
    ) -> None:

        self._comm_ready = False

    @classmethod
    @abc.abstractmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        pass

    def initialize(
        self: Self
    ) -> None:

        self._comm_ready = True
    
    @abc.abstractmethod
    def get_instance_name(
        self: Self
    ) -> str:

        pass
    
    @abc.abstractmethod
    def get_input_queue_interface(
        self: Self,
        input_stage_name: str,
        output_stage_name: str,
        index: int,
        item_type: type[QueueItemT],
        serializer: QueueItemSerializer[QueueItem]
    ) -> StageInputQueueInterface[QueueItemT]:

        pass

    @abc.abstractmethod
    def get_output_queue_interface(
        self: Self,
        input_stage_name: str,
        output_stage_name: str,
        index: int,
        item_type: type[QueueItemT],
        serializer: QueueItemSerializer[QueueItem]
    ) -> StageOutputQueueInterface[QueueItemT]:

        pass

    @abc.abstractmethod
    def signal_instance_ready(
        self: Self
    ) -> None:

        pass

    @abc.abstractmethod
    async def wait_all_instances_ready(
        self: Self
    ) -> None:

        pass

    @abc.abstractmethod
    def signal_global_stop(
        self: Self
    ) -> None:

        pass

    @abc.abstractmethod
    def should_stop(
        self: Self
    ) -> bool:

        pass


@final
class StageCommunicatorType(int, ReprEnum):
    RAY = 1
    RUST = 2


@final
@dataclass(kw_only=True, frozen=True)
class StageCommunicatorFactoryParams(BytesSerializable):
    communicator_type: StageCommunicatorType
    config_data: bytes

    def to_bytes(
        self: Self
    ) -> bytes:

        return msgpack.packb((self.communicator_type.value, self.config_data))

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        comm_type_val, config_data = msgpack.unpackb(buf)

        return cls(
            communicator_type=StageCommunicatorType(comm_type_val),
            config_data=config_data
        )


class StageCommunicatorFactory:
    @classmethod
    def _get_proxy_cls(
        cls: type[Self],
        stage_comm_type: StageCommunicatorType
    ) -> type[StageCommunicator]:

        match stage_comm_type:
            case StageCommunicatorType.RAY:
                from alto_lib.core.manager.ray_impl import RayStageCommunicator
                return RayStageCommunicator
            case _:
                raise ValueError()

    @classmethod
    def get_proxy_from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> StageCommunicator:

        params = StageCommunicatorFactoryParams.from_bytes(buf)

        return cls._get_proxy_cls(params.communicator_type).from_bytes(params.config_data)


@final
@dataclass(kw_only=True, frozen=True)
class _QueueParams:
    queue_name: str
    input_stage_name: str
    output_stage_name: str
    index: int
    item_type: type
    max_size: int


@final
@dataclass(kw_only=True, frozen=True)
class _StageParams:
    stage_name: str
    run_func: Callable[..., None] | Callable[..., Coroutine[Any, Any, None]]
    run_func_params: tuple[tuple[Any, ...], dict[str, Any]]


class StageManager(metaclass=abc.ABCMeta):
    _stage_names: set[str]
    _stage_name_to_instances: dict[str, list[InstancePlacementSpec]]
    _stage_name_to_params: dict[str, _StageParams]
    _queue_name_to_params: dict[str, _QueueParams]
    _num_queues_per_stage_pair: dict[tuple[str, str], int]

    _stages_complete: bool
    _stage_args_complete: bool
    _ready: bool

    def __init__(
        self: Self
    ) -> None:

        self._stage_name_to_instances = {}
        self._stage_name_to_params = {}
        self._queue_name_to_params = {}
        self._num_queues_per_stage_pair = {}

        self._stages_complete = False
        self._stage_args_complete = False
        self._ready = False

    @abc.abstractmethod
    def _get_proxy_factory_params(
        self: Self
    ) -> StageCommunicatorFactoryParams:

        pass

    def get_comm_params_data(
        self: Self
    ) -> bytes:

        return self._get_proxy_factory_params().to_bytes()

    def add_stages(
        self: Self,
        stage_names: list[str]
    ) -> None:

        assert not self._ready
        assert not self._stages_complete
        assert len(stage_names) == len(set(stage_names))

        self._stage_names = set(stage_names)
        self._stages_complete = True

    def add_queue(
        self: Self,
        *,
        input_stage_name: str,
        output_stage_name: str,
        item_type: type[QueueItemT],
        max_size: int = 0
    ) -> int:

        assert not self._ready
        assert self._stages_complete
        assert input_stage_name in self._stage_names
        assert output_stage_name in self._stage_names

        index = self._num_queues_per_stage_pair.get((input_stage_name, output_stage_name), 0)
        self._num_queues_per_stage_pair[(input_stage_name, output_stage_name)] = index + 1

        queue_name = get_queue_name(
            input_stage_name,
            output_stage_name,
            index
        )

        queue_params = _QueueParams(
            queue_name=queue_name,
            input_stage_name=input_stage_name,
            output_stage_name=output_stage_name,
            index=index,
            item_type=item_type,
            max_size=max_size
        )
        self._queue_name_to_params[queue_name] = queue_params

        return index

    def add_stage_args(
        self: Self,
        stage_name: str,
        run_func: Callable[StageFuncParamsT, None] | Callable[StageFuncParamsT, Coroutine[Any, Any, None]],
        *args: StageFuncParamsT.args,
        **kwargs: StageFuncParamsT.kwargs
    ) -> None:

        assert not self._ready
        assert stage_name not in self._stage_name_to_params
        assert stage_name in self._stage_names

        stage_params = _StageParams(
            stage_name=stage_name,
            run_func=run_func,
            run_func_params=(args, kwargs)
        )
        self._stage_name_to_params[stage_name] = stage_params

        if set(self._stage_name_to_params.keys()) == self._stage_names:
            self._stage_args_complete = True

    def add_instances(
        self: Self,
        *,
        stage_name: str,
        num_cpus: float,
        num_gpus: float,
        num_llm_gpus: float,
        num_instances: int = 1
    ) -> None:

        assert not self._ready
        assert stage_name in self._stage_names
        assert self._stage_args_complete

        instance_spec = InstancePlacementSpec(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            num_llm_gpus=num_llm_gpus
        )
        self._stage_name_to_instances.setdefault(stage_name, [])
        for _ in range(num_instances):
            self._stage_name_to_instances[stage_name].append(instance_spec)

    def get_placement_spec(
        self: Self
    ) -> PipelinePlacementSpec:

        assert self._stages_complete

        stages: dict[str, StagePlacementSpec] = {}
        for stage_name, instances in self._stage_name_to_instances.items():
            stages[stage_name] = StagePlacementSpec(instances=instances)

        placement_spec = PipelinePlacementSpec(
            stages=stages
        )

        return placement_spec

    def get_queues_spec(
        self: Self
    ) -> PipelineQueuesSpec:
        
        assert self._stages_complete

        queues: dict[str, QueueSpec] = {}
        for queue_name, queue_params in self._queue_name_to_params.items():
            queue_spec = QueueSpec(
                queue_name=queue_params.queue_name,
                input_stage_name=queue_params.input_stage_name,
                output_stage_name=queue_params.output_stage_name,
                index=queue_params.index,
                max_size=queue_params.max_size
            )
            queues[queue_name] = queue_spec

        queues_spec = PipelineQueuesSpec(
            queues=queues
        )

        return queues_spec

    def get_input_queues_per_stage(
        self: Self
    ) -> dict[str, list[QueueSpec]]:

        placement_spec = self.get_placement_spec()
        queues_spec = self.get_queues_spec()

        queues_per_stage: dict[str, list[QueueSpec]] = {}
        for stage_name in placement_spec.stages.keys():
            stage_queues: list[QueueSpec] = []
            for queue_spec in queues_spec.queues.values():
                if queue_spec.output_stage_name == stage_name:
                    stage_queues.append(queue_spec)
            queues_per_stage[stage_name] = stage_queues

        return queues_per_stage

    @abc.abstractmethod
    def initialize(
        self: Self
    ) -> None:

        self._ready = True

    @abc.abstractmethod
    def run_all_stages(
        self: Self
    ) -> None:

        pass
