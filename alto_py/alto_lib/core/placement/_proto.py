
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 02:07:12 2023

@author: Shahir
"""

import abc
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Self, final

from alto_lib.utils import JSONDictSerializable

DeviceId = int
StageName = str


class DeviceType(int, Enum):
    CPU = 0
    GPU = 1


@final
@dataclass(kw_only=True, frozen=True)
class DeviceSpec(JSONDictSerializable):
    id: DeviceId
    num_cpus: int
    num_gpus: int

    def __post_init__(
        self: Self
    ) -> None:

        assert all(n >= 0 for n in [self.num_cpus, self.num_gpus])

    def to_dict(
        self: Self
    ) -> dict[str, Any]:

        return asdict(self)

    @classmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, Any]
    ) -> Self:

        return cls(**data)


@final
@dataclass(kw_only=True, frozen=True)
class ClusterSpec(JSONDictSerializable):
    devices: dict[DeviceId, DeviceSpec]

    def __post_init__(
        self: Self
    ) -> None:

        assert all(k == v.id for k, v in self.devices.items())

    def to_dict(
        self: Self
    ) -> dict[str, Any]:

        return asdict(self)

    @classmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, dict[str, Any]]
    ) -> Self:

        out: dict[str, Any] = data.copy()
        for k, v in data['devices'].items():
            out[k] = DeviceSpec.from_dict(v)

        return cls(**out)


@final
@dataclass(kw_only=True, frozen=True)
class ComputeParams(JSONDictSerializable):
    num_cpus: int
    num_gpus: int
    cpu_memory_limit: int
    batch_size: int

    def __post_init__(
        self: Self
    ) -> None:

        assert all(n >= 0 for n in [
            self.num_cpus, self.num_gpus, self.cpu_memory_limit])

    def to_dict(
        self: Self
    ) -> dict[str, Any]:

        return asdict(self)

    @classmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, Any]
    ) -> Self:

        return cls(**data)


@final
@dataclass(kw_only=True, frozen=True)
class PipelineStage(JSONDictSerializable):
    name: StageName
    inputs: list[StageName]
    outputs: list[StageName]
    possible_configs: list[ComputeParams]

    def __post_init__(
        self: Self
    ) -> None:

        assert len(self.possible_configs) > 0
        assert len(set(self.inputs).intersection(self.outputs)) == 0

    def to_dict(
        self: Self
    ) -> dict[str, Any]:

        return asdict(self)

    @classmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, Any]
    ) -> Self:

        data = data.copy()
        data['possible_configs'] = [ComputeParams.from_dict(n) for n in data['possible_configs']]

        return cls(**data)


@final
@dataclass(kw_only=True, frozen=True)
class PipelineGraph(JSONDictSerializable):
    stages: dict[StageName, PipelineStage]

    def __post_init__(
        self: Self
    ) -> None:

        assert all(k == v.name for k, v in self.stages.items())

    def to_dict(
        self: Self
    ) -> dict[str, Any]:

        return asdict(self)

    @classmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, str | dict[str, Any]]
    ) -> Self:

        out: dict[str, Any] = data.copy()
        for k, v in data['stages'].items():  # type: ignore
            out[k] = PipelineStage.from_dict(v)

        return cls(**out)


@final
@dataclass(kw_only=True, frozen=True)
class InstancePlacementSpec(JSONDictSerializable):
    num_cpus: int
    num_gpus: int
    num_llm_gpus: int
    max_batch_size: int

    def __post_init__(
        self: Self
    ) -> None:

        assert all(n >= 0 for n in [self.num_cpus, self.num_gpus])

    def to_dict(
        self: Self
    ) -> dict[str, Any]:

        return asdict(self)

    @classmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, Any]
    ) -> Self:

        return cls(**data)


@final
@dataclass(kw_only=True, frozen=True)
class StagePlacementSpec(JSONDictSerializable):
    instances: list[InstancePlacementSpec]

    def to_dict(
        self: Self
    ) -> dict[str, Any]:

        return asdict(self)

    @classmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, Any]
    ) -> Self:

        data = data.copy()
        data['instances'] = [InstancePlacementSpec.from_dict(n) for n in data['instances']]

        return cls(**data)


@final
@dataclass(kw_only=True, frozen=True)
class PipelinePlacementSpec(JSONDictSerializable):
    stages: dict[StageName, StagePlacementSpec]

    def to_dict(
        self: Self
    ) -> dict[str, Any]:

        return asdict(self)

    @classmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, dict[str, Any]]
    ) -> Self:

        out: dict[str, Any] = data.copy()
        for k, v in data['placements'].items():
            out[k] = StagePlacementSpec.from_dict(v)

        return cls(**out)


class ProfilingResult(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_estimated_latency(
        self: Self,
        stage_id: StageName,
        compute_config: ComputeParams
    ) -> float:

        pass

    @abc.abstractmethod
    def get_estimated_throughput(
        self: Self,
        stage_id: StageName,
        compute_config: ComputeParams
    ) -> float:

        pass


class ComputePlacementPolicy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_placements(
        self: Self,
        cluster_spec: ClusterSpec,
        pipeline_graph: PipelineGraph,
        profiling_result: ProfilingResult
    ) -> PipelinePlacementSpec:

        pass


class DynamicBatchsizePolicy(metaclass=abc.ABCMeta):
    pipeline_graph: PipelineGraph
    compute_placements: PipelinePlacementSpec
    intial_profiling_result: ProfilingResult

    def __init__(
        self: Self,
        pipeline_graph: PipelineGraph,
        compute_placements: PipelinePlacementSpec,
        intial_profiling_result: ProfilingResult
    ) -> None:

        self.pipeline_graph = pipeline_graph
        self.compute_placements = compute_placements
        self.intial_profiling_result = intial_profiling_result

    @abc.abstractmethod
    def update_input_stats(
        self: Self,
        record_time: float,
        stage_id: StageName,
        instance_index: int,
        queue_size_per_input_stage: dict[StageName, int],
        num_new_consumed_per_input_stage: dict[StageName, int],
        num_new_produced_per_output_stage: dict[StageName, int]
    ) -> None:

        pass

    @abc.abstractmethod
    def compute_next_batch_size(
        self: Self,
        stage_id: StageName,
        instance_index: int
    ) -> int:

        pass
