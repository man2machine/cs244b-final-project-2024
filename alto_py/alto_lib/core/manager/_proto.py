# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 06:42:47 2024

@author: Shahir
"""

from dataclasses import dataclass, asdict
from typing import Any, Self, final

from alto_lib.utils import JSONDictSerializable


@final
@dataclass(kw_only=True, frozen=True)
class InstancePlacementSpec(JSONDictSerializable):
    num_cpus: float
    num_gpus: float
    num_llm_gpus: float

    def __post_init__(
        self: Self
    ) -> None:

        assert all(n >= 0 for n in [self.num_cpus, self.num_gpus, self.num_llm_gpus])

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

        instances = [InstancePlacementSpec.from_dict(n) for n in data['instances']]

        return cls(instances=instances)


@final
@dataclass(kw_only=True, frozen=True)
class PipelinePlacementSpec(JSONDictSerializable):
    stages: dict[str, StagePlacementSpec]

    def to_dict(
        self: Self
    ) -> dict[str, Any]:

        return asdict(self)

    @classmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, Any]
    ) -> Self:

        stages = {k: StagePlacementSpec.from_dict(v) for k, v in data['stages'].items()}

        return cls(stages=stages)


@final
@dataclass(kw_only=True, frozen=True)
class QueueSpec(JSONDictSerializable):
    queue_name: str
    input_stage_name: str
    output_stage_name: str
    index: int
    max_size: int

    def __post_init__(
        self: Self
    ) -> None:

        assert self.index >= 0
        assert self.max_size >= 0

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
class PipelineQueuesSpec(JSONDictSerializable):
    queues: dict[str, QueueSpec]

    def to_dict(
        self: Self
    ) -> dict[str, Any]:

        return asdict(self)

    @classmethod
    def from_dict(
        cls: type[Self],
        data: dict[str, Any]
    ) -> Self:

        queues = {k: QueueSpec.from_dict(v) for k, v in data['queues'].items()}

        return cls(queues=queues)
