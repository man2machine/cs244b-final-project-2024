# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 06:42:47 2024

@author: Shahir
"""

from typing import Self, final

from alto_lib.utils import Config


@final
class InstancePlacementSpec(Config):
    num_cpus: float
    num_gpus: float
    num_llm_gpus: float

    def __post_init__(
        self: Self
    ) -> None:

        assert all(n >= 0 for n in [self.num_cpus, self.num_gpus, self.num_llm_gpus])


@final
class StagePlacementSpec(Config):
    instances: list[InstancePlacementSpec]


@final
class PipelinePlacementSpec(Config):
    stages: dict[str, StagePlacementSpec]


@final
class QueueSpec(Config):
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


@final
class PipelineQueuesSpec(Config):
    queues: dict[str, QueueSpec]
