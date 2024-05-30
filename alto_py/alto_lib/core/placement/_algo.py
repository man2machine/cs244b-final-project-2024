# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 02:42:27 2023

@author: Shahir
"""

from dataclasses import dataclass
from typing import Self, final, cast

import numpy as np
import numpy.typing as npt

import cvxpy as cp

from alto_lib.core.placement._proto import (
    ClusterSpec, ComputeParams, PipelineGraph, ComputePlacementPolicy, DynamicBatchsizePolicy, ProfilingResult, StageName,
    StagePlacementSpec, InstancePlacementSpec, PipelinePlacementSpec
)


@dataclass
class _ILPVariableInfo:
    stage_id: StageName
    config_index: int


@final
class ILPPlacementPolicy(ComputePlacementPolicy):
    def compute_placements(
        self: Self,
        cluster_spec: ClusterSpec,
        pipeline_graph: PipelineGraph,
        profiling_result: ProfilingResult
    ) -> PipelinePlacementSpec:

        assert len(cluster_spec.devices) == 1, "Currently supports only one device"
        device = list(cluster_spec.devices.values())[0]

        b = np.array([device.num_cpus, device.num_gpus, device.cpu_memory_capacity], dtype=np.float64)

        x_infos: list[_ILPVariableInfo] = []
        for stage_id, stage in pipeline_graph.stages.items():
            for i in range(len(stage.possible_configs)):
                var_info = _ILPVariableInfo(
                    stage_id=stage_id,
                    config_index=i
                )
                x_infos.append(var_info)

        A = np.zeros((b.shape[0], len(x_infos)), dtype=np.float64)
        Z = np.zeros((len(pipeline_graph.stages), len(x_infos)), dtype=np.float64)
        c = np.zeros(len(x_infos), dtype=np.float64)
        x_min = np.zeros(len(x_infos), dtype=np.float64)
        stages_min = np.ones(len(pipeline_graph.stages), dtype=np.float64)

        stage_ids = list(pipeline_graph.stages.keys())
        for i, var_info in enumerate(x_infos):
            stage_index = stage_ids.index(var_info.stage_id)
            compute_config = pipeline_graph.stages[var_info.stage_id].possible_configs[var_info.config_index]
            throughput = profiling_result.get_estimated_throughput(
                stage_id=var_info.stage_id,
                compute_config=compute_config
            )

            A[0, i] = compute_config.num_cpus
            A[1, i] = compute_config.num_gpus
            A[2, i] = compute_config.cpu_memory_limit
            Z[stage_index, i] = 1
            c[i] = throughput

        x = cp.Variable(len(x_infos), integer=True)
        prob = cp.Problem(cp.Maximize(c.T @ x), [A @ x <= b, x >= x_min, Z @ x >= stages_min])
        prob.solve()

        x_sol = cast(npt.NDArray[np.int64], x.value)

        results: dict[StageName, StagePlacementSpec] = {}

        for stage_id in pipeline_graph.stages.keys():
            res = StagePlacementSpec(
                stage_id=stage_id,
                instances=[],
                device_id=0
            )
            results[stage_id] = res

        curr_cpu_index = 0
        curr_gpu_index = 0
        for i in range(len(x_infos)):
            var_info = x_infos[i]
            sol_value = int(round(x_sol[i]))

            compute_config = pipeline_graph.stages[var_info.stage_id].possible_configs[var_info.config_index]

            for _ in range(sol_value):
                results[var_info.stage_id].instances.append(
                    InstancePlacementSpec(
                        compute_placement=InstancePlacementSpec(
                            num_cpus=compute_config.num_cpus,
                            num_gpus=compute_config.num_gpus,
                            cpu_memory_limit=compute_config.cpu_memory_limit,
                            cpu_indices=list(range(curr_cpu_index, curr_cpu_index + compute_config.num_cpus)),
                            gpu_indices=list(range(curr_gpu_index, curr_gpu_index + compute_config.num_gpus)),
                            max_batch_size=compute_config.batch_size
                        )
                    )
                )

                curr_cpu_index = compute_config.num_cpus
                curr_gpu_index += compute_config.num_gpus

        return PipelinePlacementSpec(stages=results)


@final
class BasicBatchsizePolicy(DynamicBatchsizePolicy):
    time_window_sec: float

    _logs_per_stage: dict[tuple[StageName, int], list[tuple[float, int]]]

    def __init__(
        self: Self,
        pipeline_graph: PipelineGraph,
        compute_placements: PipelinePlacementSpec,
        intial_profiling_result: ProfilingResult,
        time_window_sec: float = 10
    ) -> None:

        super().__init__(pipeline_graph, compute_placements, intial_profiling_result)

        self.time_window_sec = time_window_sec
        self._logs_per_stage = {}

    def update_input_stats(
        self: Self,
        record_time: float,
        stage_id: StageName,
        instance_index: int,
        queue_size_per_input_stage: dict[StageName, int],
        num_new_consumed_per_input_stage: dict[StageName, int],
        num_new_produced_per_output_stage: dict[StageName, int]
    ) -> None:

        num_new_produced = min(num_new_produced_per_output_stage.values())
        key = (stage_id, instance_index)
        self._logs_per_stage.setdefault(key, [])
        self._logs_per_stage[key].append((record_time, num_new_produced))

    def compute_next_batch_size(
        self: Self,
        stage_id: StageName,
        instance_index: int
    ) -> int:

        default_placement = self.compute_placements.stages[stage_id].instances[instance_index]
        max_batch_size = default_placement.max_batch_size

        key = (stage_id, instance_index)
        total_time = 0.0
        total_produced = 0
        i = len(self._logs_per_stage[key]) - 2

        if i < 0:
            return max_batch_size

        while True:
            entry1 = self._logs_per_stage[key][i]
            entry2 = self._logs_per_stage[key][i + 1]
            delta_time = entry2[0] - entry1[0]
            delta_produced = entry2[1] - entry1[1]

            total_time += delta_time
            total_produced += delta_produced

            if total_time >= self.time_window_sec:
                break

            i -= 1
            if i < 0:
                break

        estimated_current_throughput = total_produced / total_time

        batch_size = max_batch_size
        while True:
            compute_config = ComputeParams(
                num_cpus=default_placement.num_cpus,
                num_gpus=default_placement.num_gpus,
                cpu_memory_limit=default_placement.cpu_memory_limit,
                batch_size=batch_size
            )
            current_throughput = self.intial_profiling_result.get_estimated_throughput(
                stage_id=stage_id,
                compute_config=compute_config
            )

            if current_throughput < estimated_current_throughput:
                break

            batch_size //= 2

        return batch_size


if __name__ == '__main__':
    import functools

    from alto_lib.core.placement._proto import DeviceSpec, ComputeParams, PipelineStage

    rng = np.random.default_rng()

    @final
    class DummyProfilingResult(ProfilingResult):
        _dummy_scales: dict[StageName, float]

        def __init__(
            self: Self
        ) -> None:

            self._rng = np.random.default_rng()
            self._dummy_scales = {}

        @functools.cache
        def get_estimated_latency(
            self: Self,
            stage_id: StageName,
            compute_config: ComputeParams
        ) -> float:

            return 0

        @functools.cache
        def get_estimated_throughput(
            self: Self,
            stage_id: StageName,
            compute_config: ComputeParams
        ) -> float:

            if stage_id not in self._dummy_scales:
                self._dummy_scales[stage_id] = self._rng.uniform()
            scale = self._dummy_scales[stage_id]

            return scale * compute_config.num_cpus * compute_config.num_gpus * compute_config.batch_size

    device_spec = DeviceSpec(
        id=0,
        num_cpus=32,
        num_gpus=16,
        cpu_memory_capacity=512
    )

    cluster_spec = ClusterSpec(
        devices={0: device_spec}
    )

    compue_configs: list[ComputeParams] = []

    for i in range(1, 5):
        num_cpus = i * 8
        for j in range(1, 5):
            batch_size = 2 ** j
            memory = num_cpus * batch_size * 4
            config = ComputeParams(
                num_cpus=num_cpus,
                num_gpus=1,
                cpu_memory_limit=memory,
                batch_size=batch_size
            )
            compue_configs.append(config)

    num_stages = 4
    stages: list[PipelineStage] = []

    for i in range(num_stages):
        choices = rng.choice(num_stages, 2, replace=False)
        stage = PipelineStage(
            name="stage_{}".format(i),
            inputs=["stage_{}".format(choices[0])],
            outputs=["stage_{}".format(choices[1])],
            possible_configs=compue_configs
        )
        stages.append(stage)

    pipeline_graph = PipelineGraph(
        name="dummy",
        stages={v.name: v for v in stages}
    )

    profiling_result = DummyProfilingResult()

    placement_policy = ILPPlacementPolicy()
    placements = placement_policy.compute_placements(
        cluster_spec=cluster_spec,
        pipeline_graph=pipeline_graph,
        profiling_result=profiling_result
    )

    print(placements)
