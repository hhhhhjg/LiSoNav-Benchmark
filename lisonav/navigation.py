"""Adapted from Eku127/habitat-data-collector (MIT)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import DefaultDict, Dict, List

import numpy as np


def compute_spl(success: bool, gt_distance: float, explored_distance: float) -> float:
    if not success:
        return 0.0
    denom = max(gt_distance, explored_distance)
    if denom <= 0:
        return 0.0
    value = gt_distance / denom
    if math.isnan(value):
        return 0.0
    return float(value)


@dataclass
class SubtaskNavigationMetrics:
    subtask_id: str
    goal_type: str
    success_by_distance: float
    spl_by_distance: float
    collision_ratio: float
    target_not_on_navmesh_ratio: float
    n_filtered_frames: int
    n_total_frames: int
    explored_distance: float
    gt_explore_distance: float


@dataclass
class SubtaskNavigationMetricsTracker:
    collision_steps: int = 0
    executed_steps: int = 0
    decision_steps: int = 0
    target_not_on_navmesh_steps: int = 0

    def record_step(self, collided) -> None:
        self.decision_steps += 1
        if isinstance(collided, dict):
            self.executed_steps += int(collided.get("total_steps", 0))
            self.collision_steps += int(collided.get("collision_steps", 0))
            self.target_not_on_navmesh_steps += int(
                bool(collided.get("target_not_on_navmesh", False))
            )
            return
        self.executed_steps += 1
        self.collision_steps += int(bool(collided))

    @property
    def collision_ratio(self) -> float:
        if self.executed_steps <= 0:
            return 0.0
        return self.collision_steps / self.executed_steps

    @property
    def target_not_on_navmesh_ratio(self) -> float:
        if self.decision_steps <= 0:
            return 0.0
        return self.target_not_on_navmesh_steps / self.decision_steps

    def finalize(
        self,
        *,
        subtask_id: str,
        goal_type: str,
        success_by_distance: bool,
        gt_explore_distance: float,
        explored_distance: float,
        n_filtered_frames: int,
        n_total_frames: int,
    ) -> SubtaskNavigationMetrics:
        return SubtaskNavigationMetrics(
            subtask_id=subtask_id,
            goal_type=goal_type,
            success_by_distance=1.0 if success_by_distance else 0.0,
            spl_by_distance=compute_spl(
                success=success_by_distance,
                gt_distance=gt_explore_distance,
                explored_distance=explored_distance,
            ),
            collision_ratio=self.collision_ratio,
            target_not_on_navmesh_ratio=self.target_not_on_navmesh_ratio,
            n_filtered_frames=int(n_filtered_frames),
            n_total_frames=int(n_total_frames),
            explored_distance=float(explored_distance),
            gt_explore_distance=float(gt_explore_distance),
        )


@dataclass
class NavigationMetricsStore:
    success_by_distance: Dict[str, float] = field(default_factory=dict)
    spl_by_distance: Dict[str, float] = field(default_factory=dict)
    collision_ratio_by_subtask: Dict[str, float] = field(default_factory=dict)
    target_not_on_navmesh_ratio_by_subtask: Dict[str, float] = field(default_factory=dict)
    n_filtered_frames_by_subtask: Dict[str, int] = field(default_factory=dict)
    n_total_frames_by_subtask: Dict[str, int] = field(default_factory=dict)
    success_by_task: DefaultDict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    spl_by_task: DefaultDict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    collision_ratio_by_task: DefaultDict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    target_not_on_navmesh_ratio_by_task: DefaultDict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def record(self, metrics: SubtaskNavigationMetrics) -> None:
        self.success_by_distance[metrics.subtask_id] = metrics.success_by_distance
        self.spl_by_distance[metrics.subtask_id] = metrics.spl_by_distance
        self.collision_ratio_by_subtask[metrics.subtask_id] = metrics.collision_ratio
        self.target_not_on_navmesh_ratio_by_subtask[metrics.subtask_id] = (
            metrics.target_not_on_navmesh_ratio
        )
        self.n_filtered_frames_by_subtask[metrics.subtask_id] = metrics.n_filtered_frames
        self.n_total_frames_by_subtask[metrics.subtask_id] = metrics.n_total_frames
        self.success_by_task[metrics.goal_type].append(metrics.success_by_distance)
        self.spl_by_task[metrics.goal_type].append(metrics.spl_by_distance)
        self.collision_ratio_by_task[metrics.goal_type].append(metrics.collision_ratio)
        self.target_not_on_navmesh_ratio_by_task[metrics.goal_type].append(
            metrics.target_not_on_navmesh_ratio
        )

    def mean_success_by_distance(self) -> float:
        if len(self.success_by_distance) == 0:
            return 0.0
        return float(np.mean(list(self.success_by_distance.values())))

    def mean_spl_by_distance(self) -> float:
        if len(self.spl_by_distance) == 0:
            return 0.0
        return float(np.mean(list(self.spl_by_distance.values())))

    def mean_collision_ratio(self) -> float:
        if len(self.collision_ratio_by_subtask) == 0:
            return 0.0
        return float(np.mean(list(self.collision_ratio_by_subtask.values())))

    def mean_target_not_on_navmesh_ratio(self) -> float:
        if len(self.target_not_on_navmesh_ratio_by_subtask) == 0:
            return 0.0
        return float(np.mean(list(self.target_not_on_navmesh_ratio_by_subtask.values())))
