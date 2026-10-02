"""Adapted from Eku127/habitat-data-collector (MIT)."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


DEFAULT_MAX_HORIZONTAL_SNAP_M = 1.5
DEFAULT_VERTICAL_PROBE_DEPTH_M = 2.0
DEFAULT_VERTICAL_PROBE_STEP_M = 0.05


def _to_vec3(value) -> Optional[np.ndarray]:
    try:
        if isinstance(value, np.ndarray):
            flat = value.reshape(-1)
            if flat.size < 3:
                return None
            return np.asarray([float(flat[0]), float(flat[1]), float(flat[2])], dtype=np.float32)
        if isinstance(value, (list, tuple)):
            if len(value) < 3:
                return None
            return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
    except Exception:
        return None
    return None


def parse_semantic_id_from_object_id(object_id_value) -> Optional[int]:
    """Parse semantic id from object_id like '057_racquetball_1039'."""
    if object_id_value is None:
        return None
    text = str(object_id_value).strip()
    if len(text) == 0:
        return None
    match = re.search(r"_(\d+)$", text)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def load_episode_semantic_positions(scene_config_path: Path) -> Dict[int, List[np.ndarray]]:
    """Load positions grouped by semantic id from one episode scene config."""
    with open(scene_config_path, "r", encoding="utf-8") as fp:
        scene_cfg = json.load(fp)

    grouped: Dict[int, List[np.ndarray]] = defaultdict(list)
    for obj in scene_cfg.get("objects", []):
        try:
            semantic_id = int(obj.get("semantic_id"))
        except Exception:
            continue
        pos = _to_vec3(obj.get("translation"))
        if pos is None:
            continue
        grouped[semantic_id].append(pos)
    return dict(grouped)


def resolve_subtask_target_positions(
    goal_entries: Iterable[dict], semantic_positions: Dict[int, List[np.ndarray]]
) -> List[np.ndarray]:
    """Resolve subtask target positions from current episode scene config.

    Matching strategy:
    1) Parse semantic ids from goal_entry.object_id for this subtask.
    2) For each semantic id, include all corresponding scene-object positions.
    3) If a semantic id has no scene candidates, fallback to goal_entry.position.
    4) If no semantic id can be parsed, fallback to any valid goal_entry.position.

    This ensures distance-based success uses the nearest distance to all
    valid target instances in the subtask.
    """
    goal_entries_list = [x for x in goal_entries if isinstance(x, dict)]
    if len(goal_entries_list) == 0:
        return []

    # Keep insertion order while deduplicating by semantic id.
    semantic_ids: List[int] = []
    for goal in goal_entries_list:
        semantic_id = parse_semantic_id_from_object_id(goal.get("object_id"))
        if semantic_id is None or semantic_id in semantic_ids:
            continue
        semantic_ids.append(int(semantic_id))

    resolved: List[np.ndarray] = []
    seen_keys = set()

    def _add_pos(pos: Optional[np.ndarray]) -> None:
        if pos is None:
            return
        key = (round(float(pos[0]), 6), round(float(pos[1]), 6), round(float(pos[2]), 6))
        if key in seen_keys:
            return
        seen_keys.add(key)
        resolved.append(np.asarray(pos, dtype=np.float32))

    # Primary path: include all scene candidates for each semantic id.
    for semantic_id in semantic_ids:
        candidates = semantic_positions.get(int(semantic_id), [])
        for candidate in candidates:
            _add_pos(_to_vec3(candidate))

    # Fallback for semantic ids not found in scene_config.
    for goal in goal_entries_list:
        semantic_id = parse_semantic_id_from_object_id(goal.get("object_id"))
        if semantic_id is None:
            continue
        if len(semantic_positions.get(int(semantic_id), [])) > 0:
            continue
        _add_pos(_to_vec3(goal.get("position")))

    # Last fallback: no semantic ids/candidates, use any valid goal positions.
    if len(resolved) == 0:
        for goal in goal_entries_list:
            _add_pos(_to_vec3(goal.get("position")))

    return resolved


def calc_min_horizontal_distance(agent_position, target_positions: Iterable[np.ndarray]) -> float:
    """Compute min XZ-plane L2 distance from agent to target positions."""
    agent = _to_vec3(agent_position)
    if agent is None:
        return float("inf")

    min_dist = float("inf")
    for target in target_positions:
        pos = _to_vec3(target)
        if pos is None:
            continue
        dist = float(np.linalg.norm(pos[[0, 2]] - agent[[0, 2]]))
        if dist < min_dist:
            min_dist = dist
    return float(min_dist)


def calc_min_geodesic_distance_to_targets(
    pathfinder,
    agent_position,
    target_positions: Iterable[np.ndarray],
    *,
    max_horizontal_snap_m: float = DEFAULT_MAX_HORIZONTAL_SNAP_M,
    vertical_probe_depth_m: float = DEFAULT_VERTICAL_PROBE_DEPTH_M,
    vertical_probe_step_m: float = DEFAULT_VERTICAL_PROBE_STEP_M,
    diagnostics: Optional[dict] = None,
) -> float:
    """Measure the shortest path to object anchors on the agent's NavMesh island.

    Object translations are commonly above the floor on a support surface. Probe
    downward before snapping and reject projections farther than the configured
    horizontal limit.
    """
    import habitat_sim

    details = diagnostics if diagnostics is not None else {}
    details.clear()
    details.update(
        {
            "failure_reason": None,
            "start_island": None,
            "target_count": 0,
            "targets_with_valid_snap": 0,
            "targets_with_path": 0,
            "selected_target_index": None,
            "selected_horizontal_snap_m": None,
            "selected_vertical_snap_m": None,
            "selected_snapped_target": None,
        }
    )

    start = _to_vec3(agent_position)
    targets = [point for point in (_to_vec3(value) for value in target_positions) if point is not None]
    details["target_count"] = len(targets)
    if start is None or not targets:
        details["failure_reason"] = "invalid_start_or_no_targets"
        return float("inf")

    try:
        snapped_start = np.asarray(pathfinder.snap_point(start), dtype=np.float32)
        if snapped_start.size < 3 or not np.all(np.isfinite(snapped_start[:3])):
            details["failure_reason"] = "invalid_start_snap"
            return float("inf")
        snapped_start = snapped_start[:3]
        start_island = int(pathfinder.get_island(snapped_start))
    except Exception as exc:
        details["failure_reason"] = f"start_snap_error:{type(exc).__name__}"
        return float("inf")

    details["start_island"] = start_island
    details["snapped_start"] = snapped_start.tolist()
    offsets = np.arange(
        0.0,
        float(vertical_probe_depth_m) + float(vertical_probe_step_m) / 2.0,
        float(vertical_probe_step_m),
    )
    best_distance = float("inf")

    for target_index, target in enumerate(targets):
        probe_heights = {round(float(snapped_start[1]), 4)}
        probe_heights.update(round(float(target[1] - offset), 4) for offset in offsets)
        best_snap = None
        for height in sorted(probe_heights, reverse=True):
            query = target.copy()
            query[1] = float(height)
            try:
                snapped = np.asarray(
                    pathfinder.snap_point(query, island_index=start_island), dtype=np.float32
                )
            except Exception:
                continue
            if snapped.size < 3 or not np.all(np.isfinite(snapped[:3])):
                continue
            snapped = snapped[:3]
            horizontal_gap = float(np.linalg.norm((snapped - target)[[0, 2]]))
            if horizontal_gap > float(max_horizontal_snap_m) + 1e-6:
                continue
            vertical_gap = float(abs(snapped[1] - target[1]))
            candidate = (horizontal_gap, vertical_gap, snapped)
            if best_snap is None or candidate[:2] < best_snap[:2]:
                best_snap = candidate

        if best_snap is None:
            continue
        details["targets_with_valid_snap"] += 1
        path = habitat_sim.ShortestPath()
        path.requested_start = snapped_start
        path.requested_end = best_snap[2]
        try:
            found = bool(pathfinder.find_path(path))
            distance = float(path.geodesic_distance)
        except Exception:
            continue
        if not found or not np.isfinite(distance):
            continue
        details["targets_with_path"] += 1
        if distance < best_distance:
            best_distance = distance
            details["selected_target_index"] = target_index
            details["selected_horizontal_snap_m"] = float(best_snap[0])
            details["selected_vertical_snap_m"] = float(best_snap[1])
            details["selected_snapped_target"] = best_snap[2].tolist()

    if np.isfinite(best_distance):
        return float(best_distance)
    details["failure_reason"] = (
        "no_valid_target_snap"
        if details["targets_with_valid_snap"] == 0
        else "no_path_to_valid_target_snap"
    )
    return float("inf")
