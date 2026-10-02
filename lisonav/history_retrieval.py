"""Adapted from Eku127/habitat-data-collector (MIT)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


def parse_semantic_id_from_object_id(object_id_value) -> Optional[int]:
    if object_id_value is None:
        return None
    text = str(object_id_value).strip()
    if not text:
        return None
    match = re.search(r"_(\d+)$", text)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def collect_goal_semantic_ids(goal_entries: Iterable[dict[str, Any]]) -> list[int]:
    semantic_ids: list[int] = []
    seen: set[int] = set()
    for goal in goal_entries:
        if not isinstance(goal, dict):
            continue
        semantic_id = parse_semantic_id_from_object_id(goal.get("object_id"))
        if semantic_id is None or semantic_id in seen:
            continue
        seen.add(int(semantic_id))
        semantic_ids.append(int(semantic_id))
    return semantic_ids


def collect_goal_categories(goal_entries: Iterable[dict[str, Any]]) -> list[str]:
    categories: list[str] = []
    seen: set[str] = set()
    for goal in goal_entries:
        if not isinstance(goal, dict):
            continue
        category = str(goal.get("object_category", "")).strip()
        if not category or category in seen:
            continue
        seen.add(category)
        categories.append(category)
    return categories


def collect_episode_category_semantic_ids(
    scene_config_path: Path,
    categories: Iterable[str],
) -> list[int]:
    category_set = {str(value).strip() for value in categories if str(value).strip()}
    if not category_set:
        return []
    try:
        with open(scene_config_path, "r", encoding="utf-8") as fp:
            scene_cfg = json.load(fp)
    except Exception:
        return []

    id_handle_mapping = scene_cfg.get("id_handle_mapping", {})
    if not isinstance(id_handle_mapping, dict):
        id_handle_mapping = {}
    object_semantic_ids: list[int] = []
    seen_objects: set[int] = set()
    for obj in scene_cfg.get("objects", []):
        if not isinstance(obj, dict):
            continue
        try:
            semantic_id = int(obj.get("semantic_id"))
        except Exception:
            continue
        if semantic_id in seen_objects:
            continue
        seen_objects.add(semantic_id)
        object_semantic_ids.append(semantic_id)

    matched: list[int] = []
    seen: set[int] = set()
    for semantic_id in object_semantic_ids:
        handle = str(id_handle_mapping.get(str(int(semantic_id)), "")).strip()
        if handle not in category_set:
            continue
        if semantic_id in seen:
            continue
        seen.add(semantic_id)
        matched.append(semantic_id)
    return matched


def resolve_episode_goal_semantic_ids(scene_config_path: Path, goal_entries: Iterable[dict[str, Any]]) -> list[int]:
    category_ids = collect_episode_category_semantic_ids(
        scene_config_path,
        collect_goal_categories(goal_entries),
    )
    if category_ids:
        return category_ids
    return collect_goal_semantic_ids(goal_entries)


def collect_episode_semantic_positions(
    scene_config_path: Path,
    semantic_ids: Iterable[int],
) -> dict[int, list[list[float]]]:
    target_ids: set[int] = set()
    for value in semantic_ids:
        try:
            target_ids.add(int(value))
        except Exception:
            continue
    if not target_ids:
        return {}
    try:
        with open(scene_config_path, "r", encoding="utf-8") as fp:
            scene_cfg = json.load(fp)
    except Exception:
        return {}

    grouped: dict[int, list[list[float]]] = {}
    for obj in scene_cfg.get("objects", []):
        if not isinstance(obj, dict):
            continue
        try:
            semantic_id = int(obj.get("semantic_id"))
        except Exception:
            continue
        if semantic_id not in target_ids:
            continue
        pos = _safe_position(obj.get("translation", []))
        grouped.setdefault(semantic_id, []).append(pos)
    return grouped


def collect_episode_semantic_instances(
    scene_config_path: Path,
    semantic_ids: Iterable[int],
) -> dict[int, list[dict[str, Any]]]:
    target_ids: set[int] = set()
    for value in semantic_ids:
        try:
            target_ids.add(int(value))
        except Exception:
            continue
    if not target_ids:
        return {}
    try:
        with open(scene_config_path, "r", encoding="utf-8") as fp:
            scene_cfg = json.load(fp)
    except Exception:
        return {}

    grouped: dict[int, list[dict[str, Any]]] = {}
    for object_index, obj in enumerate(scene_cfg.get("objects", [])):
        if not isinstance(obj, dict):
            continue
        try:
            semantic_id = int(obj.get("semantic_id"))
        except Exception:
            continue
        if semantic_id not in target_ids:
            continue
        grouped.setdefault(semantic_id, []).append(
            {
                "object_instance_id": obj.get("object_id", object_index),
                "object_index": int(object_index),
                "position": _safe_position(obj.get("translation", [])),
            }
        )
    return grouped


def collect_episode_goal_semantic_ids(episode_set) -> list[int]:
    semantic_ids: list[int] = []
    seen: set[int] = set()
    for episode in getattr(episode_set, "episodes", []):
        for subtask in getattr(episode, "subtasks", []):
            episode_ids = resolve_episode_goal_semantic_ids(
                Path(getattr(episode, "scene_config_path", "")),
                getattr(subtask, "goal_entries", []),
            )
            for semantic_id in episode_ids:
                if semantic_id in seen:
                    continue
                seen.add(int(semantic_id))
                semantic_ids.append(int(semantic_id))
    return semantic_ids


def new_gt_history_retrieval_state(tracked_semantic_ids: Optional[Iterable[int]] = None) -> dict[str, Any]:
    return {
        "tracked_semantic_ids": [int(v) for v in (tracked_semantic_ids or [])],
        "observations_by_semantic_id": {},
        "retrieval_by_subtask": {},
    }


def load_gt_history_retrieval_state(
    path: Path,
    *,
    tracked_semantic_ids: Optional[Iterable[int]] = None,
) -> dict[str, Any]:
    if not path.exists():
        return new_gt_history_retrieval_state(tracked_semantic_ids=tracked_semantic_ids)

    with open(path, "r", encoding="utf-8") as fp:
        loaded = json.load(fp)
    state = new_gt_history_retrieval_state(tracked_semantic_ids=tracked_semantic_ids)
    state["tracked_semantic_ids"] = [
        int(v) for v in loaded.get("tracked_semantic_ids", state["tracked_semantic_ids"])
    ]
    obs_loaded = loaded.get("observations_by_semantic_id", {})
    if isinstance(obs_loaded, dict):
        state["observations_by_semantic_id"] = {
            str(key): list(value) if isinstance(value, list) else []
            for key, value in obs_loaded.items()
        }
    retrieval_loaded = loaded.get("retrieval_by_subtask", {})
    if isinstance(retrieval_loaded, dict):
        state["retrieval_by_subtask"] = dict(retrieval_loaded)
    return state


def save_gt_history_retrieval_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(state, fp, indent=2, ensure_ascii=False)


def load_gt_history_retrieval_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        loaded = json.load(fp)
    return dict(loaded) if isinstance(loaded, dict) else {}


def drop_gt_history_retrieval_subtasks(results_by_subtask: dict[str, Any], subtask_ids: Iterable[str]) -> None:
    for subtask_id in set(str(v) for v in subtask_ids):
        results_by_subtask.pop(subtask_id, None)


def _safe_position(position: np.ndarray | list[float] | tuple[float, ...]) -> list[float]:
    pos = np.asarray(position, dtype=np.float32).reshape(-1)
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _center_crop_bounds(height: int, width: int, fraction: float) -> tuple[int, int, int, int]:
    frac = min(max(float(fraction), 0.0), 1.0)
    crop_h = max(1, int(round(height * frac)))
    crop_w = max(1, int(round(width * frac)))
    y0 = max(0, (height - crop_h) // 2)
    x0 = max(0, (width - crop_w) // 2)
    y1 = min(height, y0 + crop_h)
    x1 = min(width, x0 + crop_w)
    return y0, y1, x0, x1


def _match_visible_semantic_instances(
    *,
    mask: np.ndarray,
    depth_obs: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    candidates: list[dict[str, Any]],
    total_pixels: int,
    center_bounds: tuple[int, int, int, int],
    pixel_ratio_threshold: float,
    min_center_fraction: float,
    max_distance_m: float,
) -> list[dict[str, Any]]:
    depth = np.asarray(depth_obs, dtype=np.float32)
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float32)
    pose = np.asarray(camera_pose, dtype=np.float32)
    if depth.shape != mask.shape or intrinsics.shape[0] < 3 or intrinsics.shape[1] < 3 or pose.shape != (4, 4):
        return []

    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return []

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        return []
    forward = depth[ys, xs]
    image_height = int(depth.shape[0])
    camera_points = np.stack(
        [
            (xs.astype(np.float32) - cx) * forward / fx,
            ((image_height - 1 - ys).astype(np.float32) - cy) * forward / fy,
            -forward,
        ],
        axis=1,
    )
    homogeneous = np.concatenate(
        [camera_points, np.ones((len(camera_points), 1), dtype=np.float32)],
        axis=1,
    )
    world_points = (pose @ homogeneous.T).T[:, :3]
    candidate_positions = np.asarray([item["position"] for item in candidates], dtype=np.float32)
    distances = np.linalg.norm(world_points[:, None, :] - candidate_positions[None, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)
    y0, y1, x0, x1 = center_bounds
    in_center = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)

    matches: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        assigned = assignments == int(candidate_index)
        assigned_count = int(np.count_nonzero(assigned))
        if assigned_count <= 0:
            continue
        pixel_ratio = float(assigned_count) / float(total_pixels)
        if pixel_ratio < float(pixel_ratio_threshold):
            continue
        center_pixel_count = int(np.count_nonzero(assigned & in_center))
        center_fraction = float(center_pixel_count) / float(assigned_count)
        if center_fraction < float(min_center_fraction):
            continue
        assigned_distances = distances[assigned, candidate_index]
        median_distance = float(np.median(assigned_distances))
        if not np.isfinite(median_distance) or median_distance > float(max_distance_m):
            continue
        matches.append(
            {
                **candidate,
                "pixel_count": int(assigned_count),
                "pixel_ratio": float(pixel_ratio),
                "center_pixel_count": int(center_pixel_count),
                "center_fraction": float(center_fraction),
                "instance_match_median_distance_m": float(median_distance),
            }
        )
    return matches


def record_gt_target_observations(
    state: dict[str, Any],
    *,
    semantic_obs: np.ndarray,
    position: np.ndarray,
    semantic_positions_by_id: Optional[dict[int, list[np.ndarray | list[float] | tuple[float, ...]]]] = None,
    semantic_instances_by_id: Optional[dict[int, list[dict[str, Any]]]] = None,
    depth_obs: Optional[np.ndarray] = None,
    camera_intrinsics: Optional[np.ndarray] = None,
    camera_pose: Optional[np.ndarray] = None,
    angle_rad: float,
    subtask_id: str,
    subtask_order: int,
    episode_index: int,
    event: str,
    pixel_ratio_threshold: float = 2.5e-4,
    center_region_fraction: float = 0.8,
    min_center_fraction: float = 0.5,
    instance_match_max_distance_m: float = 1.0,
) -> list[dict[str, Any]]:
    if semantic_obs is None:
        return []
    tracked = {
        int(v) for v in state.get("tracked_semantic_ids", [])
        if v is not None
    }
    if not tracked:
        return []

    semantic = np.asarray(semantic_obs)
    if semantic.ndim != 2 or semantic.size == 0:
        return []

    visible_ids = {
        int(v) for v in np.unique(semantic).tolist()
        if int(v) in tracked
    }
    if not visible_ids:
        return []

    total_pixels = int(semantic.size)
    y0, y1, x0, x1 = _center_crop_bounds(
        int(semantic.shape[0]), int(semantic.shape[1]), float(center_region_fraction)
    )
    center_crop = semantic[y0:y1, x0:x1]
    recorded: list[dict[str, Any]] = []

    for semantic_id in sorted(visible_ids):
        mask = semantic == int(semantic_id)
        pixel_count = int(np.count_nonzero(mask))
        if pixel_count <= 0:
            continue
        pixel_ratio = float(pixel_count) / float(total_pixels)
        if pixel_ratio < float(pixel_ratio_threshold):
            continue
        center_pixel_count = int(np.count_nonzero(center_crop == int(semantic_id)))
        center_fraction = float(center_pixel_count) / float(pixel_count)
        if center_fraction < float(min_center_fraction):
            continue
        candidates = [dict(item) for item in (semantic_instances_by_id or {}).get(int(semantic_id), [])]
        if not candidates:
            candidates = [
                {
                    "object_instance_id": index,
                    "object_index": index,
                    "position": _safe_position(pos),
                }
                for index, pos in enumerate((semantic_positions_by_id or {}).get(int(semantic_id), []))
            ]
        if not candidates:
            instance_matches = [
                {
                    "position": _safe_position(position),
                    "position_type": "agent_position_fallback",
                    "pixel_count": int(pixel_count),
                    "pixel_ratio": float(pixel_ratio),
                    "center_pixel_count": int(center_pixel_count),
                    "center_fraction": float(center_fraction),
                    "instance_match_median_distance_m": None,
                }
            ]
        elif len(candidates) == 1:
            instance_matches = [
                {
                    **candidates[0],
                    "position_type": "object_semantic_translation",
                    "pixel_count": int(pixel_count),
                    "pixel_ratio": float(pixel_ratio),
                    "center_pixel_count": int(center_pixel_count),
                    "center_fraction": float(center_fraction),
                    "instance_match_median_distance_m": None,
                }
            ]
        elif candidates and depth_obs is not None and camera_intrinsics is not None and camera_pose is not None:
            instance_matches = _match_visible_semantic_instances(
                mask=mask,
                depth_obs=np.asarray(depth_obs),
                camera_intrinsics=np.asarray(camera_intrinsics),
                camera_pose=np.asarray(camera_pose),
                candidates=candidates,
                total_pixels=total_pixels,
                center_bounds=(y0, y1, x0, x1),
                pixel_ratio_threshold=float(pixel_ratio_threshold),
                min_center_fraction=float(min_center_fraction),
                max_distance_m=float(instance_match_max_distance_m),
            )
            for instance_match in instance_matches:
                instance_match["position_type"] = "object_semantic_translation"
        else:
            # Ambiguous class-level masks must not be assigned to an arbitrary instance.
            instance_matches = []

        for instance_match in instance_matches:
            record = {
                "semantic_id": int(semantic_id),
                "subtask_id": str(subtask_id),
                "subtask_order": int(subtask_order),
                "episode_index": int(episode_index),
                "event": str(event),
                "position": _safe_position(instance_match["position"]),
                "position_type": str(instance_match["position_type"]),
                "agent_position": _safe_position(position),
                "angle_rad": float(angle_rad),
                "pixel_count": int(instance_match["pixel_count"]),
                "pixel_ratio": float(instance_match["pixel_ratio"]),
                "center_pixel_count": int(instance_match["center_pixel_count"]),
                "center_fraction": float(instance_match["center_fraction"]),
                "image_height": int(semantic.shape[0]),
                "image_width": int(semantic.shape[1]),
            }
            state.setdefault("observations_by_semantic_id", {}).setdefault(
                str(int(semantic_id)), []
            ).append(record)
            recorded.append(record)
    return recorded


def _dedupe_positions(positions: Iterable[np.ndarray | list[float] | tuple[float, ...]]) -> list[np.ndarray]:
    deduped: list[np.ndarray] = []
    seen: set[tuple[float, float, float]] = set()
    for position in positions:
        arr = np.asarray(position, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            continue
        key = (
            round(float(arr[0]), 4),
            round(float(arr[1]), 4),
            round(float(arr[2]), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(np.asarray([arr[0], arr[1], arr[2]], dtype=np.float32))
    return deduped


def _position_trace(positions: Iterable[np.ndarray | list[float] | tuple[float, ...]]) -> list[np.ndarray]:
    trace: list[np.ndarray] = []
    for position in positions:
        arr = np.asarray(position, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            continue
        trace.append(np.asarray([arr[0], arr[1], arr[2]], dtype=np.float32))
    return trace


def _path_length_trace(path_lengths: Optional[Iterable[float]], expected_count: int) -> Optional[list[float]]:
    if path_lengths is None:
        return None
    values: list[float] = []
    for value in path_lengths:
        try:
            length = float(value)
        except Exception:
            return None
        if not np.isfinite(length):
            return None
        values.append(length)
    if len(values) != int(expected_count):
        return None
    return values


def _distance_m(pathfinder, a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32)[[0, 2]] - np.asarray(b, dtype=np.float32)[[0, 2]]))


def _path_length_xz(positions: list[np.ndarray], start_index: int) -> float:
    if start_index < 0 or start_index >= len(positions) - 1:
        return 0.0
    total = 0.0
    for idx in range(start_index, len(positions) - 1):
        total += float(np.linalg.norm(positions[idx + 1][[0, 2]] - positions[idx][[0, 2]]))
    return float(total)


def _snap_nav_point(pathfinder, position: np.ndarray) -> np.ndarray:
    pos = np.asarray(position, dtype=np.float32)
    if pathfinder is None:
        return pos
    try:
        snapped = pathfinder.snap_point(pos)
        snapped = np.asarray(snapped, dtype=np.float32)
        if snapped.size >= 3 and np.all(np.isfinite(snapped)):
            return snapped
    except Exception:
        pass
    return pos


def _nav_distance_m(pathfinder, start: np.ndarray, end: np.ndarray) -> float:
    start_arr = _snap_nav_point(pathfinder, np.asarray(start, dtype=np.float32))
    end_arr = _snap_nav_point(pathfinder, np.asarray(end, dtype=np.float32))
    if pathfinder is not None:
        try:
            import habitat_sim

            path = habitat_sim.ShortestPath()
            path.requested_start = start_arr
            path.requested_end = end_arr
            if pathfinder.find_path(path) and np.isfinite(float(path.geodesic_distance)):
                return float(path.geodesic_distance)
        except Exception:
            pass
    return _distance_m(pathfinder, start_arr, end_arr)


def evaluate_subtask_gt_history_retrieval(
    state: dict[str, Any],
    *,
    subtask_id: str,
    subtask_order: int,
    episode_index: int,
    goal_category: str,
    target_semantic_ids: Iterable[int],
    current_positions: Iterable[np.ndarray | list[float] | tuple[float, ...]],
    pathfinder,
    current_path_lengths: Optional[Iterable[float]] = None,
    current_target_positions_by_semantic_id: Optional[dict[int, list[np.ndarray | list[float] | tuple[float, ...]]]] = None,
    final_position: Optional[np.ndarray | list[float] | tuple[float, ...]] = None,
    success_distance_m: float = 1.5,
    retrieval_radius_m: float = 1.5,
) -> dict[str, Any]:
    current_position_trace = _position_trace(current_positions)
    current_path_length_trace = _path_length_trace(current_path_lengths, len(current_position_trace))
    deduped_current_positions = _dedupe_positions(current_position_trace)
    retrieval_positions = current_position_trace if current_position_trace else deduped_current_positions
    instance_results: list[dict[str, Any]] = []
    target_ids: list[int] = []
    seen_target_ids: set[int] = set()

    for value in target_semantic_ids:
        try:
            semantic_id = int(value)
        except Exception:
            continue
        if semantic_id in seen_target_ids:
            continue
        seen_target_ids.add(semantic_id)
        target_ids.append(semantic_id)

    for semantic_id in target_ids:
        prior_records = [
            dict(record)
            for record in state.get("observations_by_semantic_id", {}).get(str(int(semantic_id)), [])
            if int(record.get("subtask_order", -1)) < int(subtask_order)
        ]
        if not prior_records:
            instance_results.append(
                {
                    "semantic_id": int(semantic_id),
                    "eligible": False,
                    "retrieval_success": None,
                    "historical_observation_count": 0,
                    "retrieval_radius_m": float(retrieval_radius_m),
                }
            )
            continue

        best_distance = float("inf")
        best_record = None
        best_current_position = None
        best_current_index = None
        for record in prior_records:
            historical_position = np.asarray(record["position"], dtype=np.float32)
            for current_index, current_position in enumerate(retrieval_positions):
                dist = _distance_m(pathfinder, historical_position, np.asarray(current_position, dtype=np.float32))
                if dist < best_distance:
                    best_distance = float(dist)
                    best_record = record
                    best_current_position = np.asarray(current_position, dtype=np.float32)
                    best_current_index = int(current_index)

        current_target_positions = [
            np.asarray(pos, dtype=np.float32)
            for pos in (current_target_positions_by_semantic_id or {}).get(int(semantic_id), [])
        ]
        relocation_result = None
        if best_current_position is not None and best_distance <= float(retrieval_radius_m) and current_target_positions:
            nearest_target_pos = None
            nearest_target_distance = float("inf")
            historical_position = np.asarray(best_record["position"], dtype=np.float32) if best_record is not None else best_current_position
            for target_pos in current_target_positions:
                dist = _distance_m(pathfinder, historical_position, target_pos)
                if dist < nearest_target_distance:
                    nearest_target_distance = float(dist)
                    nearest_target_pos = target_pos
            moved_from_history = bool(nearest_target_distance > float(success_distance_m))
            final_arr = np.asarray(final_position, dtype=np.float32) if final_position is not None else (
                retrieval_positions[-1] if retrieval_positions else best_current_position
            )
            final_target_distance = (
                _distance_m(pathfinder, final_arr, nearest_target_pos)
                if nearest_target_pos is not None
                else float("inf")
            )
            relocation_success = bool(final_target_distance < float(success_distance_m))
            oracle_nav_distance = (
                _nav_distance_m(pathfinder, best_current_position, nearest_target_pos)
                if nearest_target_pos is not None
                else float("inf")
            )
            actual_suffix_path_length_source = "position_trace_xz_fallback"
            if current_path_length_trace is not None and best_current_index is not None:
                actual_suffix_path_length = max(
                    0.0,
                    float(current_path_length_trace[-1]) - float(current_path_length_trace[int(best_current_index)]),
                )
                actual_suffix_path_length_source = "logger_subtask_explore_dist_trace"
            else:
                actual_suffix_path_length = _path_length_xz(
                    retrieval_positions,
                    int(best_current_index) if best_current_index is not None else len(retrieval_positions) - 1,
                )
            relocation_spl = (
                float(oracle_nav_distance) / max(float(oracle_nav_distance), float(actual_suffix_path_length), 1e-6)
                if relocation_success and moved_from_history and np.isfinite(float(oracle_nav_distance))
                else 0.0
            )
            relocation_result = {
                "recorded": bool(moved_from_history),
                "moved_from_history": bool(moved_from_history),
                "relocation_success": bool(relocation_success) if moved_from_history else None,
                "relocation_spl": float(relocation_spl) if moved_from_history else None,
                "historical_to_current_target_distance_m": float(nearest_target_distance),
                "final_to_current_target_distance_m": float(final_target_distance),
                "oracle_nav_distance_m": float(oracle_nav_distance) if np.isfinite(float(oracle_nav_distance)) else None,
                "actual_suffix_path_length_m": float(actual_suffix_path_length),
                "actual_suffix_path_length_source": actual_suffix_path_length_source,
                "nearest_current_target_position": _safe_position(nearest_target_pos) if nearest_target_pos is not None else None,
                "revisit_position_trace_index": best_current_index,
            }

        instance_results.append(
            {
                "semantic_id": int(semantic_id),
                "eligible": True,
                "retrieval_success": bool(best_distance <= float(retrieval_radius_m)),
                "historical_observation_count": int(len(prior_records)),
                "retrieval_radius_m": float(retrieval_radius_m),
                "closest_historical_distance_m": float(best_distance),
                "matched_historical_observation": best_record,
                "matched_current_position": _safe_position(best_current_position) if best_current_position is not None else None,
                "relocation_after_revisit": relocation_result,
            }
        )

    target_instance_count = int(len(target_ids))
    eligible_instance_count = int(sum(1 for item in instance_results if item.get("eligible")))
    retrieved_instance_count = int(
        sum(1 for item in instance_results if item.get("eligible") and item.get("retrieval_success"))
    )
    relocation_records = [
        item.get("relocation_after_revisit")
        for item in instance_results
        if isinstance(item.get("relocation_after_revisit"), dict)
        and item["relocation_after_revisit"].get("recorded") is True
    ]
    relocation_recorded_instance_count = int(len(relocation_records))
    relocation_success_instance_count = int(
        sum(1 for item in relocation_records if item.get("relocation_success") is True)
    )
    relocation_spl_values = [
        float(item.get("relocation_spl", 0.0))
        for item in relocation_records
        if item.get("relocation_spl") is not None
    ]
    summary = {
        "subtask_id": str(subtask_id),
        "subtask_order": int(subtask_order),
        "episode_index": int(episode_index),
        "goal_category": str(goal_category),
        "target_semantic_ids": target_ids,
        "target_instance_count": int(target_instance_count),
        "previously_observed_instance_count": int(eligible_instance_count),
        "previously_observed_probability": (
            float(eligible_instance_count) / float(target_instance_count)
            if target_instance_count > 0
            else None
        ),
        "eligible_instance_count": int(eligible_instance_count),
        "retrieved_instance_count": int(retrieved_instance_count),
        "instance_success_rate": (
            float(retrieved_instance_count) / float(eligible_instance_count)
            if eligible_instance_count > 0
            else None
        ),
        "retrieval_success_any": bool(retrieved_instance_count > 0) if eligible_instance_count > 0 else None,
        "retrieval_success_all": (
            bool(retrieved_instance_count == eligible_instance_count)
            if eligible_instance_count > 0
            else None
        ),
        "relocation_after_revisit_recorded_instance_count": int(relocation_recorded_instance_count),
        "relocation_after_revisit_success_instance_count": int(relocation_success_instance_count),
        "relocation_after_revisit_success_rate": (
            float(relocation_success_instance_count) / float(relocation_recorded_instance_count)
            if relocation_recorded_instance_count > 0
            else None
        ),
        "relocation_after_revisit_spl": (
            float(sum(relocation_spl_values)) / float(len(relocation_spl_values))
            if relocation_spl_values
            else None
        ),
        "current_position_count": int(len(retrieval_positions)),
        "instance_results": instance_results,
    }
    state.setdefault("retrieval_by_subtask", {})[str(subtask_id)] = summary
    return summary


def compute_gt_history_retrieval_aggregate(results_by_subtask: dict[str, Any]) -> dict[str, Any]:
    entries = [value for value in results_by_subtask.values() if isinstance(value, dict)]
    target_instance_count = int(
        sum(int(entry.get("target_instance_count", 0)) for entry in entries)
    )
    previously_observed_instance_count = int(
        sum(int(entry.get("previously_observed_instance_count", entry.get("eligible_instance_count", 0))) for entry in entries)
    )
    eligible_instance_count = int(
        sum(int(entry.get("eligible_instance_count", 0)) for entry in entries)
    )
    retrieved_instance_count = int(
        sum(int(entry.get("retrieved_instance_count", 0)) for entry in entries)
    )
    eligible_subtask_count = int(
        sum(1 for entry in entries if int(entry.get("eligible_instance_count", 0)) > 0)
    )
    retrieved_subtask_count_any = int(
        sum(1 for entry in entries if entry.get("retrieval_success_any") is True)
    )
    retrieved_subtask_count_all = int(
        sum(1 for entry in entries if entry.get("retrieval_success_all") is True)
    )
    relocation_recorded_instance_count = int(
        sum(int(entry.get("relocation_after_revisit_recorded_instance_count", 0)) for entry in entries)
    )
    relocation_success_instance_count = int(
        sum(int(entry.get("relocation_after_revisit_success_instance_count", 0)) for entry in entries)
    )
    relocation_spl_sum = 0.0
    relocation_spl_count = 0
    for entry in entries:
        count = int(entry.get("relocation_after_revisit_recorded_instance_count", 0))
        spl = entry.get("relocation_after_revisit_spl")
        if count <= 0 or spl is None:
            continue
        relocation_spl_sum += float(spl) * float(count)
        relocation_spl_count += count
    return {
        "subtask_count": int(len(entries)),
        "target_instance_count": int(target_instance_count),
        "previously_observed_instance_count": int(previously_observed_instance_count),
        "previously_observed_probability": (
            float(previously_observed_instance_count) / float(target_instance_count)
            if target_instance_count > 0
            else None
        ),
        "eligible_instance_count": int(eligible_instance_count),
        "retrieved_instance_count": int(retrieved_instance_count),
        "instance_success_rate": (
            float(retrieved_instance_count) / float(eligible_instance_count)
            if eligible_instance_count > 0
            else None
        ),
        "eligible_subtask_count": int(eligible_subtask_count),
        "retrieved_subtask_count_any": int(retrieved_subtask_count_any),
        "retrieved_subtask_count_all": int(retrieved_subtask_count_all),
        "subtask_success_rate_any": (
            float(retrieved_subtask_count_any) / float(eligible_subtask_count)
            if eligible_subtask_count > 0
            else None
        ),
        "subtask_success_rate_all": (
            float(retrieved_subtask_count_all) / float(eligible_subtask_count)
            if eligible_subtask_count > 0
            else None
        ),
        "relocation_after_revisit_recorded_instance_count": int(relocation_recorded_instance_count),
        "relocation_after_revisit_success_instance_count": int(relocation_success_instance_count),
        "relocation_after_revisit_success_rate": (
            float(relocation_success_instance_count) / float(relocation_recorded_instance_count)
            if relocation_recorded_instance_count > 0
            else None
        ),
        "relocation_after_revisit_spl": (
            float(relocation_spl_sum) / float(relocation_spl_count)
            if relocation_spl_count > 0
            else None
        ),
    }


def save_gt_history_retrieval_results(
    output_dir: Path,
    *,
    start_ratio: float,
    end_ratio: float,
    split: int,
    results_by_subtask: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = compute_gt_history_retrieval_aggregate(results_by_subtask)
    suffix = f"{start_ratio}_{end_ratio}_{split}"
    by_subtask_path = output_dir / f"gt_history_retrieval_by_subtask_{suffix}.json"
    aggregate_path = output_dir / f"gt_history_retrieval_aggregate_{suffix}.json"
    for path in [
        by_subtask_path,
        output_dir / "gt_history_retrieval_by_subtask.json",
    ]:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(results_by_subtask, fp, indent=2, ensure_ascii=False)
    for path in [
        aggregate_path,
        output_dir / "gt_history_retrieval_aggregate.json",
    ]:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(aggregate, fp, indent=2, ensure_ascii=False)
