"""Adapted from Eku127/habitat-data-collector (MIT)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class EpisodeSubtask:
    task_index: int
    goal_type: str
    goal_category: str
    goal_key: str
    goal_entries: List[Dict]


@dataclass
class EpisodeSpec:
    episode_index: int
    scene_config_path: Path
    start_agent_state: Dict
    mid_object_states: List[Dict]
    subtasks: List[EpisodeSubtask]


@dataclass
class EpisodeSet:
    scene_dir: Path
    scene_id: str
    episode_id: int
    scene_path: str
    scene_dataset_config: str
    episodes: List[EpisodeSpec]


def discover_scene_dirs(dataset_root: Path) -> List[Path]:
    """Find generated scene dirs that contain runtime bundle."""
    if not dataset_root.exists():
        return []
    scene_dirs: List[Path] = []
    for candidate in sorted(dataset_root.iterdir()):
        if not candidate.is_dir():
            continue
        runtime_bundle = (
            candidate / "navigation_task_pipeline" / "navigation_task_runtime_bundle.json"
        )
        sample_json = candidate / "navigation_task_pipeline" / "goat_navigation_sample.json"
        if runtime_bundle.exists() or sample_json.exists():
            scene_dirs.append(candidate)
    return scene_dirs


def load_runtime_bundle(scene_dir: Path) -> Dict:
    runtime_bundle_path = (
        scene_dir / "navigation_task_pipeline" / "navigation_task_runtime_bundle.json"
    )
    if runtime_bundle_path.exists():
        with open(runtime_bundle_path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    bundle = {
        "task_data": {},
        "episode_scene_configs": [],
        "episode_object_states": [],
        "episode_mid_object_states": [],
    }
    sample_path = scene_dir / "navigation_task_pipeline" / "goat_navigation_sample.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as fp:
            bundle["task_data"] = json.load(fp)

    episode_states_path = scene_dir / "navigation_task_pipeline" / "episode_object_states.json"
    if episode_states_path.exists():
        with open(episode_states_path, "r", encoding="utf-8") as fp:
            bundle["episode_object_states"] = json.load(fp)

    episode_mid_states_path = (
        scene_dir / "navigation_task_pipeline" / "episode_mid_object_states.json"
    )
    if episode_mid_states_path.exists():
        with open(episode_mid_states_path, "r", encoding="utf-8") as fp:
            bundle["episode_mid_object_states"] = json.load(fp)

    episode_index_path = (
        scene_dir / "navigation_task_pipeline" / "episode_scene_configs" / "index.json"
    )
    if episode_index_path.exists():
        with open(episode_index_path, "r", encoding="utf-8") as fp:
            bundle["episode_scene_configs"] = json.load(fp)

    return bundle


def _build_task_episode_index_lookup(task_data: Dict, episode_index: int = 0) -> List[Optional[int]]:
    episodes = task_data.get("episodes", [])
    if episode_index < 0 or episode_index >= len(episodes):
        return []
    task_count = len(episodes[episode_index].get("tasks", []))
    lookup: List[Optional[int]] = [None] * task_count
    episode_summaries = task_data.get("metadata", {}).get("episode_summaries", [])
    for item in episode_summaries:
        try:
            episode_index = int(item.get("episode_index"))
            start_idx = int(item.get("start_target_index"))
            end_idx = int(item.get("end_target_index"))
        except Exception:
            continue
        if end_idx < start_idx:
            continue
        for task_idx in range(max(start_idx, 0), min(end_idx, task_count - 1) + 1):
            lookup[task_idx] = episode_index
    return lookup


def _resolve_episode_scene_configs(scene_dir: Path, bundle: Dict) -> List[Dict]:
    episode_scene_configs = bundle.get("episode_scene_configs", [])
    if isinstance(episode_scene_configs, list) and episode_scene_configs:
        return episode_scene_configs
    index_path = scene_dir / "navigation_task_pipeline" / "episode_scene_configs" / "index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        if isinstance(loaded, list):
            return loaded
    return []


def _resolve_episode_start_agent_states(scene_dir: Path, task_data: Dict) -> List[Dict]:
    metadata_states = task_data.get("metadata", {}).get("episode_start_agent_states", [])
    if isinstance(metadata_states, list) and metadata_states:
        return metadata_states

    states_path = scene_dir / "navigation_task_pipeline" / "episode_start_agent_states.json"
    if states_path.exists():
        with open(states_path, "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        if isinstance(loaded, list):
            return loaded
    return []


def _resolve_episode_mid_object_states(scene_dir: Path, bundle: Dict, task_data: Dict) -> List[Dict]:
    bundle_states = bundle.get("episode_mid_object_states", [])
    if isinstance(bundle_states, list) and bundle_states:
        return bundle_states

    metadata_states = task_data.get("metadata", {}).get("episode_mid_object_states", [])
    if isinstance(metadata_states, list) and metadata_states:
        return metadata_states

    states_path = scene_dir / "navigation_task_pipeline" / "episode_mid_object_states.json"
    if states_path.exists():
        with open(states_path, "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        if isinstance(loaded, list):
            return loaded
    return []


def _resolve_mid_objects_for_episode(
    episode_mid_object_states: Sequence[Dict], episode_index: int
) -> List[Dict]:
    for item in episode_mid_object_states:
        try:
            idx = int(item.get("episode_index"))
        except Exception:
            continue
        if idx != int(episode_index):
            continue
        mid_objects = item.get("mid_objects", [])
        if isinstance(mid_objects, list):
            return [obj for obj in mid_objects if isinstance(obj, dict)]
        return []
    return []


def _goal_episode_from_key(goal_key: str) -> Optional[int]:
    match = re.search(r"_episode(\d+)_", str(goal_key))
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _resolve_goal_key(
    scene_asset_name: str,
    task_entry: Sequence,
    goals: Dict,
    task_episode_index: Optional[int],
) -> Optional[str]:
    if len(task_entry) > 2 and isinstance(task_entry[2], str):
        explicit_key = task_entry[2]
        if explicit_key in goals:
            return explicit_key

    goal_category = str(task_entry[0])
    if task_episode_index is not None:
        key_new = f"{scene_asset_name}_episode{int(task_episode_index):02d}_{goal_category}"
        if key_new in goals:
            return key_new

    key_legacy = f"{scene_asset_name}_{goal_category}"
    if key_legacy in goals:
        return key_legacy

    return None


def _resolve_scene_config_path_for_episode(
    episode_scene_configs: Sequence[Dict], episode_index: int
) -> Optional[str]:
    for item in episode_scene_configs:
        if item.get("stage") != "after_episode":
            continue
        try:
            idx = int(item.get("episode_index"))
        except Exception:
            continue
        if idx == int(episode_index):
            return item.get("scene_config_path")
    return None


def _resolve_start_agent_state_for_episode(
    episode_start_agent_states: Sequence[Dict], episode_index: int
) -> Optional[Dict]:
    for item in episode_start_agent_states:
        try:
            idx = int(item.get("episode_index"))
        except Exception:
            continue
        if idx == episode_index:
            state = item.get("agent_state")
            if isinstance(state, dict):
                return state
    return None


def build_episodes(
    scene_dir: Path,
    bundle: Dict,
    episode_index: int = 0,
) -> EpisodeSet:
    """Build the ordered episode states for one lifelong sequence."""
    task_data = bundle.get("task_data", {})
    episodes = task_data.get("episodes", [])
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(f"Episode index out of range: {episode_index}/{len(episodes)}")

    episode = episodes[episode_index]
    goals = task_data.get("goals", {})
    tasks = episode.get("tasks", [])
    episode_lookup = _build_task_episode_index_lookup(task_data, episode_index=episode_index)
    episode_scene_configs = _resolve_episode_scene_configs(scene_dir, bundle)
    episode_start_agent_states = _resolve_episode_start_agent_states(scene_dir, task_data)
    episode_mid_object_states = _resolve_episode_mid_object_states(
        scene_dir=scene_dir, bundle=bundle, task_data=task_data
    )
    scene_path = str(episode.get("scene_id", ""))
    scene_dataset_config = str(episode.get("scene_dataset_config", ""))
    scene_asset_name = Path(scene_path).name
    scene_id = Path(scene_path).parent.name
    episode_id = int(episode.get("episode_id", episode_index))

    subtasks_by_episode: Dict[int, List[EpisodeSubtask]] = {}
    for task_index, task_entry in enumerate(tasks):
        if not isinstance(task_entry, list) or len(task_entry) < 2:
            continue
        goal_type = str(task_entry[1])
        goal_category = str(task_entry[0])
        episode_index = None
        if task_index < len(episode_lookup):
            episode_index = episode_lookup[task_index]
        if episode_index is None and len(task_entry) > 2:
            episode_index = _goal_episode_from_key(str(task_entry[2]))
        if episode_index is None:
            episode_index = 0

        goal_key = _resolve_goal_key(
            scene_asset_name=scene_asset_name,
            task_entry=task_entry,
            goals=goals,
            task_episode_index=episode_index,
        )
        if goal_key is None:
            continue
        goal_entries = goals.get(goal_key, [])
        if not isinstance(goal_entries, list) or not goal_entries:
            continue
        subtasks_by_episode.setdefault(int(episode_index), []).append(
            EpisodeSubtask(
                task_index=task_index,
                goal_type=goal_type,
                goal_category=goal_category,
                goal_key=goal_key,
                goal_entries=goal_entries,
            )
        )

    episodes: List[EpisodeSpec] = []
    for episode_index in sorted(subtasks_by_episode.keys()):
        scene_config_rel = _resolve_scene_config_path_for_episode(episode_scene_configs, episode_index)
        if scene_config_rel is None:
            raise KeyError(f"Missing episode scene config for episode {episode_index}")
        start_agent_state = _resolve_start_agent_state_for_episode(
            episode_start_agent_states, episode_index
        )
        if start_agent_state is None:
            raise KeyError(f"Missing episode start agent state for episode {episode_index}")
        episodes.append(
            EpisodeSpec(
                episode_index=episode_index,
                scene_config_path=scene_dir / scene_config_rel,
                start_agent_state=start_agent_state,
                mid_object_states=_resolve_mid_objects_for_episode(
                    episode_mid_object_states, episode_index
                ),
                subtasks=subtasks_by_episode[episode_index],
            )
        )

    return EpisodeSet(
        scene_dir=scene_dir,
        scene_id=scene_id,
        episode_id=episode_id,
        scene_path=scene_path,
        scene_dataset_config=scene_dataset_config,
        episodes=episodes,
    )
