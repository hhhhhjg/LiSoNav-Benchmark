"""Public, model independent agent contract. Coordinates: Habitat XYZ, metres."""
from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np

@dataclass(frozen=True)
class Action:
    name: str  # move_forward, turn_right, turn_left, stop
    amount: Optional[float] = None  # metres for movement, degrees for rotation

@dataclass(frozen=True)
class TaskRequest:
    subtask_id: str
    sequence_id: str
    scene_id: str
    episode_index: int
    task_index: int
    goal_type: str
    goal_category: str
    instruction: str
    target_description: str
    target_surface_description: str
    reference_image: Optional[np.ndarray]
    new_sequence: bool
    new_episode: bool

@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    depth: np.ndarray  # metres
    position: np.ndarray
    rotation_xyzw: np.ndarray
    camera_intrinsics: np.ndarray
    camera_pose: np.ndarray  # sensor to world
    path_length_m: float
    action_count: int
    collided: bool
    done: bool
    stop_reason: Optional[str]

class Agent(Protocol):
    def reset(self, task: TaskRequest) -> None:
        """Prepare next task; retain memory unless task.new_sequence is true."""
    def act(self, observation: Observation) -> Action: ...
    def on_task_end(self, result: dict) -> None: ...
    def close(self) -> None: ...
