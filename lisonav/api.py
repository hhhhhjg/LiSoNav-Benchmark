"""Trusted evaluation environment: lifecycle, observations, actions and metrics."""
from dataclasses import dataclass, asdict
from typing import Optional
import math
import numpy as np
from .types import Action, Observation
from .backends import MockSimulator, HabitatSimulator
from .navigation import SubtaskNavigationMetricsTracker
from .target_distance import (load_episode_semantic_positions, resolve_subtask_target_positions,
                              calc_min_horizontal_distance, calc_min_geodesic_distance_to_targets)
from .history_retrieval import (collect_episode_goal_semantic_ids, new_gt_history_retrieval_state,
    collect_episode_semantic_instances, record_gt_target_observations,
    resolve_episode_goal_semantic_ids, collect_episode_semantic_positions,
    evaluate_subtask_gt_history_retrieval)

@dataclass(frozen=True)
class BenchmarkConfig:
    success_distance: float = 1.5
    path_budget_multiplier: float = 10.0
    path_budget_m: Optional[float] = None  # explicit override for smoke testing
    max_actions: int = 1000
    translation_step_m: float = 0.25  # sample path, collisions and sensors per primitive
    max_move_m: float = 10.0
    width: int = 1280
    height: int = 1280
    hfov: float = 120.0
    camera_height: float = 1.5
    camera_tilt_deg: float = -30.0
    agent_height: float = 1.5
    navmesh_agent_radius: float = 0.0
    gpu_device_id: int = -1
    seed: int = 77

    def __post_init__(self):
        for name in ['success_distance','path_budget_multiplier','translation_step_m','max_move_m','camera_height','agent_height']:
            if not math.isfinite(getattr(self,name)) or getattr(self,name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.path_budget_m is not None and (not math.isfinite(self.path_budget_m) or self.path_budget_m < 0):
            raise ValueError('path_budget_m must be finite and nonnegative')
        if self.max_actions <= 0 or self.width <= 0 or self.height <= 0 or not 0 < self.hfov < 180:
            raise ValueError('Invalid action limit or camera dimensions/FOV')
        for name in ['max_actions', 'width', 'height']:
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise ValueError(f'{name} must be an integer')
        if not math.isfinite(self.camera_tilt_deg) or not math.isfinite(self.navmesh_agent_radius) or self.navmesh_agent_radius < 0:
            raise ValueError('Invalid camera tilt or navmesh radius')

class NavigationEnvironment:
    def __init__(self, dataset, config=None, backend='habitat'):
        self.dataset = dataset
        self.config = config or BenchmarkConfig()
        if backend not in ['mock','habitat']:
            raise ValueError(f'Unknown backend: {backend}')
        self.backend_name = backend
        self._sim = (MockSimulator if backend=='mock' else HabitatSimulator)(dataset.assets,self.config)
        self._sequence = None
        self._episode = None
        self._task = None
        self._result = None
        self._results = []
        # Match the upstream camera matrix, including odd image resolutions.
        hfov_rad = self.config.hfov*np.pi/180
        vfov_rad = 2*np.arctan(np.tan(hfov_rad/2)*self.config.height/self.config.width)
        fx = (1/np.tan(hfov_rad/2))*self.config.width/2
        fy = (1/np.tan(vfov_rad/2))*self.config.height/2
        self._intrinsics = np.array([[fx,0,self.config.width//2],[0,fy,self.config.height//2],[0,0,1]])

    def begin_sequence(self, sequence):
        self._ensure_finished()
        self._sequence = sequence
        self._episode = None
        self._task = None
        self._result = None
        self._history = new_gt_history_retrieval_state(collect_episode_goal_semantic_ids(sequence))
        self._order = -1

    def begin_episode(self, episode):
        self._ensure_finished()
        if self._sequence is None or not any(episode is item for item in self._sequence.episodes):
            raise ValueError('begin_sequence must precede begin_episode with its episode')
        self._sim.load_episode(self._sequence,episode)
        self._episode = episode
        self._task = None
        self._result = None
        self._semantic_positions = load_episode_semantic_positions(episode.scene_config_path)
        self._instances = collect_episode_semantic_instances(episode.scene_config_path,self._history['tracked_semantic_ids'])

    def _ensure_finished(self):
        if self._task is not None and self._result is None:
            raise RuntimeError('Finish and evaluate the current subtask before advancing')

    def reset_task(self, subtask):
        """Starts counters for a task, preserving pose and sequence history."""
        self._ensure_finished()
        if self._episode is None or not any(subtask is item for item in self._episode.subtasks):
            raise ValueError('begin_episode must precede reset_task with its subtask')
        self._order += 1
        self._task = self.dataset.task_request(self._sequence,self._episode,subtask,
            new_sequence=self._order==0,new_episode=subtask is self._episode.subtasks[0])
        self._subtask = subtask
        self._targets = resolve_subtask_target_positions(subtask.goal_entries,self._semantic_positions)
        if not self._targets:
            raise ValueError(f'No valid targets for {self._task.subtask_id}')
        raw = self._sim.capture()
        self._optimal_source = 'geodesic'
        self._distance_diagnostics = {}
        self._optimal = (calc_min_geodesic_distance_to_targets(self._sim.pathfinder,raw['position'],self._targets,
                             diagnostics=self._distance_diagnostics) if self._sim.pathfinder is not None else math.inf)
        if not math.isfinite(self._optimal):
            self._optimal = calc_min_horizontal_distance(raw['position'],self._targets)
            self._optimal_source = 'horizontal_fallback'
        self._budget = self.config.path_budget_m if self.config.path_budget_m is not None else self._optimal*self.config.path_budget_multiplier
        self._path = 0.0
        self._actions = 0
        self._done = False
        self._stop_reason = None
        self._collided = False
        self._tracker = SubtaskNavigationMetricsTracker()
        self._result = None
        self._trace = [raw['position'].tolist()]
        self._length_trace = [0.0]
        self._events = []
        self._frames = 0
        self._record(raw,'task_start')
        self._raw = raw
        return self._task, self.observe()

    def _record(self, raw, event):
        self._frames += 1
        rotation = raw['rotation_xyzw']
        heading = 2*math.atan2(float(rotation[1]),float(rotation[3]))
        record_gt_target_observations(self._history,semantic_obs=raw['semantic'],
            position=raw['position'],semantic_instances_by_id=self._instances,
            depth_obs=raw['depth'],camera_intrinsics=self._intrinsics,camera_pose=raw['camera_pose'],
            angle_rad=heading,subtask_id=self._task.subtask_id,subtask_order=self._order,
            episode_index=self._episode.episode_index,event=event)

    def observe(self):
        """Read the current observation; does not move or create another GT record."""
        if self._task is None:
            raise RuntimeError('reset_task must precede observe')
        raw = self._raw
        return Observation(raw['rgb'].copy(),raw['depth'].copy(),raw['position'].copy(),
                           raw['rotation_xyzw'].copy(),self._intrinsics.copy(),raw['camera_pose'].copy(),
                           self._path,self._actions,self._collided,self._done,self._stop_reason)

    def step(self, action):
        """Execute an agent command; environment owns path budget and termination."""
        if self._task is None or self._done:
            raise RuntimeError('No active task (actions after termination are rejected)')
        if not isinstance(action,Action) or action.name not in ['move_forward','turn_right','turn_left','stop']:
            raise ValueError('Expected Action(move_forward/turn_right/turn_left/stop)')
        if action.name == 'stop':
            if action.amount is not None:
                raise ValueError('stop has no amount')
            self._actions += 1
            self._done, self._stop_reason = True,'agent_stop'
            self._events.append(dict(action='stop',amount=None,path_length_m=self._path,collided=False))
            return self.observe()
        amount = action.amount if action.amount is not None else (0.25 if action.name=='move_forward' else 90.0)
        if isinstance(amount,bool) or not isinstance(amount,(int,float)) or not math.isfinite(amount) or amount <= 0:
            raise ValueError('Action amount must be finite and positive')
        if amount > (self.config.max_move_m if action.name=='move_forward' else 360):
            raise ValueError('Action amount exceeds limit')
        self._actions += 1
        remaining = float(amount)
        collision_steps, executed_steps = 0,0
        start_length = self._path
        while remaining > 1e-8:
            delta = min(remaining,self.config.translation_step_m) if action.name=='move_forward' else remaining
            previous = self._raw['position'].copy()
            collided = self._sim.execute(action.name,delta)
            executed_steps += 1
            collision_steps += int(collided)
            raw = self._sim.capture()
            self._path += float(np.linalg.norm((raw['position']-previous)[[0,2]]))
            self._trace.append(raw['position'].tolist())
            self._length_trace.append(self._path)
            self._raw = raw
            self._record(raw,action.name)
            remaining -= delta
            # Matches original > budget condition. Overshoot is at most one primitive.
            if self._path > self._budget+1e-6:
                self._done,self._stop_reason = True,'path_budget_exceeded'
                break
        self._collided = collision_steps > 0
        self._tracker.record_step(dict(total_steps=executed_steps,collision_steps=collision_steps))
        self._events.append(dict(action=action.name,amount=amount,executed_amount=amount-remaining,
            path_delta_m=self._path-start_length,path_length_m=self._path,collided=self._collided,
            primitive_steps=executed_steps,collision_steps=collision_steps))
        if not self._done and self._actions >= self.config.max_actions:
            self._done,self._stop_reason = True,'max_actions_exceeded'
        return self.observe()

    def evaluate(self):
        """Compute once after stop/forced stop. Repeated calls return a copy."""
        import copy
        if self._task is None or not self._done:
            raise RuntimeError('Metrics are available only after termination')
        if self._result is not None:
            return copy.deepcopy(self._result)
        distance = calc_min_horizontal_distance(self._raw['position'],self._targets)
        metrics = self._tracker.finalize(subtask_id=self._task.subtask_id,goal_type=self._task.goal_type,
            success_by_distance=distance < self.config.success_distance,gt_explore_distance=self._optimal,
            explored_distance=self._path,n_filtered_frames=0,n_total_frames=self._frames)
        ids = resolve_episode_goal_semantic_ids(self._episode.scene_config_path,self._subtask.goal_entries)
        history = evaluate_subtask_gt_history_retrieval(self._history,subtask_id=self._task.subtask_id,
            subtask_order=self._order,episode_index=self._episode.episode_index,
            goal_category=self._task.goal_category,target_semantic_ids=ids,current_positions=self._trace,
            current_path_lengths=self._length_trace,pathfinder=self._sim.pathfinder,
            current_target_positions_by_semantic_id=collect_episode_semantic_positions(self._episode.scene_config_path,ids),
            final_position=self._raw['position'],success_distance_m=self.config.success_distance)
        self._result = dict(asdict(metrics),backend=self.backend_name,sequence_id=self._task.sequence_id,
            episode_index=self._episode.episode_index,task_index=self._task.task_index,
            stop_reason=self._stop_reason,forced_stop=self._stop_reason!='agent_stop',
            path_budget_m=self._budget,optimal_distance_source=self._optimal_source,
            optimal_distance_diagnostics=self._distance_diagnostics,agent_target_distance=distance,
            final_position=self._raw['position'].tolist(),action_count=self._actions,
            position_trace=self._trace,path_length_trace=self._length_trace,actions=self._events,
            gt_history_retrieval=history)
        self._results.append(copy.deepcopy(self._result))
        return copy.deepcopy(self._result)

    def aggregate(self):
        from .history_retrieval import compute_gt_history_retrieval_aggregate
        def group(rows):
            return dict(count=len(rows), **{key: float(np.mean([r[key] for r in rows])) if rows else 0.0
                for key in ['success_by_distance','spl_by_distance','collision_ratio','target_not_on_navmesh_ratio']},
                forced_stop_count=sum(r['forced_stop'] for r in rows))
        return dict(backend=self.backend_name,overall=group(self._results),
            by_goal_type={kind:group([r for r in self._results if r['goal_type']==kind]) for kind in sorted({r['goal_type'] for r in self._results})},
            by_episode_index={str(index):group([r for r in self._results if r['episode_index']==index]) for index in sorted({r['episode_index'] for r in self._results})},
            by_sequence={sid:group([r for r in self._results if r['sequence_id']==sid]) for sid in sorted({r['sequence_id'] for r in self._results})},
            gt_history_retrieval=compute_gt_history_retrieval_aggregate({r['subtask_id']:r['gt_history_retrieval'] for r in self._results}))

    def history_state(self):
        """Trusted evaluator snapshot, including every visibility record for this sequence.

        This is for persistence/reanalysis and must not be passed to an agent.
        """
        import copy
        if self._sequence is None:
            raise RuntimeError('begin_sequence must precede history_state')
        return copy.deepcopy(self._history)

    def close(self):
        self._sim.close()
