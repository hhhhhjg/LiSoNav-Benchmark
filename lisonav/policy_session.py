"""Synchronous adapter for policies with an existing NavMesh path controller.

The controller retains its original sampling/quantization. Simulator reads,
trajectory accounting and ground truth scoring are routed through this session.
The primitive NavigationEnvironment API remains the entry for new agents.
"""
from dataclasses import dataclass
import math
import numpy as np
from .target_distance import calc_min_horizontal_distance, resolve_subtask_target_positions, load_episode_semantic_positions
from .history_retrieval import (resolve_episode_goal_semantic_ids, collect_episode_semantic_positions,
                               evaluate_subtask_gt_history_retrieval)

@dataclass(frozen=True)
class PoseAction:
    position: object
    angle: float = None
    rotation: object = None

@dataclass(frozen=True)
class SessionCommand:
    operation: str
    arguments: tuple = ()
    keywords: object = None

class SessionProxy:
    """Policy-side gateway. The driver executes requests in its own thread."""
    def __init__(self, request):
        self.request = request
    def observe_at(self, pts, angle=None, rotation=None):
        return self.request(SessionCommand('pose', (PoseAction(pts, angle, rotation),)))
    def observe_frontier(self, pts, view_dir, camera_tilt=0.0):
        return self.request(SessionCommand('frontier', (pts, view_dir, camera_tilt)))
    def record_navigation(self, tracker, planner, position):
        return self.request(SessionCommand('navigation', (tracker, planner, position)))
    def path_budget_exceeded(self, budget):
        return self.request(SessionCommand('budget', (budget,)))
    def stop(self, position):
        return self.request(SessionCommand('stop', (position,)))
    def evaluate(self, tracker):
        return self.request(SessionCommand('evaluate', (tracker,)))

class PolicySession:
    def __init__(self, dataset, sequence, scene, cfg, logger):
        self.dataset, self.sequence, self.scene = dataset,sequence,scene
        self.cfg,self.logger = cfg,logger
        self._capture = scene.get_observation
        self._frontier_capture = scene.get_frontier_observation
        self.scene.get_observation = self.observe_at
        self.scene.get_frontier_observation = self.observe_frontier
        self.logger._benchmark_session = self
        self.closed = False
        self.active = False
        self.done = False
        self.observation_calls = 0
        self.navigation_steps = 0
        self.metric_calls = 0

    def begin_episode(self, episode):
        if self.active and not self.done:
            raise RuntimeError('Current task has not stopped')
        self.episode = episode
        self.scene.apply_episode_state(episode.scene_config_path,episode.start_agent_state,episode.mid_object_states)
        self._positions = load_episode_semantic_positions(episode.scene_config_path)

    def begin_task(self, subtask, *, subtask_id, subtask_order, history_state):
        if self.active and not self.done:
            raise RuntimeError('Current task has not stopped')
        self.subtask, self.subtask_id, self.order = subtask,subtask_id,subtask_order
        self.history = history_state
        self._targets = resolve_subtask_target_positions(subtask.goal_entries,self._positions)
        self.active,self.done = True,False
        self._result = None
        self._start_distance = float(self.logger.subtask_explore_dist)
        self.request = self.dataset.task_request(self.sequence,self.episode,subtask,
            new_sequence=subtask_order==0,new_episode=subtask is self.episode.subtasks[0])
        return self.request

    def observe_at(self, pts, angle=None, rotation=None):
        return self.step(PoseAction(pts,angle,rotation))

    def execute(self, command):
        if not isinstance(command, SessionCommand):
            raise ValueError('Expected SessionCommand')
        operations = dict(pose=self.step, frontier=self.observe_frontier,
                          navigation=self.record_navigation, budget=self.path_budget_exceeded,
                          stop=self.stop, evaluate=self.evaluate)
        if command.operation not in operations:
            raise ValueError(f'Unsupported session operation: {command.operation}')
        return operations[command.operation](*command.arguments, **(command.keywords or {}))

    def observe_frontier(self, pts, view_dir, camera_tilt=0.0):
        if self.closed or (self.active and self.done):
            raise RuntimeError('Observation requested after stop/close')
        self.observation_calls += 1
        return self._frontier_capture(pts,view_dir,camera_tilt)

    def step(self, action):
        if self.closed or (self.active and self.done):
            raise RuntimeError('Pose action requested after stop/close')
        if not isinstance(action,PoseAction):
            raise ValueError('Expected PoseAction')
        position = np.asarray(action.position,dtype=np.float32)
        if position.shape!=(3,) or not np.isfinite(position).all():
            raise ValueError('Invalid observation position')
        if (action.angle is None)==(action.rotation is None):
            raise ValueError('Specify exactly one of angle and rotation')
        if action.angle is not None and not math.isfinite(action.angle):
            raise ValueError('Invalid heading')
        self.observation_calls += 1
        return self._capture(position,angle=action.angle,rotation=action.rotation)

    def record_navigation(self, tracker, planner, position):
        if not self.active or self.done:
            raise RuntimeError('Navigation outside active task')
        pos = np.asarray(position,dtype=np.float32)
        self.logger.log_step(pts_voxel=planner.habitat2voxel(pos)[:2])
        trace = getattr(tracker,'position_trace',None)
        if isinstance(trace,list) and (not trace or np.linalg.norm(np.asarray(trace[-1])[[0,2]]-pos[[0,2]])>0.0001):
            trace.append(pos.copy())
            lengths = getattr(tracker,'path_length_trace',None)
            if isinstance(lengths,list):
                lengths.append(float(self.logger.subtask_explore_dist))
        tracker.record_step(dict(total_steps=1,collision_steps=0,target_not_on_navmesh=False))
        self.navigation_steps += 1

    def path_budget_exceeded(self, budget_m):
        return math.isfinite(budget_m) and float(self.logger.subtask_explore_dist)-self._start_distance>budget_m+1e-6

    def stop(self, position):
        if not self.active:
            raise RuntimeError('No active task')
        self.done = True
        self._final = np.asarray(position,dtype=np.float32).copy()

    def evaluate(self, tracker):
        if not self.done:
            raise RuntimeError('Stop before evaluating')
        if self._result is not None:
            return self._result
        distance = calc_min_horizontal_distance(self._final,self._targets)
        ids = resolve_episode_goal_semantic_ids(self.episode.scene_config_path,self.subtask.goal_entries)
        history = evaluate_subtask_gt_history_retrieval(self.history,subtask_id=self.subtask_id,
            subtask_order=self.order,episode_index=self.episode.episode_index,
            goal_category=self.subtask.goal_category,target_semantic_ids=ids,
            current_positions=getattr(tracker,'position_trace',[]),
            current_path_lengths=getattr(tracker,'path_length_trace',[]),pathfinder=self.scene.pathfinder,
            current_target_positions_by_semantic_id=collect_episode_semantic_positions(self.episode.scene_config_path,ids),
            final_position=self._final,success_distance_m=float(self.cfg.success_distance),
            retrieval_radius_m=float(self.cfg.gt_history_retrieval_radius_m))
        self._result = dict(agent_target_distance=float(distance),success_by_distance=bool(distance<float(self.cfg.success_distance)),gt_history_retrieval=history)
        self.metric_calls += 1
        return self._result

    def close(self):
        if not self.closed:
            self.scene.get_observation = self._capture
            self.scene.get_frontier_observation = self._frontier_capture
            self.closed = True
