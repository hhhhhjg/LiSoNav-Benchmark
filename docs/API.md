# Python API

The runner loads an agent class from the `--agent module:ClassName` argument. The class must support a no-argument constructor and four methods:

| Method | When it is called | Input or result |
| --- | --- | --- |
| `reset(task)` | At the start of each navigation task | A `TaskRequest` |
| `act(observation)` | Until the task ends | Return one `Action` |
| `on_task_end(result)` | After the task is scored | A small dictionary of navigation results |
| `close()` | When the run finishes | Release agent resources |

The runner calls the environment in this order:

```text
begin_sequence(sequence)
  begin_episode(episode)
    reset_task(subtask) -> task, observation
    agent.reset(task)
    while observation.done is false:
        observation = step(agent.act(observation))
    result = evaluate()
    agent.on_task_end(navigation_result)
close()
```

`begin_sequence` clears the evaluator's history. `begin_episode` loads the next object layout, rebuilds the NavMesh, and sets the recorded starting pose. Within an episode, the next task starts from the previous task's final pose. Each task has fresh path, collision, and action counters.

## TaskRequest

`TaskRequest` includes sequence, scene, episode, and task identifiers; `goal_type`, `goal_category`, `instruction`, `target_description`, and `target_surface_description`; an optional `reference_image`; and `new_sequence` and `new_episode` flags. The reference image is an RGB array or `None` when it is unavailable or `--no-reference-images` is set.

The agent does not receive target coordinates, target semantic IDs, layout files, runtime bundles, or ground-truth history. These remain inside the evaluator.

## Observation

| Field | Meaning |
| --- | --- |
| `rgb` | RGB image, H × W × 3, `uint8` |
| `depth` | Depth image in metres, H × W, `float32` |
| `position` | Agent position in Habitat world coordinates, XYZ metres |
| `rotation_xyzw` | Agent quaternion `[x, y, z, w]` |
| `camera_intrinsics` | 3 × 3 camera intrinsic matrix |
| `camera_pose` | 4 × 4 transform from depth-camera to world coordinates |
| `path_length_m` | Distance travelled during the current task in the XZ plane |
| `action_count` | Accepted agent commands, including `stop` |
| `collided` | Whether the latest non-stop command included a collision |
| `done` | Whether the task has ended |
| `stop_reason` | `agent_stop`, `path_budget_exceeded`, or `max_actions_exceeded`; otherwise `None` |

The world Y axis points upward. The camera looks along its local negative Z axis. Returned arrays are copies, so modifying them does not change simulator state.

## Actions

```python
from lisonav import Action

Action("move_forward", 1.0)  # metres
Action("turn_right", 90.0)   # degrees
Action("turn_left", 90.0)
Action("stop")
```

The default forward amount is 0.25 m and the default turn amount is 90 degrees. Forward amounts must be finite values in `(0, 10]` metres; turn amounts must be finite values in `(0, 360]` degrees. `stop` does not accept an amount.

Forward commands are executed in steps of at most 0.25 m. After each step, the evaluator records the observation, movement, and collision status and applies the path budget. A forced stop interrupts the rest of the command. The action record distinguishes requested `amount`, attempted `executed_amount`, and actual `path_delta_m`.

`step` cannot be called after termination. `evaluate` can only be called after termination and returns the same result on repeated calls. The current task must be evaluated before the next task or layout begins.

## Results and history

`evaluate()` returns navigation metrics, termination reason, path budget, final position, action and position traces, and `gt_history_retrieval` details. The default runner writes complete task results to `results.json`, aggregate metrics to `summary.json`, and configuration to `run_config.json`. It also writes `<sequence>__<scene>_ep_<id>/gt_history_retrieval_state.json` with the evaluator's complete ground-truth history state.

`on_task_end` receives only the task ID, stop information, path length, action count, success, SPL, and collision ratio. Ground-truth history state is never passed to the agent.

For a custom evaluation driver, use `Dataset`, `NavigationEnvironment`, and `BenchmarkConfig` from `lisonav`. Call `begin_sequence`, `begin_episode`, `reset_task`, `step`, and `evaluate` in the order shown above. `env.aggregate()` returns accumulated scores; `env.history_state()` returns an evaluator-only copy of the current sequence history. Call `env.close()` when finished. This API runs in the same Python process as the agent; isolate untrusted submissions in a separate process or container.
