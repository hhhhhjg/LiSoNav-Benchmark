# Evaluation protocol

## Dataset and task sequence

Each lifelong sequence contains multiple object-layout episodes. An episode contains navigation tasks executed in order. At the start of a sequence, the evaluator clears its history. At the start of an episode, it loads the episode's object layout, rebuilds the NavMesh, and sets the recorded agent start pose. Subsequent tasks in the same episode begin at the previous task's final pose and heading. The evaluator resets path, collision, and action counters for each task while retaining sequence history.

A task ends when the agent calls `stop`, exceeds the path budget, or reaches the action limit. Metrics are calculated after termination. The default runner records each task before advancing.

## Navigation metrics

**Success rate (SR).** The evaluator takes the minimum horizontal XZ distance from the final agent position to the valid target positions in the current layout. A task succeeds when this distance is strictly less than 1.5 m. Target positions are resolved from the current layout, with task annotations used when needed. The same rule applies to agent stops and forced stops.

**Success weighted by path length (SPL).** For a successful task, SPL is `shortest_distance / max(shortest_distance, travelled_distance)`; otherwise it is zero. A zero denominator also produces zero. The shortest distance starts at the task's actual start position. The evaluator uses a NavMesh geodesic distance when available and falls back to horizontal distance when no valid path is found. The result includes the chosen distance source and diagnostic fields.

**Path budget.** By default, the budget is ten times the shortest distance from the task start to a target. The evaluator accumulates actual XZ movement; turning does not add distance. After each movement step, a task is stopped when `travelled_distance > budget + 1e-6`. A single step may cross the boundary. `--path-budget-m` overrides the budget for a run.

**Collision ratio.** This is the number of executed movement or turn steps that collided, divided by all executed movement or turn steps. `stop` is excluded. `target_not_on_navmesh_ratio` is included in results; it remains zero for the available primitive action types.

**Action limit.** The default limit is 1,000 accepted commands per task. Reaching it ends the task with `max_actions_exceeded`.

**Frame counts.** `n_total_frames` counts the initial observation and subsequent observations recorded after primitive actions. `n_filtered_frames` is a reserved result field and is zero for this runner.

## Ground-truth history retrieval

The evaluator reads its internal semantic image at task start and after each executed action step. It combines semantic labels, depth, camera geometry, and the current layout to record visible target instances. These internal labels are never included in agent observations.

A target must occupy at least `0.00025` of the full image, equivalent to 0.025% of its pixels. At least half of those target pixels must lie within the central region spanning 80% of the image width and height. Where multiple object instances share a semantic ID, depth and geometry are used to associate visible pixels with an instance.

For each target in the current task, only observations from earlier tasks in the same sequence count as history. History retrieval compares previously observed agent positions with the current task's position trace using a 1.5 m XZ radius. Results include eligible and retrieved instance counts, per-instance details, success rates, and any/all task-level outcomes.

When a previously observed target has moved, the evaluator also records whether the agent revisits its historical location and then reaches the target's current location. Relocation results include success and SPL based on the route from the revisit point to the current target. Aggregate history metrics use the denominators recorded for eligible and relocated instances.

## Output files

The runner writes `run_config.json`, per-task `results.json`, and aggregate `summary.json`. Each sequence directory also contains `gt_history_retrieval_state.json`, including tracked semantic IDs, past observations, and retrieval results by task. JSON uses `null` for unavailable non-finite distances. An output directory with an existing `results.json` is not overwritten.
