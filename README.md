# LiSoNav-Benchmark

This is the official repository for **Lifelong Small-object Navigation in Changing Object Layouts**.

LiSoNav-Benchmark is an independent benchmark for lifelong small-object navigation. It provides interfaces for loading tasks, interacting with a Habitat simulator, executing actions, and calculating metrics. A mock agent demonstrates how to connect another navigation method. The evaluation runner switches between object layouts and tasks in sequence and saves both per-task and aggregate results.

## Installation

Use Linux and Python 3.9. Install the benchmark package from the repository root:

```bash
python -m pip install -e .
```

This installs the package and its Python dependencies in editable mode, so `python -m lisonav.run` can import it while source edits take effect without reinstalling. Run the command from the repository root.

Running real scenes also requires Habitat-Sim 0.2.5, `numpy-quaternion`, and an NVIDIA/EGL rendering environment. The [3D-Mem installation guide](https://github.com/UMass-Embodied-AGI/3D-Mem#installation) describes how to prepare a Habitat environment. Then install the benchmark's optional dependencies:

```bash
python -m pip install -e '.[habitat]'
```

The optional dependency group does not install Habitat-Sim itself. The mock backend does not require Habitat or scene assets.

## Scene and object assets

Place the assets under `LiSoNav-Eval/assets/data/` with the structure below. Alternatively, pass `--assets` to point to another directory with the same structure.

```text
LiSoNav-Eval/assets/data/
├── hm3d/val/
│   ├── hm3d_annotated_val_basis.scene_dataset_config.json
│   └── <scene>/<scene>.basis.glb
└── ycb_and_hssd/
    ├── configs/
    ├── meshes/
    ├── collision_meshes/
    ├── views_center_640/
    ├── name2description.json
    └── source_dataset_map.json
```

- **HM3D v0.2 validation scenes:** Obtain them through the [official HM3D access process](https://github.com/matterport/habitat-matterport-3dresearch) and place them under `hm3d/val/`. HM3D scenes are license controlled and are not distributed with this benchmark.
- **Movable-object assets:** Download `lisonav_ycb_hssd_assets.tar.gz` from [Baidu Netdisk](https://pan.baidu.com/s/1VK2DCOrc1YcFRzTTkhzmLg?pwd=liso) using access code `liso`. Extract it so that the resulting `ycb_and_hssd/` directory has the structure shown above.

See [the asset README](LiSoNav-Eval/assets/README.md) for more information about the required files.

## Dataset parts

The annotations are organized into five parts, `LiSoNav-Eval/lifelong_navigation_sequences/part00` through `part04`, totaling **145 sequences, 500 layouts, and 1,199 navigation tasks**. The runner uses `part00` by default. Scene and movable-object assets are obtained separately as described above.

## Run the mock agent

From the repository root, run:

```bash
python -m lisonav.run \
  --backend mock \
  --no-reference-images \
  --limit-tasks 2 \
  --output results/mock_example
```

The mock backend uses the dataset's task annotations and synthetic observations to exercise the evaluation interface. The mock agent serves as an example for integrating another method. Once Habitat and the assets are available, the same agent can run in real scenes:

```bash
python -m lisonav.run \
  --backend habitat \
  --limit-sequences 1 \
  --gpu-device-id 0 \
  --output results/habitat_example
```

Select another part with `--dataset`:

```bash
python -m lisonav.run \
  --dataset LiSoNav-Eval/lifelong_navigation_sequences/part01 \
  --backend habitat \
  --gpu-device-id 0 \
  --output results/part01
```

Omit `--limit-tasks` or `--limit-sequences` to run the entire selected part. Use a new output directory for each run; the runner will not overwrite an existing `results.json`.

## Connect your own agent

Use [agents/mock_agent.py](agents/mock_agent.py) as an interface example. Implement a Python class that can be constructed without arguments and provides `reset(task)`, `act(observation)`, `on_task_end(result)`, and `close()`. Pass its module and class name through `--agent`:

```bash
python -m lisonav.run \
  --agent your_package.your_agent:YourAgent \
  --backend habitat \
  --gpu-device-id 0 \
  --output results/your_agent
```

The runner calls `reset` at the start of each task; the task indicates when a new sequence or layout begins. `act` receives RGB-D observations, camera parameters, pose, and run state, and returns an action. Target coordinates and ground-truth history used for scoring remain inside the evaluator and are not passed to the agent. See the [API documentation](docs/API.md) for the action types and complete call sequence, and the [evaluation protocol](docs/EVALUATION_PROTOCOL.md) for metric definitions.

Each run writes `run_config.json`, `results.json`, and `summary.json`. The per-task results include navigation metrics and history-retrieval details. Each sequence also has a `gt_history_retrieval_state.json` file containing the complete ground-truth history state.

## Licenses and acknowledgements

Benchmark code is released under the [MIT License](LICENSE). LiSoNav-Eval metadata and annotations are released under [CC BY-NC 4.0](LiSoNav-Eval/LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for notices covering adapted evaluation components.

HM3D, YCB/HSSD, and other external assets retain their respective licenses and terms of use. The movable-object pack includes HSSD-derived assets licensed under CC BY-NC 4.0. See the [asset attribution](LiSoNav-Eval/assets/ASSET_ATTRIBUTION.md) for details.
