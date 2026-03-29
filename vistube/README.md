# VLNTube: VLN Data Generation Pipeline

A three-stage pipeline for generating Vision-Language Navigation (VLN) training data from indoor 3D scenes.

## Pipeline Overview

```
Stage 1: Sample Walkable Points
        │  For each room, sample representative navigable positions
        │  using Monte Carlo sampling + K-Medoids clustering.
        ▼
Stage 2: Generate Goals & Discrete Paths
        │  Find bidirectionally unique object relationships in the scene graph,
        │  then plan discrete-action paths (forward 0.25m, turn 15°) from
        │  sampled start points to goal objects. Generates natural language
        │  instructions for each goal using template-based NLP.
        ▼
Stage 3: Render Navigation Videos
        │  Replay discrete paths in Isaac Sim, capturing RGB and depth
        │  images at each step to produce navigation video sequences.
        ▼
    Output: per-scene folders with paths (.npy), action sequences (.json),
            instructions (goal_inst.json), and rendered image sequences.
```

## Quick Start

```bash
# Run the full pipeline (edit shared config in run_pipeline.sh first)
bash vlntube/run_pipeline.sh

# Or run individual stages:
python -m vlntube.stage1_sample_walkable --dataroot /path/to/scenes --metaroot /path/to/metadata
python -m vlntube.stage2_generate_goals /path/to/scene_dir --dataroot ... --task-dir goalnav_discrete
python -m vlntube.stage3_render_video /path/to/scene_dir --dataroot ... --task-dir goalnav_discrete
```

## Shared CLI Arguments

All paths are configured via CLI arguments (with defaults). The shell script `run_pipeline.sh` defines them once and passes to all stages:

| Argument | Description | Used by |
|---|---|---|
| `--dataroot` | Root dir containing scene folders (occupancy maps, sampled points, outputs) | Stage 1, 2, 3 |
| `--metaroot` | Root dir containing scene metadata (freemap.npy, room_region.json) | Stage 1, 2, 3 |
| `--usd-root` | Root dir containing USD scene files for Isaac Sim | Stage 2, 3 |
| `--scene-graph` | Root dir containing scene graph data (object_dict.json) | Stage 2 |
| `--task-dir` | Subdirectory name for task outputs (e.g. `goalnav_discrete`) | Stage 2, 3 |
| `--seq-dir` | Subdirectory name for rendered video sequences | Stage 3 |
| `scene_dir` | (positional, optional) Path to a specific scene directory | Stage 2, 3 |

## Directory Structure

```
vlntube/
├── run_pipeline.sh              # Shell script to run all stages with shared config
├── stage1_sample_walkable.py    # Stage 1: walkable point sampling
├── stage2_generate_goals.py     # Stage 2: goal selection + discrete path planning
├── stage3_render_video.py       # Stage 3: video rendering in Isaac Sim
├── tube_utils.py                # Shared utilities (deduplicated across stages)
├── path_utils.py                # Path processing, smoothing, visualization
├── path_finder.py               # Time-limited bidirectional A* pathfinding
├── discrete_path_planner.py     # Discrete A* planner (forward/turn action space)
├── find_unique_objects.py       # Scene graph analysis for unique object relationships
└── goal_gen/                    # Natural language instruction generation
    ├── gen_goal_inst.py         #   Instruction generator (spaCy-based verb conjugation)
    ├── template.py              #   Sentence templates
    ├── target_action.py         #   Object-to-action mapping
    └── action_category.py       #   Action category definitions
```

## Output Structure

For each scene, stage 2 produces:

```
<dataroot>/<scene_id>/<task_dir>/
├── goal_inst.json               # Goal instructions (10 templates per goal)
├── all_action_sequences.json    # All action sequences grouped by goal
├── npy/                         # Path waypoints: path_<goal>_<attempt>.npy
├── vis/                         # Path visualizations on occupancy map
├── ref/                         # Reference images of goal objects
└── actions/                     # Per-path action sequences with metadata
```

Stage 3 adds rendered sequences under `<task_dir>/<seq_dir>/`.

## Notes

- **Isaac Sim**: Stages 2 and 3 require Isaac Sim. Stage 1 runs standalone.
- **Reproducibility**: Both `random` and `np.random` are seeded. The `np.random` seed is critical because `find_endpoint_in_arc` randomly samples viewpoints; without it, different viewpoints cause path planning to succeed/fail differently in narrow areas, which shifts the `random` module's state via control flow divergence, ultimately changing which goals get selected downstream.
- **Discrete action space**: Forward 0.25m, turn left/right 15°, stop. These parameters can be overridden in stage 2's config section.
