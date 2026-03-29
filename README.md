# 🚀 VLNTube

An end-to-end pipeline for generating Vision-Language Navigation (VLN) training data from indoor 3D scenes. Starting from raw USD scene files, VLNTube produces complete trajectory datasets with RGB/depth observations, discrete action sequences, and multi-granularity natural language instructions.

## 🔄 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  scene_graph/         Extract objects & spatial relationships       │
│                       from USD scenes                               │
│  Input:  USD files + room metadata                                  │
│  Output: object_dict.json, room_dict.json, category_dict.json       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ object positions, relationships
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  vistube/             Visual data generation (3 stages)             │
│                                                                     │
│  Stage 1: Sample walkable points per room (Monte Carlo + K-Medoids) │
│  Stage 2: Select goal objects from scene graph, plan discrete       │
│           A* paths (forward 0.25m, turn 15°), generate template     │
│           instructions                                              │
│  Stage 3: Replay paths in Isaac Sim, render RGB + depth sequences   │
│                                                                     │
│  Input:  scene_graph outputs + occupancy maps + USD scenes          │
│  Output: path waypoints, action sequences, rendered image sequences,│
│          template-based goal instructions                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ rendered sequences + goal instructions
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  instube/             Instruction generation via Gemini API         │
│                                                                     │
│  Step 1: Feed image sequences to Gemini → fine-grained nav         │
│          instructions (second-person imperative)                    │
│  Step 2: Generate targeted captions from goal reference images      │
│  Step 3: Fuse template instructions + captions → augmented         │
│          instructions in 3 styles (formal, natural, casual)         │
│                                                                     │
│  Input:  vistube rendered sequences + goal_inst.json                │
│  Output: inst_img_sequence.json, goal_inst_aug_enhance.json         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ fine + coarse instructions
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  datatube/            Format conversion                             │
│                                                                     │
│  Converts everything into InteriorNav training format:              │
│  - Parquet files (positions, orientations, actions)                 │
│  - NPY arrays (RGB + depth)                                        │
│  - JSON metadata with 50/50 fine/coarse instruction split           │
│                                                                     │
│  Input:  vistube + instube outputs                                  │
│  Output: InteriorNav-format dataset ready for training              │
└─────────────────────────────────────────────────────────────────────┘
```

## 📦 Data Download

Before running the pipeline, download the required scene data:

| Dataset | Contents | Link |
|---|---|---|
| **TataServices** | USD scene files | [Hugging Face](https://huggingface.co/datasets/Eyz/TataServices) |
| **TaTaMeta** | Scene metadata (collision maps, room layouts) | [Hugging Face](https://huggingface.co/datasets/Eyz/TaTaMeta) |

## ⚡ Quick Start

🎮 **New here?** Try [IAmGoodNavigator](https://github.com/william13077/IAmGoodNavigator) first — walk through our 3D scenes yourself, get Isaac Sim set up, and see what the generated data looks like in action!

```bash
# 0. Build scene graphs (requires Isaac Sim)
python -m scene_graph.summarizer -d /path/to/scenes -o scene_summary

# 1-3. Run visual data pipeline (edit config in run_pipeline.sh)
bash vistube/run_pipeline.sh

# 4. Generate instructions via Gemini API
export GOOGLE_API_KEY='your_key'
python instube/gemini_images_analyzer.py
python instube/gemini_aug_goal_image_enhance.py

# 5. Convert to training format
python datatube/convert_data.py
```

## 🤖 Modules

| Module | Description | Isaac Sim Required |
|---|---|---|
| [`scene_graph/`](scene_graph/README.md) | Extract objects and spatial relationships from USD scenes | Yes |
| [`vistube/`](vistube/README.md) | Sample walkable points, plan paths, render video sequences | Stage 1: No, Stages 2-3: Yes |
| [`instube/`](instube/README.md) | Generate and augment navigation instructions via Gemini API | No |
| [`datatube/`](datatube/README.md) | Convert to InteriorNav training format | No |

## 🗺️ Data Flow

```
USD scenes + metadata
        │
        ├──▶ scene_graph ──▶ object_dict.json (per scene)
        │                         │
        ▼                         ▼
  occupancy maps ──▶ vistube Stage 1 ──▶ sampled_points.json
                     vistube Stage 2 ──▶ paths (.npy) + actions (.json) + goal_inst.json
                     vistube Stage 3 ──▶ RGB/depth image sequences
                                              │
                              ┌────────────────┤
                              ▼                ▼
                     instube Step 1      instube Steps 2-3
                     (fine-grained       (coarse augmented
                      instructions)       instructions)
                              │                │
                              ▼                ▼
                         datatube ──▶ InteriorNav dataset
```

## 📋 Requirements

- **Isaac Sim** (for scene_graph, vistube stages 2-3)
- **Gemini API key** (for instube)
- **Python packages**: See each module's README for specific dependencies
