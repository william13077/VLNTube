# DataTube: VLN Data Format Converter

Converts VLN trajectory data (produced by [vistube](../vistube/) and [instube](../instube/)) into InteriorNav training format.

## What It Does

For each trajectory, the converter:
1. Loads RGB images, depth maps, action sequences, and path waypoints
2. Computes 3D positions and quaternion orientations (accounting for in-place turns)
3. Saves everything in InteriorNav format: Parquet files, NPY image arrays, and JSON metadata
4. Splits trajectories 50/50 into **fine-grained** (per-trajectory image-based instructions from instube) and **coarse-grained** (augmented goal instructions from instube)

## Usage

```bash
python datatube/convert_data.py
```

Edit the config variables at the top of `main()` to match your setup:

| Variable | Default | Description |
|---|---|---|
| `source_root` | `/mnt/6t/dataset/vlnverse` | Root directory containing raw scene folders |
| `target_root` | `/data/dataset/vlnverse/traj_data/vlnverse` | Output directory for InteriorNav-format data |
| `task_dir` | `goalnav_discrete` | Subdirectory name for task outputs under each scene |
| `seq_dir` | `sequence_discrete` | Subdirectory name for rendered sequences under task_dir |

## Input Structure

Expects vistube + instube outputs per scene:
```
<source_root>/<scene_id>/<task_dir>/
├── <seq_dir>/path_<goal>_<start>/   # RGB images + depth maps
│   ├── rgb_*.png
│   └── depth_*.npy
├── actions/actions_<goal>_<start>.json
├── npy/path_<goal>_<start>.npy
├── goal_inst_aug_enhance.json        # Coarse instructions (from instube)
└── inst/inst_img_sequence.json       # Fine instructions (from instube)
```

## Output Structure

```
<target_root>/<scene_id>/<goal>_<start>/
├── data/chunk-000/
│   └── episode_000000.parquet    # Positions, orientations, actions, progress
├── videos/chunk-000/
│   ├── observation.images.rgb/rgb.npy
│   └── observation.images.depth/depth.npy
└── meta/
    ├── episodes.jsonl            # Episode metadata + instruction
    ├── info.json                 # Dataset info (fps, encoding)
    └── tasks.jsonl               # Task type + instruction

<raw_data_root>/
├── all_fine_grained/             # Aggregated fine-grained episodes
│   ├── all_fine_grained.json
│   └── all_fine_grained.json.gz
└── all_coarse_grained/           # Aggregated coarse-grained episodes
    ├── all_coarse_grained.json
    └── all_coarse_grained.json.gz
```

## Dependencies

```bash
pip install numpy pandas Pillow opencv-python tqdm natsort pyarrow
```

## Notes

- **Reproducibility**: Uses `np.random.seed(42)` for deterministic fine/coarse splitting.
- **Error logging**: Skipped trajectories are logged to `conversion_errors.log` in the target directory.
- **Image resizing**: All RGB and depth frames are resized to 256x256.
- **Depth normalization**: Depth values (meters) > 1.0 are divided by 100.0; NaN/inf values are clamped. This could be a bug but works well.
