# Scene Graph

Build scene graphs from USD files with spatial relationship inference.

Given a USD scene and room metadata, this tool extracts every object instance, assigns it to a room, infers pairwise spatial relationships, and outputs structured JSON.

## Structure

```
scene_graph/
├── summarizer.py                # Main pipeline and CLI entry point
└── utils/
    ├── utils_v2.py              # USD mesh extraction (IsaacSim + Open3D)
    ├── point_cloud_utils.py     # Point cloud distance and containment tests
    └── scene_graph_utils.py     # Room assignment, spatial inference, constants
```

## Pipeline

1. **Extract objects** — Walk `/Root/Meshes` in the USD stage, sample 7000 points per mesh, compute bounding boxes
2. **Assign rooms** — Match each object to a room using 2D overlap with room boundary polygons from metadata
3. **Infer relationships** — For all object pairs within the same room, determine spatial relationships:
   - `above` / `below` — vertical separation with 2D overlap
   - `on` / `below` — support (object resting on another)
   - `in` / `contain` — containment
   - `near` — close but no overlap
4. **Save** — Write `object_dict.json`, `room_dict.json`, `category_dict.json`

## Requirements

- NVIDIA Isaac Sim (provides `pxr`, `isaacsim`)
- Python packages: `open3d`, `numpy`, `shapely`, `scipy`, `scikit-learn`, `tqdm`, `natsort`

## Usage

**Single scene:**

```bash
python -m scene_graph.summarizer \
  -u /path/to/scene.usd \
  -n scene_name \
  -o scene_summary
```

**Batch (all scenes in a directory):**

```bash
python -m scene_graph.summarizer \
  -d /path/to/scenes_dir \
  -o scene_summary
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-u, --usd` | — | Path to a single USD file |
| `-n, --scene-name` | — | Scene name (metadata lookup + output naming) |
| `-d, --dirs` | — | Directory of scene folders (batch mode) |
| `-o, --output-dir` | `scene_summary` | Output directory |
| `--metadata-dir` | `metadata` | Per-scene metadata root |
| `--excluded-scopes` | `Base` | USD scopes to skip |
| `--usd-filename` | `start_result_navigation.usd` | USD filename in each scene folder (batch) |

## Filtering Objects of Interest

By default, all object instances under `/Root/Meshes` are processed (except those in excluded scopes). You can customize which objects to include by filtering the output JSON or by modifying the `excluded_scopes` argument to skip entire USD scopes. For finer control, filter `object_dict.json` by category name after generation — for example, keep only furniture categories and discard structural elements like `wall`, `floor`, `ceiling`.

## Notes

**Point cloud sampling and reproducibility.** Each mesh is sampled to 7000 points using `Open3D`'s uniform sampling. This is stochastic — different samples produce slightly different bounding boxes and distances. The effect is most noticeable on large meshes (e.g. floors, walls) and on object pairs whose distance is near a threshold boundary (e.g. ~0.5m for "close", ~0.06m for "touching"). A fixed random seed (`numpy` and `Open3D`) is set before each scene to ensure deterministic results across runs.

**Geometry-only relationships.** The spatial relationships describe pure geometric relations (bounding box overlap, point cloud distance, vertical ordering). They do not encode semantic meaning — for example, the tool may report "wall on floor" because the wall mesh geometrically rests on top of the floor mesh. This is correct in terms of geometry but may seem counterintuitive semantically. Downstream consumers should apply domain-specific filtering if needed.

## Input

Each scene requires metadata under `<metadata-dir>/<scene-name>/`:

- `room_region.json` — Room boundary pixel coordinates
- `freemap.npy` — Occupancy grid mapping pixels to world coordinates

## Output

Each scene produces three JSON files under `<output-dir>/<scene-name>/`:

- **`object_dict.json`** — Per-object info: category, room, position, bounding box, spatial relationships
- **`room_dict.json`** — Room name → list of object instance IDs
- **`category_dict.json`** — Category name → list of object instance IDs
