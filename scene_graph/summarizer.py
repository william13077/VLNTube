"""Build scene graphs from USD files with spatial relationship inference.

Extracts object meshes from a USD stage, assigns each object to a room based
on occupancy-map metadata, infers pairwise spatial relationships (above, below,
on, in, contain, near), and writes the results as JSON.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
from pathlib import Path

import natsort
import numpy as np
import open3d as o3d
import tqdm

# utils_v2 must be imported before pxr — it initializes IsaacSim's
# SimulationApp which makes the pxr module available.
from scene_graph.utils.utils_v2 import get_mesh_via_prim
from pxr import Usd

from scene_graph.utils.scene_graph_utils import (
    POINT_CLOUD_SAMPLE_COUNT,
    DEFAULT_EXCLUDED_SCOPES,
    ObjectBounds,
    get_room_infos,
    in_which_room,
    infer_spatial_relationship,
    strip_point_clouds,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def extract_objects(
    stage: Usd.Stage,
    room_infos: list[dict],
    room_names: list[str],
    excluded_scopes: tuple[str, ...] = DEFAULT_EXCLUDED_SCOPES,
) -> tuple[dict, dict, dict]:
    """Extract objects from the USD stage and assign each to a room.

    Walks ``/Root/Meshes`` in the USD hierarchy, samples a point cloud for
    each instance, computes its bounding box, and determines which room it
    belongs to.

    Args:
        stage: An opened USD stage.
        room_infos: Room boundary information from :func:`get_room_infos`.
        room_names: List of room names (keys from room_region.json).
        excluded_scopes: Scope names to skip (e.g. ``("Base",)``).

    Returns:
        ``(object_dict, room_dict, category_dict)``
    """
    object_dict: dict = {}
    room_dict: dict = {name: [] for name in room_names}
    category_dict: dict = {}

    meshes = stage.GetPrimAtPath("/Root/Meshes")
    scopes = meshes.GetChildren()

    for scope in scopes:
        scope_name = scope.GetName()
        if scope_name in excluded_scopes:
            continue

        for category in scope.GetChildren():
            category_name = category.GetName()
            instances = category.GetChildren()

            for instance in tqdm.tqdm(instances, desc=f"{scope_name}/{category_name}"):
                path_parts = str(instance.GetPath()).split("/")
                instance_id = f"{path_parts[-2]}/{path_parts[-1]}"

                try:
                    mesh = get_mesh_via_prim(instance, category)
                    point_cloud = mesh.sample_points_uniformly(
                        number_of_points=POINT_CLOUD_SAMPLE_COUNT,
                    )
                except Exception as e:
                    logger.warning("Skipping %s: %s", instance_id, e)
                    continue

                point_cloud = np.asarray(point_cloud.points)
                max_point = point_cloud.max(axis=0)
                min_point = point_cloud.min(axis=0)
                position = (max_point + min_point) / 2
                room_name = in_which_room(point_cloud, room_infos)

                object_dict[instance_id] = {
                    "instance_id": instance_id,
                    "category": category_name,
                    "scope": scope_name,
                    "room": room_name,
                    "position": position,
                    "min_points": min_point,
                    "max_points": max_point,
                    "point_cloud": point_cloud,
                    "nearby_objects": {},
                }

                room_dict[room_name].append(instance_id)
                category_dict.setdefault(category_name, []).append([instance_id])

    return object_dict, room_dict, category_dict


def compute_relationships(object_dict: dict, room_dict: dict) -> None:
    """Infer spatial relationships between all object pairs in each room.

    Updates ``object_dict`` in-place, populating the ``"nearby_objects"``
    field with ``{other_id: [relationship, distance]}`` entries.

    Args:
        object_dict: Object information as returned by :func:`extract_objects`.
        room_dict: Room-to-object mapping as returned by :func:`extract_objects`.
    """
    for room_name, object_ids in room_dict.items():
        for id_a, id_b in itertools.combinations(object_ids, 2):
            obj_a = object_dict[id_a]
            obj_b = object_dict[id_b]

            bounds_a = ObjectBounds(
                point_cloud=obj_a["point_cloud"],
                min_point=obj_a["min_points"],
                max_point=obj_a["max_points"],
            )
            bounds_b = ObjectBounds(
                point_cloud=obj_b["point_cloud"],
                min_point=obj_b["min_points"],
                max_point=obj_b["max_points"],
            )

            rel_a, rel_b, dist = infer_spatial_relationship(bounds_a, bounds_b)

            if rel_a is not None:
                obj_a["nearby_objects"][id_b] = [rel_b, dist]
                obj_b["nearby_objects"][id_a] = [rel_a, dist]


def save_scene_graph(
    object_dict: dict,
    room_dict: dict,
    category_dict: dict,
    output_dir: str | Path,
) -> None:
    """Write scene graph results to JSON files.

    Args:
        object_dict: Object information (point clouds will be stripped).
        room_dict: Room-to-object mapping.
        category_dict: Category-to-object mapping.
        output_dir: Directory to write output files into.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable = strip_point_clouds(object_dict)

    with open(output_dir / "object_dict.json", "w") as f:
        json.dump(serializable, f, indent=4)
    with open(output_dir / "room_dict.json", "w") as f:
        json.dump(room_dict, f, indent=4)
    with open(output_dir / "category_dict.json", "w") as f:
        json.dump(category_dict, f, indent=4)

    logger.info("Saved scene graph to %s", output_dir)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def summarize_scene(
    usd_path: str | Path,
    scene_name: str,
    metadata_dir: str | Path = "metadata",
    output_dir: str | Path = "scene_summary",
    excluded_scopes: tuple[str, ...] | None = None,
) -> None:
    """Build a scene graph from a USD file and save results as JSON.

    End-to-end pipeline that loads the USD stage and room metadata, extracts
    objects, infers spatial relationships, and writes JSON output.

    Args:
        usd_path: Path to the USD/USDA file.
        scene_name: Scene identifier used for metadata lookup and output naming.
        metadata_dir: Root directory containing per-scene metadata folders.
        output_dir: Root directory for output (scene_name subfolder is created).
        excluded_scopes: USD scopes to skip. Defaults to ``("Base",)``.
    """
    if excluded_scopes is None:
        excluded_scopes = DEFAULT_EXCLUDED_SCOPES

    # Fix random seed for deterministic point cloud sampling
    np.random.seed(42)
    o3d.utility.random.seed(42)

    metadata_dir = Path(metadata_dir)
    scene_output = Path(output_dir) / scene_name

    # Load USD stage
    stage = Usd.Stage.Open(str(usd_path))

    # Load room metadata
    with open(metadata_dir / scene_name / "room_region.json") as f:
        room_regions = json.load(f)
    freemap = np.load(str(metadata_dir / scene_name / "freemap.npy"))

    room_infos = get_room_infos(room_regions, freemap)
    room_names = list(room_regions.keys())

    # Pipeline
    object_dict, room_dict, category_dict = extract_objects(
        stage, room_infos, room_names, excluded_scopes,
    )
    compute_relationships(object_dict, room_dict)
    save_scene_graph(object_dict, room_dict, category_dict, scene_output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build scene graphs from USD files.",
    )
    parser.add_argument(
        "-u", "--usd", type=str, default=None,
        help="Path to a single USD file.",
    )
    parser.add_argument(
        "-n", "--scene-name", type=str, default=None,
        help="Scene name (used for metadata lookup and output naming).",
    )
    parser.add_argument(
        "-d", "--dirs", type=str, default=None,
        help="Directory containing multiple scene folders for batch processing.",
    )
    parser.add_argument(
        "--metadata-dir", type=str, default="metadata",
        help="Root directory for per-scene metadata (default: metadata).",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="scene_summary",
        help="Root output directory (default: scene_summary).",
    )
    parser.add_argument(
        "--excluded-scopes", nargs="*", default=["Base"],
        help="USD scopes to skip (default: Base).",
    )
    parser.add_argument(
        "--usd-filename", type=str, default="start_result_navigation.usd",
        help="USD filename within each scene folder (batch mode, default: start_result_navigation.usd).",
    )
    args = parser.parse_args()

    excluded = tuple(args.excluded_scopes)

    if args.dirs:
        # Batch mode: process all scene folders in the directory
        entries = os.listdir(args.dirs)
        scene_dirs = natsort.natsorted(
            d for d in entries if os.path.isdir(os.path.join(args.dirs, d))
        )
        for scene_name in scene_dirs:
            output_path = os.path.join(args.output_dir, scene_name, "object_dict.json")
            if os.path.exists(output_path):
                logger.info("Skipping %s: already processed", scene_name)
                continue
            usd_path = os.path.join(args.dirs, scene_name, args.usd_filename)
            summarize_scene(
                usd_path, scene_name,
                metadata_dir=args.metadata_dir,
                output_dir=args.output_dir,
                excluded_scopes=excluded,
            )

    elif args.usd and args.scene_name:
        # Single scene mode
        summarize_scene(
            args.usd, args.scene_name,
            metadata_dir=args.metadata_dir,
            output_dir=args.output_dir,
            excluded_scopes=excluded,
        )

    else:
        parser.error("Provide either --dirs for batch mode, or both --usd and --scene-name.")
