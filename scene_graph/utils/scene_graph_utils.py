"""Room assignment and spatial relationship inference utilities for scene graphs."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon

from scene_graph.utils.point_cloud_utils import calculate_distance_between_two_point_clouds, is_inside

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Spatial relationship thresholds
DISTANCE_CLOSE_THRESHOLD = 0.5          # Max distance (m) to consider objects "close"
MAX_TOUCHING_DISTANCE = 0.06            # Max distance (m) to be considered touching
MIN_ABOVE_BELOW_DISTANCE = 0.05         # Min vertical gap (m) for above/below
MAX_SUPPORTING_AREA_RATIO = 1.5         # Max area ratio for support relationship
MIN_SUPPORTED_AREA_RATIO = 0.3          # Min intersection ratio for support
MIN_ABOVE_BELOW_AREA_RATIO = 0.2        # Min 2D overlap ratio for above/below
INSIDE_PROPORTION_THRESHOLD = 0.5       # Fraction of points inside hull for "in"
Z_DIST_EXPANSION_FACTOR = 0.15          # Bbox expansion for diagonal above/below
NEAR_IOU_THRESHOLD = 0.001              # Below this 2D IoU => "near"
Z_CENTER_EPSILON = 0.01                 # Vertical center sanity check (m)

# Point cloud sampling
POINT_CLOUD_SAMPLE_COUNT = 7000

# Room assignment
ROOM_OVERLAP_THRESHOLD = 0.1            # Min overlap ratio before falling back to nearest

# Default scopes to exclude from processing
DEFAULT_EXCLUDED_SCOPES = ("Base",)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ObjectBounds:
    """Bounding information for a single object instance."""

    point_cloud: np.ndarray
    min_point: np.ndarray
    max_point: np.ndarray


# ---------------------------------------------------------------------------
# Room assignment
# ---------------------------------------------------------------------------

def pixel_to_coord(
    freemap: np.ndarray,
    pixel: tuple[int, int],
) -> tuple[float, float]:
    """Convert a pixel index to world coordinates using the freemap lookup table.

    Args:
        freemap: Occupancy grid whose first row/column encode world coordinates.
        pixel: ``(row, col)`` pixel index.

    Returns:
        ``(x, y)`` world coordinates.
    """
    i, j = pixel
    x = float(freemap[i, 0])
    y = float(freemap[0, j])
    return x, y


def get_room_infos(
    room_regions: dict[str, list[list[int]]],
    freemap: np.ndarray,
) -> list[dict]:
    """Build a list of room info dicts with world-space boundary polygons.

    Args:
        room_regions: Mapping from room name to list of pixel boundary points.
        freemap: Occupancy grid for coordinate conversion.

    Returns:
        List of dicts, each with ``"room_name"`` and ``"shape"`` (list of
        ``(x, y)`` world coordinates).
    """
    room_infos = []
    for room_name, pixels in room_regions.items():
        # Note: coordinates are swapped (y, x) to align the room polygon
        # with the world-space coordinate system used by the USD meshes.
        points = [pixel_to_coord(freemap, p)[::-1] for p in pixels]
        room_infos.append({"room_name": room_name, "shape": points})
    return room_infos


def in_which_room(
    point_cloud: np.ndarray,
    room_infos: list[dict],
) -> str:
    """Determine which room an object belongs to based on point cloud overlap.

    The room with the highest 2D overlap ratio is selected.  If no room
    exceeds the overlap threshold, the nearest room is chosen instead.

    Args:
        point_cloud: ``(N, 3)`` array of object points in world coordinates.
        room_infos: Room information as returned by :func:`get_room_infos`.

    Returns:
        Name of the assigned room.
    """
    total_points = len(point_cloud)
    overlap_ratios = []
    distances = []

    for room in room_infos:
        polygon = Polygon(room["shape"])
        pcd_2d = point_cloud[..., :2]
        multi_point = MultiPoint(pcd_2d)

        intersection = polygon.intersection(multi_point)
        if isinstance(intersection, Point):
            count = 0
        else:
            count = len(intersection.geoms)

        overlap_ratios.append(count / total_points)
        distances.append(polygon.distance(multi_point.convex_hull))

    best_idx = int(np.argmax(overlap_ratios))
    if overlap_ratios[best_idx] < ROOM_OVERLAP_THRESHOLD:
        best_idx = int(np.argmin(distances))

    return room_infos[best_idx]["room_name"]


# ---------------------------------------------------------------------------
# Spatial relationship inference
# ---------------------------------------------------------------------------

def iou_2d_via_boundaries(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> tuple[float, list[float], list[float]]:
    """Compute 2D IoU and area ratios from axis-aligned bounding boxes.

    Args:
        min_a: Min corner of box A ``(x, y, ...)``.
        max_a: Max corner of box A.
        min_b: Min corner of box B.
        max_b: Max corner of box B.

    Returns:
        Tuple of ``(iou, [inter/area_a, inter/area_b], [area_a/area_b, area_b/area_a])``.
    """
    x_overlap = max(0, min(max_a[0], max_b[0]) - max(min_a[0], min_b[0]))
    y_overlap = max(0, min(max_a[1], max_b[1]) - max(min_a[1], min_b[1]))
    inter_area = x_overlap * y_overlap

    area_a = (max_a[0] - min_a[0]) * (max_a[1] - min_a[1])
    area_b = (max_b[0] - min_b[0]) * (max_b[1] - min_b[1])

    union = area_a + area_b - inter_area
    if union == 0 or area_a == 0 or area_b == 0:
        return 0.0, [0.0, 0.0], [0.0, 0.0]

    iou = inter_area / float(union)
    i_ratios = [inter_area / float(area_a), inter_area / float(area_b)]
    a_ratios = [area_a / area_b, area_b / area_a]

    return iou, i_ratios, a_ratios


def infer_spatial_relationship(
    a: ObjectBounds,
    b: ObjectBounds,
) -> tuple[str | None, str | None, float]:
    """Infer the spatial relationship between two objects.

    The decision follows a priority chain:
    1. Too far apart → ``(None, None, dist)``
    2. Above / below (vertical separation with 2D overlap)
    3. Near (no 2D overlap)
    4. In / contain (one inside the other)
    5. On / below (support relationship with vertical sanity check)
    6. Otherwise → ``(None, None, dist)``

    Args:
        a: Bounds of the first object.
        b: Bounds of the second object.

    Returns:
        ``(rel_a_to_b, rel_b_to_a, distance)`` where the relationship strings
        describe A's relation to B and B's relation to A respectively.
    """
    dist = calculate_distance_between_two_point_clouds(a.point_cloud, b.point_cloud)

    if dist > DISTANCE_CLOSE_THRESHOLD:
        return None, None, dist

    a_bottom_b_top_dist = b.min_point[2] - a.max_point[2]
    a_top_b_bottom_dist = a.min_point[2] - b.max_point[2]

    # --- Above / below ---
    if a_bottom_b_top_dist > 0 or a_top_b_bottom_dist > 0:
        z_dist = max(a_bottom_b_top_dist, a_top_b_bottom_dist)
        expansion = Z_DIST_EXPANSION_FACTOR * z_dist
        _, i_ratios, _ = iou_2d_via_boundaries(
            a.min_point - expansion, a.max_point + expansion,
            b.min_point - expansion, b.max_point + expansion,
        )
        i_target_ratio, i_anchor_ratio = i_ratios

        if (a_bottom_b_top_dist > MIN_ABOVE_BELOW_DISTANCE
                and max(i_anchor_ratio, i_target_ratio) > MIN_ABOVE_BELOW_AREA_RATIO):
            return "below", "above", dist
        if (a_top_b_bottom_dist > MIN_ABOVE_BELOW_DISTANCE
                and max(i_anchor_ratio, i_target_ratio) > MIN_ABOVE_BELOW_AREA_RATIO):
            return "above", "below", dist

    # --- 2D IoU analysis ---
    iou, i_ratios, a_ratios = iou_2d_via_boundaries(
        a.min_point, a.max_point, b.min_point, b.max_point,
    )
    i_target_ratio, i_anchor_ratio = i_ratios
    target_anchor_area_ratio, anchor_target_area_ratio = a_ratios

    # Near (no overlap)
    if iou < NEAR_IOU_THRESHOLD:
        return "near", "near", dist

    # --- In / contain ---
    if dist < MAX_TOUCHING_DISTANCE:
        if is_inside(src_pts=a.point_cloud, target_pts=b.point_cloud,
                     thresh=INSIDE_PROPORTION_THRESHOLD):
            return "in", "contain", dist
        if is_inside(src_pts=b.point_cloud, target_pts=a.point_cloud,
                     thresh=INSIDE_PROPORTION_THRESHOLD):
            return "contain", "in", dist

    # --- On / support ---
    a_supported_by_b = (
        dist < MAX_TOUCHING_DISTANCE
        and i_target_ratio > MIN_SUPPORTED_AREA_RATIO
        and abs(a_top_b_bottom_dist) <= MAX_TOUCHING_DISTANCE
        and target_anchor_area_ratio < MAX_SUPPORTING_AREA_RATIO
    )
    a_supporting_b = (
        dist < MAX_TOUCHING_DISTANCE
        and i_anchor_ratio > MIN_SUPPORTED_AREA_RATIO
        and abs(a_bottom_b_top_dist) <= MAX_TOUCHING_DISTANCE
        and anchor_target_area_ratio < MAX_SUPPORTING_AREA_RATIO
    )

    # Z-center sanity check: reject labels that contradict vertical ordering
    center_a_z = (a.min_point[2] + a.max_point[2]) / 2
    center_b_z = (b.min_point[2] + b.max_point[2]) / 2
    if a_supported_by_b and not (center_a_z - center_b_z > Z_CENTER_EPSILON):
        a_supported_by_b = False
    if a_supporting_b and not (center_b_z - center_a_z > Z_CENTER_EPSILON):
        a_supporting_b = False

    if a_supported_by_b:
        return "on", "below", dist
    if a_supporting_b:
        return "below", "on", dist

    return None, None, dist


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def strip_point_clouds(object_dict: dict) -> dict:
    """Prepare object dict for JSON serialization by removing point clouds.

    Creates a new dict (does not mutate the original) with ``point_cloud``
    fields removed and numpy arrays converted to Python lists.

    Args:
        object_dict: Mapping from instance ID to object info dicts.

    Returns:
        A JSON-serializable copy of *object_dict*.
    """
    return {
        obj_id: {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in info.items()
            if k != "point_cloud"
        }
        for obj_id, info in object_dict.items()
    }
