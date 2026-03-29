from isaacsim import SimulationApp
CONFIG = {"headless": True}
simulation_app = SimulationApp(CONFIG)
import open3d as o3d
from pxr import Usd, UsdGeom, Sdf, Gf, Vt, UsdShade
import numpy as np
from copy import deepcopy
import os
import tqdm
import json
from shapely.geometry import MultiPoint, Point, Polygon


def _collect_mesh_points_world(prim, xform_cache):
    """Recursively collect mesh vertices under prim, transformed to world space
    using UsdGeom.XformCache (respects xformOpOrder, all ancestors, pivots, etc.)."""
    points_world = []

    if prim.IsA(UsdGeom.Mesh):
        mesh_name = str(prim.GetPath()).split("/")[-1]
        if mesh_name != 'SM_Dummy':
            raw_points = prim.GetAttribute("points").Get()
            if raw_points is not None and len(raw_points) > 0:
                local_to_world = xform_cache.GetLocalToWorldTransform(prim)
                mat = np.array(local_to_world).T  # Gf.Matrix4d is row-major, numpy @ expects column-major
                for p in raw_points:
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                    ph = np.array([x, y, z, 1.0])
                    pw = mat @ ph
                    points_world.append(pw[:3])

    for child in prim.GetChildren():
        points_world += _collect_mesh_points_world(child, xform_cache)

    return points_world


def get_mesh_via_prim(prim, prim_father=None):
    """Build an o3d TriangleMesh with vertices in world space.
    Uses UsdGeom.XformCache for correct transforms (fixes depth cap,
    sibling leaking, missing ancestors, and xformOpOrder issues).
    prim_father is kept for call-site compatibility but unused."""
    xform_cache = UsdGeom.XformCache()

    # Collect all mesh data (points + faces) with world-space vertices
    points_total = []
    faceVertexCounts_total = []
    faceVertexIndices_total = []

    def _collect_mesh_data(p):
        if p.IsA(UsdGeom.Mesh):
            mesh_name = str(p.GetPath()).split("/")[-1]
            if mesh_name != 'SM_Dummy':
                raw_points = p.GetAttribute("points").Get()
                faceVertexCounts = p.GetAttribute("faceVertexCounts").Get()
                faceVertexIndices = p.GetAttribute("faceVertexIndices").Get()
                if raw_points is not None and len(raw_points) > 0:
                    local_to_world = xform_cache.GetLocalToWorldTransform(p)
                    mat = np.array(local_to_world).T

                    base_num = len(points_total)
                    for pt in raw_points:
                        ph = np.array([float(pt[0]), float(pt[1]), float(pt[2]), 1.0])
                        pw = mat @ ph
                        points_total.append(pw[:3])

                    if faceVertexCounts is not None:
                        faceVertexCounts_total.extend(faceVertexCounts)
                    if faceVertexIndices is not None:
                        for idx in faceVertexIndices:
                            faceVertexIndices_total.append(base_num + int(idx))

        for child in p.GetChildren():
            _collect_mesh_data(child)

    _collect_mesh_data(prim)

    if len(points_total) == 0:
        # Return empty mesh
        return o3d.geometry.TriangleMesh()

    # Fan-triangulate quads/ngons into triangles
    triangles = []
    count = 0
    for n in faceVertexCounts_total:
        n = int(n)
        face_verts = [faceVertexIndices_total[count + i] for i in range(n)]
        for i in range(1, n - 1):
            triangles.append([face_verts[0], face_verts[i], face_verts[i + 1]])
        count += n

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(points_total)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles) if triangles else np.zeros((0, 3), dtype=np.int32))

    return o3d_mesh


def iou_2d_via_boundaries(min_points_a, max_points_a, min_points_b, max_points_b):
    a_xmin, a_xmax, a_ymin, a_ymax = min_points_a[0], max_points_a[0], min_points_a[1], max_points_a[1]
    b_xmin, b_xmax, b_ymin, b_ymax = min_points_b[0], max_points_b[0], min_points_b[1], max_points_b[1]

    box_a = [a_xmin, a_ymin, a_xmax, a_ymax]
    box_b = [b_xmin, b_ymin, b_xmax, b_ymax]
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    i_ratios = [inter_area / float(box_a_area), inter_area / float(box_b_area)]
    a_ratios = [box_a_area / box_b_area, box_b_area / box_a_area]

    return iou, i_ratios, a_ratios


def to_list(data):
    res = []
    if data is not None:
        res = [_ for _ in data]
    return res


def write_obj(path, name, points, faceuv, normals, faceVertexCounts, faceVertexIndices):
    with open(path,"w") as fp:
        for p in points:
            x,y,z = p
            fp.write(f"v {x} {y} {z}\n")
        for n in normals:
            x,y,z = n
            fp.write(f"vn {x} {y} {z}\n")
        for uv in faceuv:
            x,y = uv
            fp.write(f"vt {x} {y}\n")
        count = 0
        for n in faceVertexCounts:
            f = [_ for _ in faceVertexIndices[count:count+n]]
            fp.write("f")
            for idx in f:
                idx += 1
                fp.write(f" {idx}/{idx}/{idx}")
            fp.write("\n")
            count += n


def recursive_parse_point_cloud(prim):
    """Collect mesh vertices under prim, transformed to world space via XformCache."""
    xform_cache = UsdGeom.XformCache()
    return _collect_mesh_points_world(prim, xform_cache)


def extract_obj_mesh(stage, scene_name, black_list=[]):
    bbox_list = []
    mesh_paths = []

    meshes = stage.GetPrimAtPath("/Root/Meshes")
    scopes = meshes.GetChildren()
    for scope in tqdm.tqdm(scopes):
        scope_name = scope.GetName()
        if scope_name in black_list:
            continue
        categories = scope.GetChildren()
        for category in categories:
            category_name = category.GetName()
            instances = category.GetChildren()
            for instance in instances:
                mesh = get_mesh_via_prim(instance, category)
                point_cloud = mesh.sample_points_uniformly(number_of_points=7000)
                point_cloud = np.asarray(point_cloud.points)
                max_point = point_cloud.max(0)
                min_point = point_cloud.min(0)
                mesh_path = str(instance.GetPath())
                bbox_list.append([min_point, max_point])
                mesh_paths.append(mesh_path)
    return bbox_list, mesh_paths
