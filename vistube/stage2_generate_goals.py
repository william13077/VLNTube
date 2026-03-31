# -*- coding: utf-8 -*-
"""
Discretized version of path planning.
Uses discrete action space during path planning: forward 0.25m, turn 15 degrees.
"""

import numpy as np
import cv2
import os
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import datetime
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from PIL import Image
import natsort
import sys
import math
from vistube.discrete_path_planner import DiscretePathPlanner, get_discrete_path, ACTION_STOP, ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT, remove_initial_turns
from vistube.goal_gen.gen_goal_inst import generate_instruction_smart, generate_instruction_v8, correct_description_v2
from vistube.tube_utils import extract_object_type_outer
from splits.split_utils import is_trainval
import random

# --- Parse CLI arguments before Isaac Sim init (SimulationApp may consume sys.argv) ---
parser = argparse.ArgumentParser(description='Stage 2: Generate goal navigation paths with discrete action space.')
parser.add_argument('scene_dir', nargs='?', default=None,
                    help='Path to a specific scene directory to process')
parser.add_argument('--dataroot', type=str, default='/mnt/6t/dataset/vlnverse/',
                    help='Root directory containing scene folders')
parser.add_argument('--metaroot', type=str, default='/data/lsh/scene_summary/metadata/',
                    help='Root directory containing scene metadata (freemap, room_region, etc.)')
parser.add_argument('--usd-root', type=str, default='/mnt/6t/dataset/vlnverse/',
                    help='Root directory containing USD scene files')
parser.add_argument('--scene-graph', type=str, default='/data/lsh/scene_summary/scene_summary/',
                    help='Root directory containing scene graph data (object_dict.json)')
parser.add_argument('--task-dir', type=str, default='goalnav_discrete',
                    help='Subdirectory name for saving task outputs')
parser.add_argument('--sample-dir', type=str, default='sampled_points',
                    help='Subdirectory name for sampled points (output of stage1, under each scene folder)')
parser.add_argument('--splits-file', type=str, default='splits/scene_splits.json',
                    help='Path to the scene splits JSON file')
args, unknown_args = parser.parse_known_args()
# Restore sys.argv with only unknown args so SimulationApp doesn't choke on ours
sys.argv = [sys.argv[0]] + unknown_args

# Both random and np.random must be seeded for reproducibility.
# np.random is used in find_endpoint_in_arc to sample viewpoints; if unseeded,
# different viewpoints cause path planning to succeed/fail differently (especially
# in narrow areas), which shifts the random module's state via control flow
# divergence, ultimately changing which goals get selected downstream.
random.seed(1025)
np.random.seed(1025)
# ==========================================================
#              Isaac Sim initialization and related libraries
# ==========================================================
from omni.isaac.kit import SimulationApp
CONFIG = {"headless": True}
print("Initializing Isaac Sim, this may take some time...")
simulation_app = SimulationApp(CONFIG)
import carb

import omni.usd
import omni.kit.commands
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
from omni.isaac.core.prims import XFormPrim
from pyquaternion import Quaternion
from vistube.find_unique_objects import find_bidirectionally_unique_objects_debug, find_bidirectionally_unique_objects_exact

# ==========================================================
#                   Path planning libraries
# ==========================================================
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.grid import Grid
from pathfinding.core.diagonal_movement import DiagonalMovement
from vistube.path_finder import simplify_path_with_collision_check
from vistube.path_utils import sample_walkable_point_in_polygon, get_path, get_front_face_info, find_endpoint_in_arc, douglas_peucker, densify_path_float, capture_final_scene_photo, draw_semitransparent_fan, world_to_pixel, segment_path_by_all_intersections, proc_path_zerui, get_opposing_faces_info, calculate_proximity_risk_score, find_containing_room, capture_final_scene_photo_twostep


# ==========================================================
#              Helper functions
# ==========================================================

def get_object_data(instance_id, data):
    return data.get(instance_id)

def describe_actions(actions, forward_distance=0.25, turn_angle=15):
    """Convert action sequence to description."""
    if not actions:
        return "No actions"

    forward_count = actions.count(ACTION_FORWARD)
    left_count = actions.count(ACTION_TURN_LEFT)
    right_count = actions.count(ACTION_TURN_RIGHT)

    total_distance = forward_count * forward_distance
    total_left_angle = left_count * turn_angle
    total_right_angle = right_count * turn_angle

    return f"Forward: {forward_count}x ({total_distance:.2f}m), Left: {left_count}x ({total_left_angle}°), Right: {right_count}x ({total_right_angle}°)"

def save_discrete_path_visualization(occupancy_map, actions, path_pixels, start_point, end_point, save_path):
    """Save discrete path visualization."""
    rgb_display = np.stack([occupancy_map] * 3, axis=-1) * 255

    # Draw path
    if len(path_pixels) > 0:
        for i in range(len(path_pixels) - 1):
            cv2.line(rgb_display,
                    tuple(path_pixels[i].astype(int)),
                    tuple(path_pixels[i+1].astype(int)),
                    (0, 255, 128), 2)

    # Mark start and end points
    cv2.circle(rgb_display, tuple(start_point.astype(int)), 5, (0, 255, 0), -1)  # Green start point
    cv2.circle(rgb_display, tuple(end_point.astype(int)), 5, (255, 0, 0), -1)   # Red end point

    # Add action statistics
    text = describe_actions(actions)
    cv2.putText(rgb_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(save_path, rgb_display)

# ==========================================================
#                         Main program
# ==========================================================

# --- Configuration (from CLI args) ---

dataroot = args.dataroot
usd_root = args.usd_root
metaroot = args.metaroot
scene_graph = args.scene_graph

task_dir = args.task_dir
PRIM_PATH_IN_STAGE = "/World/MyLoadedAsset"
CAMERA_PRIM_PATH = "/World/MyCamera"

SCALE_FACTOR = 1
AGENT_HEIGHT = 1.2
IMAGE_WIDTH, IMAGE_HEIGHT = 500, 500

LENGTH_THRE = 80 #100  # Minimum path length
PATH_MAX = 5       # Maximum number of paths per goal
ENDPOINT_RADIUS_MAX = 1.5
ENDPOINT_RADIUS_MIN = 1.5
ENDPOINT_ARC_DEGREES = 60

# Discretization parameters (these values override defaults in discrete_path_planner)
FORWARD_DISTANCE = 0.25  # Forward distance (meters) - exact distance per forward step
TURN_ANGLE = 15         # Turn angle (degrees) - exact angle per turn
GOAL_REACH_THRESHOLD = 0.125 #0.375  # Goal reach threshold (meters)
PARTIAL_PATH_THRESHOLD = 3.0   # Maximum distance for accepting partial paths (meters) - tightened from 8.0 to 3.0
MAX_ENDPOINT_DEVIATION = 1.5  # Maximum allowed deviation between path endpoint and target endpoint (meters)
MAX_GOAL_DEVIATION = 1.5 # 2.5

# Update discrete_path_planner global parameters
from vistube import discrete_path_planner
discrete_path_planner.FORWARD_DISTANCE = FORWARD_DISTANCE
discrete_path_planner.TURN_ANGLE = TURN_ANGLE
discrete_path_planner.TURN_ANGLE_RAD = math.radians(TURN_ANGLE)
discrete_path_planner.GOAL_REACH_THRESHOLD = GOAL_REACH_THRESHOLD
discrete_path_planner.PARTIAL_PATH_THRESHOLD = PARTIAL_PATH_THRESHOLD

if __name__ == '__main__':
    temp_dir = os.listdir(dataroot)
    dir = natsort.natsorted([i for i in temp_dir if os.path.isdir(os.path.join(dataroot, i))])

    tmp_id = None
    if args.scene_dir is not None:
        dir_path_from_shell = args.scene_dir
        tmp_id = dir_path_from_shell.rstrip('/').split('/')[-1]

    if tmp_id is not None:
        if not is_trainval(args.splits_file, tmp_id):
            sys.exit()
        print(f'==> Processing scene: {tmp_id}')

    # Test scene
    for scene_id in [tmp_id]:

        try:
            # --- 1. Configure paths ---
            ROOT_DIR = os.path.join(dataroot, scene_id)
            SCENE_USD_PATH = os.path.join(usd_root,scene_id,f'start_result_navigation.usd')
            OBJECTS_INFO_PATH = os.path.join(scene_graph, scene_id, 'object_dict.json')
            OCCUPANCY_PATH = os.path.join(metaroot, scene_id, 'freemap.npy')
            RGB_INIT_PATH = os.path.join(dataroot, scene_id, 'occupancy.png')
            ANNOTATION_PATH = os.path.join(metaroot, scene_id, 'room_region.json')
            annotation_path = os.path.join(dataroot, scene_id, args.sample_dir, 'sampled_points.json')

            # Create save directories (with _discrete suffix to distinguish)
            save_dir = os.path.join(ROOT_DIR, task_dir)
            npy_dir = os.path.join(save_dir, 'npy')
            vis_dir = os.path.join(save_dir, 'vis')
            ref_dir = os.path.join(save_dir, 'ref')
            action_dir = os.path.join(save_dir, 'actions')  # Save action sequences

            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(npy_dir, exist_ok=True)
            os.makedirs(vis_dir, exist_ok=True)
            os.makedirs(ref_dir, exist_ok=True)
            os.makedirs(action_dir, exist_ok=True)

            # --- 2. Load Isaac Sim scene and data ---
            print("--- Stage 1: Loading scene and data ---")
            world = World(stage_units_in_meters=1.0)
            add_reference_to_stage(usd_path=SCENE_USD_PATH, prim_path="/World/scene")
            asset_xform = XFormPrim(prim_path="/World/scene")
            asset_xform.set_local_scale(np.array([SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR]))

            camera = Camera(prim_path="/World/CombinedCamera", resolution=(IMAGE_WIDTH, IMAGE_HEIGHT))
            camera.initialize()

            ## zerui's setting
            camera.add_distance_to_image_plane_to_frame()
            camera.set_clipping_range(near_distance=0.01, far_distance=10000.0)
            focal_length = 10.0
            aperture = 2 * focal_length
            camera.set_focal_length(focal_length)
            camera.set_horizontal_aperture(aperture)
            camera.set_vertical_aperture(aperture)

            world.reset()
            with open(ANNOTATION_PATH, 'r') as f:
                scene_anno = json.load(f)
            room_polygon = [[(p[1], p[0]) for p in scene_anno[r]] for r in scene_anno]

            with open(annotation_path, 'r') as f:
                gaus_anno = json.load(f)
            possible_start = []
            for i in [v['sampled_points'] for v in gaus_anno]:
                possible_start += i

            occ_map = np.load(OCCUPANCY_PATH)
            rgb_map_image = cv2.imread(RGB_INIT_PATH)

            # --- 3. Prepare occupancy map ---
            print("\n--- Stage 2: Preparing discretized path planning ---")

            matrix_fat = copy.copy(occ_map[1:, 1:])
            matrix_fat[matrix_fat == 2] = 0
            matrix_fat = 1 - cv2.dilate((1 - matrix_fat).astype(np.uint8), np.ones([4, 4], dtype=np.uint8), iterations=2)
            fat_map_uint8 = (1 - cv2.dilate((1 - matrix_fat).astype(np.uint8), np.ones([4, 4], dtype=np.uint8), iterations=2)).astype(np.uint8)
            matrix_thin = copy.copy(occ_map[1:, 1:])
            matrix_thin[matrix_thin == 2] = 0
            matrix_thin = 1 - cv2.dilate((1 - matrix_thin).astype(np.uint8), np.ones([3, 3], dtype=np.uint8), iterations=1)

            # Create discrete path planner
            discrete_planner = DiscretePathPlanner(matrix_fat, occ_map)

            with open(OBJECTS_INFO_PATH, 'r', encoding='utf-8') as f:
                objects_data = json.load(f)

            # Process door coordinates
            obj_k = [p for p in objects_data.keys() if 'doorsill_' in p]
            door_coor = [[objects_data[k]['min_points'], objects_data[k]['max_points']] for k in obj_k]
            door_coor = [[[np.argmin(np.abs(occ_map[0] - i[0][0])),
                          np.argmin(np.abs(occ_map[:, 0] - i[0][1]))],
                         [np.argmin(np.abs(occ_map[0] - i[1][0])),
                          np.argmin(np.abs(occ_map[:, 0] - i[1][1]))]]
                        for i in door_coor]
            occ_map[0,1:]
            door_coor = [d for d in door_coor if np.linalg.norm(np.array(d[0]) - np.array(d[1])) < 50]

            source_blacklist=['ceiling','doorsil','ornament','daily_equipment','tooling','unknown','menorah','celling','chandelier']
            obj_description,obj_dict = find_bidirectionally_unique_objects_debug(
                objects_data, n=100000000, debug = False,
                source_blacklist=source_blacklist, target_blacklist=source_blacklist,
            )
            Template = "Go to the {location} and find the {goal} that is {rel} the {ref}"

            goal_count = 0
            instruction_dict = {}
            action_sequences = {}  # Store action sequences

            for _, goal in enumerate(obj_dict['rel'] + obj_dict['sem']):
                print('\n', '+' * 60)
                print(f'==> Processing {goal_count} goal with DISCRETE path planning')


                # Parse goal
                TARGET_OBJECT_INSTANCE_ID = goal['object_1_id']
                SOURCE_OBJECT_INSTANCE_ID = goal['object_2_id']
                target_name = TARGET_OBJECT_INSTANCE_ID.split('/')[0]
                target_name = extract_object_type_outer(target_name)
                source_name = SOURCE_OBJECT_INSTANCE_ID.split('/')[0]
                source_name = extract_object_type_outer(source_name)

                if target_name == source_name:
                    continue

                rel_1_to_2 = goal['object_1_relation_to_2']
                if rel_1_to_2 == 'out of':
                    rel_1_to_2 = 'near'

                target_object = get_object_data(TARGET_OBJECT_INSTANCE_ID, objects_data)
                if not target_object:
                    raise ValueError(f"Error: Cannot find ID '{TARGET_OBJECT_INSTANCE_ID}'")
                ref_object = get_object_data(SOURCE_OBJECT_INSTANCE_ID, objects_data)

                cur_room = target_object.get('scope').split('_')[0]
                if cur_room.lower() in ['unknown', 'wall', 'other']:
                    print("===> Skip unknown room")
                    continue

                goal_description = Template.format(
                    location=correct_description_v2(cur_room),
                    goal=correct_description_v2(target_name),
                    ref=correct_description_v2(source_name),
                    rel=rel_1_to_2 if '/' not in rel_1_to_2 else rel_1_to_2.split('/')[0]
                )

                smart_description = generate_instruction_v8(
                    target=correct_description_v2(target_name),
                    reference=correct_description_v2(source_name),
                    relation=rel_1_to_2 if '/' not in rel_1_to_2 else rel_1_to_2.split('/')[0],
                    location=correct_description_v2(cur_room)
                )

                print(goal_description)
                print('==> Instruction:', smart_description[0])

                # Compute goal position

                distance_transform_map = cv2.distanceTransform(fat_map_uint8, cv2.DIST_L2, 5)
                face_A, face_B = get_opposing_faces_info(target_object.get("min_points"), target_object.get("max_points"))
                risk_A = calculate_proximity_risk_score(face_A, ENDPOINT_RADIUS_MAX, ENDPOINT_ARC_DEGREES, occ_map, distance_transform_map)
                risk_B = calculate_proximity_risk_score(face_B, ENDPOINT_RADIUS_MAX, ENDPOINT_ARC_DEGREES, occ_map, distance_transform_map)

                try:
                    A_point, _ = find_endpoint_in_arc(face_center=face_A['center'], normal_vector=face_A['normal'], min_radius=ENDPOINT_RADIUS_MIN, max_radius=ENDPOINT_RADIUS_MAX, arc_degrees=ENDPOINT_ARC_DEGREES, max_tries=1000, occ_map=occ_map, fat_map=matrix_fat)
                except:
                    A_point = [-100, -100]
                try:
                    B_point, _ = find_endpoint_in_arc(face_center=face_B['center'], normal_vector=face_B['normal'], min_radius=ENDPOINT_RADIUS_MIN, max_radius=ENDPOINT_RADIUS_MAX, arc_degrees=ENDPOINT_ARC_DEGREES, max_tries=1000, occ_map=occ_map, fat_map=matrix_fat)
                except:
                    B_point = [-100, -100]

                if B_point == [-100, -100] and A_point == [-100, -100]: continue

                T_x = np.argmin(np.abs(occ_map[0] - target_object['position'][0]))
                T_y = np.argmin(np.abs(occ_map[:, 0] - target_object['position'][1]))

                target_room = find_containing_room([T_x, T_y], room_polygon)
                a_room = find_containing_room(A_point, room_polygon)
                b_room = find_containing_room(B_point, room_polygon)

                if a_room != target_room and b_room != target_room: continue

                if a_room != target_room:
                    final_face, end_point = face_B, B_point
                elif b_room != target_room:
                    final_face, end_point = face_A, A_point
                elif risk_A <= risk_B:
                    final_face, end_point = face_A, A_point
                else:
                    final_face, end_point = face_B, B_point

                if end_point == [-100,-100]: continue

                camera_target_position = final_face["center"]
                front_normal = final_face["normal"]

                # Try multiple start points for each goal
                count = 0
                attempt = 0
                is_goal_successful = False
                random.shuffle(possible_start)

                for start_point in possible_start:
                    if count >= PATH_MAX:
                        break
                    if attempt >= 51:
                        break

                    attempt += 1
                    start_point = np.array(start_point, dtype=np.int32)

                    print(f"\n--- Planning path with discrete A* algorithm (attempt {attempt}) ---")

                    # Use discretized path planning
                    actions, path_world, _ = get_discrete_path(
                        matrix_fat, occ_map,
                        start_point, end_point,
                        initial_angle=None,  # Auto-orient towards target
                        visualize=False
                    )

                    if actions is None or path_world is None:
                        print(f"Discrete path planning failed")
                        continue

                    # Post-processing: remove initial turns and get correct initial orientation
                    actions, path_world, agent_initial_angle = remove_initial_turns(actions, path_world)

                    # --- Endpoint quality check ---
                    ## detection collision
                    collision_detected = False
                    offsets = [
                        (0, 0), (2, 0), (0, 1), (-2, 0), (0, -2),
                        (1, 1), (-1, 1), (-1, -1), (1, -1)
                    ]
                    test_occ = copy.copy(occ_map[1:,1:])
                    test_occ[test_occ==2]=0
                    map_h, map_w = test_occ.shape
                    # breakpoint()
                    path_px = np.array([np.argmin(np.abs(occ_map[0] - p)) for p in path_world[:,0]])
                    path_py = np.array([np.argmin(np.abs(occ_map[:, 0] - p)) for p in path_world[:,1]])
                    for dy, dx in offsets:
                        check_y = path_py + dy
                        check_x = path_px + dx

                        # Check if all indices are within bounds
                        if np.any(check_y < 0) or np.any(check_y >= map_h) or \
                            np.any(check_x < 0) or np.any(check_x >= map_w):
                            collision_detected = True
                            break

                        # Check collision
                        if not np.all(test_occ[check_y, check_x]):
                            collision_detected = True
                            break

                    if collision_detected:
                        print('collision')
                        continue

                    # Check 1: Distance between actual endpoint and target endpoint
                    if len(path_world) > 0:
                        final_world_pos = path_world[-1]
                        # Convert end_point (pixel coordinates) to world coordinates for comparison
                        end_world_x = occ_map[0, end_point[0]]
                        end_world_y = occ_map[end_point[1], 0]

                        actual_distance = np.linalg.norm(
                            np.array([final_world_pos[0], final_world_pos[1]]) -
                            np.array([end_world_x, end_world_y])
                        )
                        # breakpoint()
                        goal_distance = np.linalg.norm(np.array([final_world_pos[0], final_world_pos[1]]) -np.array(target_object['position'][:2]))
                        # Use globally defined endpoint deviation threshold
                        if actual_distance > MAX_ENDPOINT_DEVIATION and goal_distance > MAX_GOAL_DEVIATION:
                            print(f'Path endpoint deviation too large: {actual_distance:.2f}m > {MAX_ENDPOINT_DEVIATION}m')
                            continue

                        # Check 2: Whether endpoint is in the correct room
                        final_px = np.argmin(np.abs(occ_map[0] - final_world_pos[0]))
                        final_py = np.argmin(np.abs(occ_map[:, 0] - final_world_pos[1]))

                        # Room containing the endpoint
                        final_room = find_containing_room([final_px, final_py], room_polygon)

                        # Room containing the target object (already computed above)
                        if final_room != target_room:
                            print(f'Path endpoint not in target room (endpoint room: {final_room}, target room: {target_room})')
                            continue

                        print(f'Endpoint quality check passed - deviation: {actual_distance:.2f}m, room match: OK')

                    # Check path length
                    path_pixels = []
                    for world_pos in path_world:
                        px = np.argmin(np.abs(occ_map[0] - world_pos[0]))
                        py = np.argmin(np.abs(occ_map[:, 0] - world_pos[1]))
                        path_pixels.append([px, py])
                    path_pixels = np.array(path_pixels)

                    if len(path_pixels) > 1:
                        total_length = np.sum(np.linalg.norm(
                            path_pixels[1:] - path_pixels[:-1], axis=1
                        ))

                        if total_length < LENGTH_THRE:
                            print(f'Path length {total_length} below threshold {LENGTH_THRE}')
                            continue

                        print(f'Discrete path length: {total_length:.2f}')
                        print(f'Action sequence: {describe_actions(actions)}')

                        # Visualization
                        rgb_display = copy.deepcopy(rgb_map_image) / 255.0

                        # Draw fan-shaped region
                        pixel_size = np.abs(occ_map[0, 2] - occ_map[0, 1])
                        radius_px = ENDPOINT_RADIUS_MAX / pixel_size

                        # Draw two candidate faces
                        for face, color, alpha in [(face_A, (0.8, 0.2, 0.8), 0.3),
                                                  (face_B, (0.8, 0.2, 0.8), 0.3),
                                                  (final_face, (1.0, 1.0, 0.0), 0.4)]:
                            center_px = world_to_pixel(face["center"][:2], occ_map)
                            rgb_display = draw_semitransparent_fan(
                                rgb_display, center_px, radius_px,
                                face["normal"][:2], ENDPOINT_ARC_DEGREES,
                                color, alpha
                            )


                        rgb_display[path_pixels[:, 1], path_pixels[:, 0]] = [0, 1, 0.5]

                        # Mark start and end points
                        cv2.circle(rgb_display, tuple(start_point.astype(int)),
                                 3, (0.0, 1.0, 0.0), -1)
                        cv2.circle(rgb_display, tuple(np.array(end_point).astype(int)),
                                 3, (1.0, 0.0, 0.0), -1)

                        # Save visualization
                        vis_path = os.path.join(vis_dir, f'vis_{goal_count}_{count}.png')
                        plt.imsave(vis_path, np.clip(rgb_display, 0.0, 1.0))
                        print(f"==> Saved visualization to {vis_path}")

                        # Save path data
                        npy_path = os.path.join(npy_dir, f'path_{goal_count}_{count}.npy')
                        np.save(npy_path, path_world)
                        print(f'==> Saved path to {npy_path}')

                        # Save action sequence (convert to native Python types to avoid JSON serialization errors)
                        action_path = os.path.join(action_dir, f'actions_{goal_count}_{count}.json')
                        action_data = {
                            'actions': [int(a) for a in actions],  # Convert to native int type
                            'forward_distance': FORWARD_DISTANCE,
                            'turn_angle': TURN_ANGLE,
                            'description': describe_actions(actions),
                            'start_pos': start_point.tolist(),
                            'end_pos': [int(end_point[0]), int(end_point[1])] if isinstance(end_point, (list, np.ndarray)) else end_point,
                            'path_length': float(total_length),  # Convert to native float type
                            'initial_angle_rad': float(agent_initial_angle),  # Initial orientation (radians)
                            'initial_angle_deg': float(math.degrees(agent_initial_angle))  # Initial orientation (degrees)
                        }
                        with open(action_path, 'w') as f:
                            json.dump(action_data, f, indent=2)
                        print(f'==> Saved action sequence to {action_path}')

                        # Capture scene photo
                        if len(path_world) > 0:
                            final_world_pos = path_world[-1]
                            final_camera_position = np.array([
                                final_world_pos[0],
                                final_world_pos[1],
                                AGENT_HEIGHT
                            ])

                            capture_final_scene_photo_twostep(
                                camera=camera, world=world,
                                viewpoint_pos=final_camera_position,
                                target_pos=camera_target_position,
                                world_up_vec=(0, 0, 1),
                                save_path=os.path.join(ref_dir, f'goal_{goal_count}.png')
                            )

                        count += 1
                        is_goal_successful = True

                        if goal_count not in instruction_dict:
                            instruction_dict[goal_count] = {}
                            instruction_dict[goal_count]['instruction'] = smart_description
                            instruction_dict[goal_count]['goal'] = goal
                            instruction_dict[goal_count]['room'] = target_object.get('scope')
                            action_sequences[goal_count] = []

                        action_sequences[goal_count].append({
                            'path_id': count - 1,
                            'actions': [int(a) for a in actions],  # Convert to native int type
                            'description': describe_actions(actions),
                            'initial_angle_rad': float(agent_initial_angle),
                            'initial_angle_deg': float(math.degrees(agent_initial_angle))
                        })

                if is_goal_successful:
                    goal_count += 1

                # Save instructions and action sequences
                with open(os.path.join(save_dir, 'goal_inst.json'), 'w', encoding='utf-8') as f:
                    json.dump(instruction_dict, f, ensure_ascii=False, indent=2)

                with open(os.path.join(save_dir, 'all_action_sequences.json'), 'w', encoding='utf-8') as f:
                    json.dump(action_sequences, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"\nError occurred: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print("\nScript execution complete, closing Isaac Sim application.")
            simulation_app.close()
