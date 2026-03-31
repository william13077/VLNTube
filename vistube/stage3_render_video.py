# --- Parse CLI arguments before Isaac Sim init (SimulationApp may consume sys.argv) ---
import argparse
import sys

parser = argparse.ArgumentParser(description='Stage 3: Render navigation videos from discrete paths.')
parser.add_argument('scene_dir', nargs='?', default=None,
                    help='Path to a specific scene directory to process')
parser.add_argument('--dataroot', type=str, default='/mnt/6t/dataset/vlnverse',
                    help='Root directory containing scene folders')
parser.add_argument('--metaroot', type=str, default='/data/lsh/scene_summary/metadata/',
                    help='Root directory containing scene metadata')
parser.add_argument('--usd-root', type=str, default='/mnt/6t/dataset/vlnverse',
                    help='Root directory containing USD scene files')
parser.add_argument('--task-dir', type=str, default='goalnav_discrete',
                    help='Subdirectory name for task outputs (input paths)')
parser.add_argument('--seq-dir', type=str, default='sequence_discrete',
                    help='Subdirectory name for rendered video sequences')
parser.add_argument('--splits-file', type=str, default='splits/scene_splits.json',
                    help='Path to the scene splits JSON file')
args, unknown_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown_args

# import omni.usd
from isaacsim import SimulationApp
CONFIG = {"headless": True}
simulation_app = SimulationApp(CONFIG)
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera # Import Camera class
import omni.kit.commands # <--- need to import commands
from omni.isaac.core.prims import XFormPrim
import numpy as np # Import numpy for image data processing
from PIL import Image # Import PIL (Pillow) for image file saving
import os
import carb
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import glob, natsort, random
from tqdm import tqdm
import signal, cv2, json, math
from vistube.tube_utils import remove_initial_turns
from vistube.tube_utils import rot3_from_O_to_AB, DEFAULT_CAMERA_FORWARD
from splits.split_utils import is_trainval

# Global random seed - for reproducibility
SAMPLE_SEED = 1024  # Can be changed as needed
random.seed(SAMPLE_SEED)
np.random.seed(SAMPLE_SEED)


# --- Timeout handling mechanism ---
def normalize_depth_to_png(depth_array: np.ndarray, max_depth: float = 10.0):
    depth_clipped = np.clip(depth_array, 0, max_depth)
    normalized_depth = depth_clipped / max_depth
    depth_8bit = (normalized_depth * 255)
    return depth_8bit

class TimeoutException(Exception):
    """Custom timeout exception"""
    pass

def timeout_handler(signum, frame):
    """Signal handler function, called when a signal is received"""
    print("!!! Path processing timed out !!!")
    raise TimeoutException("Path processing timed out after 2 minutes")

# Register signal handler, associate SIGALRM signal with our handler function
signal.signal(signal.SIGALRM, timeout_handler)

def quat2euler(q):
    '''quaternion'''
    return R.from_quat(q).as_euler('XYZ')

def euler2quat(euler):
    return R.from_euler('XYZ', euler).as_quat()

def euler2mat(euler):
    return R.from_euler('XYZ', euler).as_matrix()

def RT2HT(R, T):
    HT = np.eye(4)
    HT[:3, :3] = R
    HT[:3, 3] = T
    return HT

def quat2mat(q):
    return R.from_quat(q).as_matrix()

def eePose2HT(eePose):
    HT = RT2HT(quat2mat(eePose[3:]), eePose[:3])
    return HT

def mat2quat(mat):
    return R.from_matrix(mat).as_quat()

def HT2eePose(T):
    out = np.hstack((T[:3, 3], mat2quat(T[:3, :3])))
    return out

def rotate_x(cameraPose, angle, degrees=True):
    pos = cameraPose[:3]
    quat = cameraPose[3:]
    euler = R.from_quat(quat).as_euler('xyz', degrees=degrees) - np.array([angle, 0, 0])
    new_quat = R.from_euler('xyz', euler, degrees=degrees).as_quat()
    return np.concatenate((pos, new_quat))


def rotate_y(cameraPose, angle, degrees=True):
    pos = cameraPose[:3]
    quat = cameraPose[3:]
    euler = R.from_quat(quat).as_euler('yxz', degrees=degrees) - np.array([angle, 0, 0])
    new_quat = R.from_euler('yxz', euler, degrees=degrees).as_quat()
    return np.concatenate((pos, new_quat))


def rotate_z(cameraPose, angle, degrees=True):
    pos = cameraPose[:3]
    quat = cameraPose[3:]
    euler = R.from_quat(quat).as_euler('zxy', degrees=degrees) - np.array([angle, 0, 0])
    new_quat = R.from_euler('zxy', euler, degrees=degrees).as_quat()
    return np.concatenate((pos, new_quat))

def rotate_camera_world_frame(cameraPose, Euler_rot_change):
    """
    Rotate the camera in the world frame.
    :param cameraPose: [x, y, z, qx, qy, qz, qw]
    :param Euler_rot_change: [roll, pitch, yaw]
    :return: new camera pose
    """
    # Convert quaternion to rotation matrix
    R_camera_old = quat2mat(cameraPose[3:])

    # Convert Euler angles to rotation matrix
    Matrix_rot_change = euler2mat(np.radians(Euler_rot_change))

    # Calculate new rotation matrix
    R_camera_new = Matrix_rot_change @ R_camera_old

    # Convert rotation matrix back to quaternion
    new_quat = mat2quat(R_camera_new)

    # Return new camera pose
    return np.concatenate((cameraPose[:3], new_quat))

def rotate_camera_body_frame(cameraPose, Euler_rot_change):
    """
    Rotate the camera in the body frame.
    :param cameraPose: [x, y, z, qx, qy, qz, qw]
    :param Euler_rot_change: [roll, pitch, yaw]
    :return: new camera pose
    """
    # Convert quaternion to rotation matrix
    R_camera_old = quat2mat(cameraPose[3:])

    # Convert Euler angles to rotation matrix
    Matrix_rot_change = euler2mat(np.radians(Euler_rot_change))

    # Calculate new rotation matrix
    R_camera_new = R_camera_old @ Matrix_rot_change

    # Convert rotation matrix back to quaternion
    new_quat = mat2quat(R_camera_new)

    # Return new camera pose
    return np.concatenate((cameraPose[:3], new_quat))


def cameraPose2Heading(cameraPose: np.ndarray) -> np.ndarray:
    """
    Args:
        cameraPose (np.ndarray): Array of shape (7,), representing [x, y, z, qx, qy, qz, qw].

    Returns:
        float: The heading angle in radians.
    """

    return R.from_quat(cameraPose[3:7]).as_euler('ZXY', degrees=False)

## get RT matrix ##
# When pose is all zeros, the camera forward direction is vector O
O = DEFAULT_CAMERA_FORWARD  # Set it to point along the x-axis (backward compat)

def intelligent_sample_paths(npy_files, max_samples=50):
    """
    Intelligently sample npy files, trying to cover all different goals.

    Args:
        npy_files: list of npy file paths
        max_samples: maximum number of samples

    Returns:
        list of sampled file paths
    """
    import re

    # Uses the global random seed for reproducibility
    # Seed has been set at the beginning of the program

    # If total files <= max_samples, return all files directly
    if len(npy_files) <= max_samples:
        print(f"Total files ({len(npy_files)}) <= max_samples ({max_samples}), using all files")
        return npy_files

    # Parse filenames, extract goal labels and path indices
    goal_to_paths = {}
    for filepath in npy_files:
        filename = os.path.basename(filepath).replace('.npy', '')
        # Assume filename format is path_X_Y or similar
        match = re.match(r'.*?_(\d+)_(\d+)', filename)
        if match:
            goal_id = int(match.group(1))
            # path_id = int(match.group(2))  # No need to use path_id, only need goal_id for grouping
            if goal_id not in goal_to_paths:
                goal_to_paths[goal_id] = []
            goal_to_paths[goal_id].append(filepath)
        else:
            # If it doesn't match the expected format, try other possible formats
            # Or handle it as a special case
            print(f"Warning: Could not parse filename pattern from {filename}")
            # Put unparseable files into a special goal group
            if -1 not in goal_to_paths:
                goal_to_paths[-1] = []
            goal_to_paths[-1].append(filepath)

    num_goals = len(goal_to_paths)
    print(f"Found {num_goals} different goals in {len(npy_files)} files")

    sampled_paths = []

    # Case 1: Number of goals >= max_samples
    if num_goals >= max_samples:
        # Select 1 per goal (if goals exceed max_samples, randomly choose some goals)
        selected_goals = list(goal_to_paths.keys())
        if num_goals > max_samples:
            # Randomly select max_samples goals
            selected_goals = random.sample(selected_goals, max_samples)
            print(f"More goals ({num_goals}) than max_samples ({max_samples}), randomly selected {max_samples} goals")

        for goal_id in selected_goals:
            # Randomly select one path from each goal
            sampled_paths.append(random.choice(goal_to_paths[goal_id]))

    # Case 2: Number of goals < max_samples
    else:
        # First ensure each goal has at least one path
        for goal_id, paths in goal_to_paths.items():
            sampled_paths.append(random.choice(paths))

        # Calculate remaining quota
        remaining_quota = max_samples - len(sampled_paths)

        if remaining_quota > 0:
            # Collect all unselected paths
            all_remaining_paths = []
            for goal_id, paths in goal_to_paths.items():
                for path in paths:
                    if path not in sampled_paths:
                        all_remaining_paths.append(path)

            # If there are remaining paths, randomly select to fill the quota
            if all_remaining_paths:
                additional_samples = min(remaining_quota, len(all_remaining_paths))
                additional_paths = random.sample(all_remaining_paths, additional_samples)
                sampled_paths.extend(additional_paths)
                print(f"Added {additional_samples} additional paths to reach total of {len(sampled_paths)} samples")

    # Shuffle order (maintain randomness)
    random.shuffle(sampled_paths)

    # Output sampling statistics
    sampled_goals = set()
    for filepath in sampled_paths:
        filename = os.path.basename(filepath).replace('.npy', '')
        match = re.match(r'.*?_(\d+)_(\d+)', filename)
        if match:
            sampled_goals.add(int(match.group(1)))
        else:
            sampled_goals.add(-1)  # Special marker

    print(f"Final sampling: {len(sampled_paths)} paths covering {len(sampled_goals)} goals")

    return sampled_paths

# --- Configuration ---
MAX_ATTEMPTS = 5
BATCH_SIZE = 25 # some scene memory leak, it will restart each batch
EXIT_CODE_ALL_DONE = 10
EXIT_CODE_SKIP_DONE = 11
HEIGHT = 1.2
MAX_EPISODE = 50 # 200 Maximum number of samples, up to you
# --- Configuration (from CLI args) ---
dataroot = args.dataroot
usd_root = args.usd_root
metaroot = args.metaroot

task_dir = args.task_dir

seq_dir = args.seq_dir


PRIM_PATH_IN_STAGE = "/World/MyLoadedAsset"


# --- Camera configuration ---
# 3. Camera prim path in the scene
CAMERA_PRIM_PATH = "/World/MyCamera"


# 6. Camera resolution (width, height)
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224 # depends on your camera settings


# --- Main program ---
if __name__ == "__main__":

    temp_dir = os.listdir(dataroot)
    dir = natsort.natsorted([i for i in temp_dir if os.path.isdir(os.path.join(dataroot,i))])
    tmp_id = None
    if args.scene_dir is not None:
        dir_path_from_shell = args.scene_dir
        tmp_id = dir_path_from_shell.rstrip('/').split('/')[-1]
    if tmp_id is not None:
        if not is_trainval(args.splits_file, tmp_id):
            sys.exit(EXIT_CODE_SKIP_DONE)
        print(f'==> Processing scene: {tmp_id}')
    for scene_id in [tmp_id]:
        SAVE_DIR = os.path.join(dataroot,scene_id,task_dir,seq_dir)
        os.makedirs(SAVE_DIR,exist_ok=True)

        # Get all npy files
        all_npy_files = natsort.natsorted(glob.glob(os.path.join(dataroot,scene_id,task_dir,'npy/*.npy')))

        # Check if inst file exists (if so, use paths from inst file)
        inst_path = os.path.join(dataroot,scene_id,task_dir,'inst/inst_img_sequence.json')
        if os.path.exists(inst_path):
            with open(inst_path,'r') as f:
                inst_dict = json.load(f)
            all_npy_files = [os.path.join(dataroot,scene_id,task_dir,'npy',f'{p}.npy') for p in list(inst_dict.keys())]
            print(f"Using inst file paths: {len(all_npy_files)} files")

        # Use intelligent sampling function for sampling
        r2r_list = intelligent_sample_paths(all_npy_files, max_samples=MAX_EPISODE)
        print(f"Sampled {len(r2r_list)} paths from {len(all_npy_files)} total paths")

        pending_paths = []
        # for path_file in r2r_list[:50]:  # No longer need to limit to 50 here, sampling function already handles it
        for path_file in r2r_list:
            path_basename = path_file.split('/')[-1].split('.')[0]
            SUCCESS_MARKER_PATH = os.path.join(SAVE_DIR, path_basename+'.success')
            ATTEMPTS_MARKER_PATH = os.path.join(SAVE_DIR, path_basename+'.attempts')

            if os.path.exists(SUCCESS_MARKER_PATH):
                continue # Already succeeded, skip

            attempts = 0
            if os.path.exists(ATTEMPTS_MARKER_PATH):
                try:
                    with open(ATTEMPTS_MARKER_PATH, 'r') as f:
                        attempts = int(f.read().strip())
                except (ValueError, IOError):
                    pass

            if attempts >= MAX_ATTEMPTS:
                continue # Too many failures, skip

            # If we reach here, this is a pending path
            pending_paths.append(path_file)

        # If no pending paths, exit with special exit code
        if not pending_paths:
            print(f"All paths for scene {scene_id} are processed. Exiting.")
            sys.exit(EXIT_CODE_ALL_DONE)

        # Select a batch from the pending list
        paths_to_process_this_run = pending_paths[:BATCH_SIZE]
        print(f"Found {len(pending_paths)} pending paths. Processing a batch of {len(paths_to_process_this_run)}.")

        # 1. USD file path (make sure it has been modified)
        USD_FILE_PATH = os.path.join(usd_root,scene_id,f'start_result_navigation.usd')

        if not USD_FILE_PATH.startswith("omniverse://") and not os.path.exists(USD_FILE_PATH):
            carb.log_error(f"Error: Cannot find the specified local USD file: {USD_FILE_PATH}")
            simulation_app.close()
            exit()
        elif USD_FILE_PATH == "PATH_TO_YOUR_USD_FILE.usd":
            carb.log_warn(f"Warning: Please change the 'USD_FILE_PATH' variable in the script to your actual USD file path.")

        try:
            # Create World
            world = World(stage_units_in_meters=1.0)

            # Load USD file
            asset_prim = add_reference_to_stage(usd_path=USD_FILE_PATH, prim_path=PRIM_PATH_IN_STAGE)
            if not asset_prim or not asset_prim.IsValid():
                carb.log_error(f"Failed to load USD file: '{USD_FILE_PATH}'. Please check the path and file.")
                raise Exception("USD loading failed")
            else:
                carb.log_info(f"Successfully loaded '{USD_FILE_PATH}' to '{PRIM_PATH_IN_STAGE}'")
            asset_xform = XFormPrim(prim_path=PRIM_PATH_IN_STAGE,
                                            name=f"{asset_prim.GetName()}_xform")
            scale_factor = 1
            # 3. Set local scale (relative to parent Prim, usually /World)
            asset_xform.set_local_scale(np.array([scale_factor, scale_factor, scale_factor]))


            # --- Create and set up camera ---
            carb.log_info(f"Creating camera at path: {CAMERA_PRIM_PATH}")
            camera = Camera(
                prim_path=CAMERA_PRIM_PATH,        # Camera path in the Stage
                frequency=30,                  # Set data publishing frequency (Hz) - not important for single capture, but must be set
                resolution=(IMAGE_WIDTH, IMAGE_HEIGHT) # Set image resolution
            )
            # Initialize camera. This creates the camera prim and sets its properties, and adds it to the scene
            camera.initialize()


            ## zerui's setting
            camera.add_distance_to_image_plane_to_frame()
            camera.set_clipping_range(near_distance=0.01, far_distance=10000.0)# lsh
            focal_length = 10.0
            aperture = 2 * focal_length
            camera.set_focal_length(focal_length)
            camera.set_horizontal_aperture(aperture)
            camera.set_vertical_aperture(aperture)

            carb.log_info(f"Camera '{CAMERA_PRIM_PATH}' initialization complete.")
            world.reset()
            for _ in range(10): world.step(render=True)
            # Use sampled paths_to_process_this_run for processing
            for path_file in tqdm(paths_to_process_this_run,desc=f'Processing paths for {scene_id}'):

                # 7. Filename and path for saving images
                path_basename = path_file.split('/')[-1].split('.')[0]
                SUCCESS_MARKER_PATH = os.path.join(SAVE_DIR, path_basename+'.success') # <--- New: Define success marker file path
                ATTEMPTS_MARKER_PATH = os.path.join(SAVE_DIR, path_basename+'.attempts') # <--- New: Define attempts count file path

                # <--- New: Checkpoint resume check
                if os.path.exists(SUCCESS_MARKER_PATH):
                    continue

                # 2. Check failure count
                attempts = 0
                if os.path.exists(ATTEMPTS_MARKER_PATH):
                    try:
                        with open(ATTEMPTS_MARKER_PATH, 'r') as f:
                            attempts = int(f.read().strip())
                    except (ValueError, IOError):
                        attempts = 0 # If file is empty or content is wrong, reset to 0

                if attempts >= MAX_ATTEMPTS:
                    continue

                try:
                    signal.alarm(1200)  # let him coooookkk for 20 minutes

                    # Load waypoints
                    path_xy = np.load(path_file)
                    print(f"Loaded waypoints from {path_file}, shape: {path_xy.shape}")

                    # Load corresponding actions
                    action_file = path_file.replace('/npy/', '/actions/').replace('.npy', '.json').replace('path','actions')
                    print(f"Looking for action file: {action_file}")
                    if not os.path.exists(action_file):
                        carb.log_error(f"Cannot find corresponding action file: {action_file}")
                        print(f"Action file not found: {action_file}")
                        # Try other possible path formats
                        alt_action_file = path_file.replace('/npy/', '/actions/').replace('.npy', '.json')
                        print(f"Trying alternative path: {alt_action_file}")
                        if os.path.exists(alt_action_file):
                            action_file = alt_action_file
                            print(f"Found action file at alternative path: {action_file}")
                        else:
                            continue

                    with open(action_file, 'r') as f:
                        action_data = json.load(f)

                    # actions is a dict, the actual action list is under the 'actions' key
                    if isinstance(action_data, dict) and 'actions' in action_data:
                        actions = action_data['actions']
                        print(f"Loaded {len(actions)} actions from {action_file}")
                        print(f"Additional info - Forward distance: {action_data.get('forward_distance', 'N/A')}, Turn angle: {action_data.get('turn_angle', 'N/A')}")
                    else:
                        # If it's a direct list format (for compatibility)
                        actions = action_data
                        print(f"Loaded {len(actions)} actions from {action_file} (list format)")

                    # --- Run simulation loop ---
                    max_wait_frames = 100 # Wait a few frames for camera to stabilize and render
                    cur_frame = 0
                    global_frame = 0
                    heading_dict = {}

                    # Use remove_initial_turns to handle initial turns
                    actions, path_xy, initial_angle = remove_initial_turns(actions, path_xy)
                    # breakpoint()
                    print(f"After removing initial turns: {len(actions)} actions for {len(path_xy)} waypoints")
                    print(f"Initial angle: {initial_angle} radians ({math.degrees(initial_angle)} degrees)")

                    # Ensure actions and waypoints counts match
                    if len(actions) != len(path_xy):
                        print(f"Warning: Actions count ({len(actions)}) doesn't match waypoints count ({len(path_xy)})")

                    # Create output directory
                    OUTPUT_DIR = os.path.join(SAVE_DIR, path_basename)
                    os.makedirs(OUTPUT_DIR, exist_ok=True)

                    # Initialize variables
                    frame_index = 0  # For output file naming
                    current_heading = initial_angle  # Current heading angle (radians)

                    # Get initial position (point A) and first different position (point B)
                    if len(path_xy) < 2:
                        print(f"Path too short: {len(path_xy)} points")
                        continue

                    # First frame: at point B, facing from A to B
                    A_pos = path_xy[0]  # Starting point
                    B_pos = path_xy[1]  # First different point (guaranteed by remove_initial_turns)

                    # Set camera at point B
                    position_B = np.array([B_pos[0], B_pos[1], HEIGHT])

                    # Calculate orientation from A to B
                    position_A = np.array([A_pos[0], A_pos[1], HEIGHT])
                    if np.linalg.norm(position_B - position_A) > 1e-6:
                        A2B_rot_matrix = rot3_from_O_to_AB(O, position_A, position_B)
                    else:
                        A2B_rot_matrix = np.eye(3)

                    orientation = Quaternion(matrix=A2B_rot_matrix)
                    rotation = np.array([orientation.x, orientation.y, orientation.z, orientation.w])

                    # Set camera position and orientation
                    camera.set_world_pose(position=position_A,
                                        orientation=[rotation[3], rotation[0], rotation[1], rotation[2]]) # wxyz

                    # Capture the first frame image
                    print(f"Frame {frame_index}: Initial view at position B, looking from A to B")
                    for _ in range(10):
                        world.step(render=True)

                    # Save the first frame
                    frame_count = 0
                    image_saved = False
                    while not image_saved and camera.is_valid() and frame_count < max_wait_frames:
                        frame_count += 1
                        for _ in range(30):
                            world.step(render=True)

                        depth_data = camera.get_depth()
                        rgba_data = camera.get_rgba()

                        if rgba_data is not None and rgba_data.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 4):
                            if rgba_data.dtype != np.uint8:
                                rgba_data = np.clip(rgba_data * 255.0, 0, 255).astype(np.uint8)

                            image = Image.fromarray(rgba_data, "RGBA")
                            depth_image = normalize_depth_to_png(depth_data, 10.0)

                            OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, f"rgb_{frame_index}.png")
                            OUTPUT_DEPTH_PATH = os.path.join(OUTPUT_DIR, f"depth_{frame_index}")

                            try:
                                np.save(OUTPUT_DEPTH_PATH, depth_data)
                                cv2.imwrite(OUTPUT_DEPTH_PATH + '.png', depth_image)
                                image.save(OUTPUT_IMAGE_PATH)
                                print(f"[Frame {frame_index}] Initial frame saved to: {OUTPUT_IMAGE_PATH}")
                                image_saved = True
                                frame_index += 1
                            except Exception as e:
                                carb.log_error(f"Error saving image: {e}")
                                image_saved = True

                    # Keep current orientation
                    current_rotation = rotation.copy()

                    # Process actions, starting from the first action (corresponding to action from point A)
                    current_pos_idx = 0  # Currently at point B (path_xy[1])

                    for action_idx, action in enumerate(actions):
                        # If stop, end processing
                        if action == 0:
                            print(f"Action {action_idx}: STOP, ending video generation")
                            break

                        # All non-stop actions will move to the next point
                        if current_pos_idx + 1 >= len(path_xy):
                            print(f"Reached end of path at action {action_idx}")
                            break

                        # Move to next point
                        current_pos_idx += 1
                        current_pos = path_xy[current_pos_idx]
                        position = np.array([current_pos[0], current_pos[1], HEIGHT])

                        # Adjust orientation based on action type
                        if action == 1:  # forward
                            # Keep current orientation unchanged
                            print(f"Action {action_idx}: FORWARD to position {current_pos_idx}")
                        elif action == 2:  # turn left
                            # Turn left 15 degrees
                            print(f"Action {action_idx}: TURN LEFT 15° and move to position {current_pos_idx}")
                            cameraPose = np.concatenate((position, current_rotation))
                            cameraPose = rotate_camera_world_frame(cameraPose, [0, 0, 15])
                            current_rotation = cameraPose[3:]
                        elif action == 3:  # turn right
                            # Turn right 15 degrees
                            print(f"Action {action_idx}: TURN RIGHT 15° and move to position {current_pos_idx}")
                            cameraPose = np.concatenate((position, current_rotation))
                            cameraPose = rotate_camera_world_frame(cameraPose, [0, 0, -15])
                            current_rotation = cameraPose[3:]

                        # Set camera position and orientation
                        camera.set_world_pose(position=position,
                                           orientation=[current_rotation[3], current_rotation[0],
                                                      current_rotation[1], current_rotation[2]]) # wxyz

                        # Capture image
                        frame_count = 0
                        image_saved = False

                        # Run multiple simulation steps to stabilize camera
                        for _ in range(10):
                            world.step(render=True)

                        # Get and save image
                        while not image_saved and camera.is_valid() and frame_count < max_wait_frames:
                            frame_count += 1
                            for _ in range(30):
                                world.step(render=True)

                            depth_data = camera.get_depth()
                            rgba_data = camera.get_rgba()

                            if rgba_data is not None and rgba_data.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 4):
                                if rgba_data.dtype != np.uint8:
                                    rgba_data = np.clip(rgba_data * 255.0, 0, 255).astype(np.uint8)

                                image = Image.fromarray(rgba_data, "RGBA")
                                depth_image = normalize_depth_to_png(depth_data, 10.0)

                                OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, f"rgb_{frame_index}.png")
                                OUTPUT_DEPTH_PATH = os.path.join(OUTPUT_DIR, f"depth_{frame_index}")

                                try:
                                    np.save(OUTPUT_DEPTH_PATH, depth_data)
                                    cv2.imwrite(OUTPUT_DEPTH_PATH + '.png', depth_image)
                                    image.save(OUTPUT_IMAGE_PATH)
                                    print(f"[Frame {frame_index}] Action {action_idx} saved to: {OUTPUT_IMAGE_PATH}")
                                    image_saved = True
                                    frame_index += 1
                                except Exception as e:
                                    carb.log_error(f"Error saving image: {e}")
                                    image_saved = True

                        global_frame += 1

                    # Mark as complete
                    with open(SUCCESS_MARKER_PATH, 'w') as f:
                        f.write('done')
                    if os.path.exists(ATTEMPTS_MARKER_PATH):
                        os.remove(ATTEMPTS_MARKER_PATH)

                except Exception as e:
                    if isinstance(e, TimeoutException):
                        print(f"Path {path_basename} timed out. Logging as a failed attempt.")
                        with open(ATTEMPTS_MARKER_PATH, 'w') as f:
                            f.write('999') # lets forget about it
                    else:
                        # If it's another error
                        print(f"Path {path_basename} failed with an error: {e}. Logging as a failed attempt.")
                        print(f"Error type: {type(e).__name__}")
                        import traceback
                        print(f"Traceback:\n{traceback.format_exc()}")
                        attempts += 1
                        with open(ATTEMPTS_MARKER_PATH, 'w') as f:
                            f.write(str(attempts))

                finally:
                    signal.alarm(0)

        except Exception as e:

            carb.log_error(f"Critical error during simulation setup or execution: {e}")
            import traceback
            traceback.print_exc() # Print detailed error stack trace

        finally:
            # Ensure Isaac Sim application is closed
            carb.log_info("Closing SimulationApp...")
            simulation_app.close()
