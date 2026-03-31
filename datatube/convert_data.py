#!/usr/bin/env python3
"""
Convert VLN trajectory data  into InteriorNav format.
"""

import json
import gzip
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import datetime
from natsort import natsorted
from splits.split_utils import is_trainval


class DataConverter:
    def __init__(self, source_root, target_root, task_dir='goalnav_discrete', seq_dir='sequence_discrete'):
        """
        Args:
            source_root: path to raw trajectory data (e.g. /mnt/6t/dataset/vlnverse/)
            target_root: path to output InteriorNav-format data
            task_dir: subdirectory name for task outputs under each scene
            seq_dir: subdirectory name for rendered sequences under task_dir
        """
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.task_dir = task_dir
        self.seq_dir = seq_dir
        self.target_root.mkdir(parents=True, exist_ok=True)

        # Create raw_data directory
        self.raw_data_root = Path(str(target_root).replace('traj_data', 'raw_data'))
        self.raw_data_root.mkdir(parents=True, exist_ok=True)

        # Initialize error log
        self.error_log_file = self.target_root / "conversion_errors.log"
        try:
            with open(self.error_log_file, 'w', encoding='utf-8') as f:
                f.write(f"Data Conversion Error Log - Run started at {datetime.datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n")
            print(f"Log file: {self.error_log_file}")
        except Exception as e:
            print(f"Warning: cannot initialize log file {self.error_log_file}. Error: {e}")

    def _log_error(self, scene_id, goal_id, start_id, reason):
        """Log a skipped trajectory to console and log file."""
        traj_name = f"path_{goal_id}_{start_id}"
        log_message = f"SKIPPED: [Scene: {scene_id}, Traj: {traj_name} (Goal: {goal_id}, Start: {start_id})] - Reason: {reason}\n"

        print(f"⚠️  {log_message}", end="")

        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(log_message)
        except Exception as e:
            print(f"Critical warning: cannot write to log file! {e}")

    def resize_image(self, img, target_size=(256, 256)):
        """Resize image to target size."""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return np.array(img.resize(target_size, Image.LANCZOS))

    def calculate_orientation(self, positions, actions=None, rotation_angle=15):
        """
        Calculate orientation from path, accounting for in-place turns.
        Returns (orientations, yaws) as numpy arrays.
        """
        orientations = []
        yaws = []

        # Step 0: face toward Step 1
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]
        current_yaw = np.arctan2(dy, dx)
        yaws.append(current_yaw)

        # Step 1 onwards
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            distance = np.sqrt(dx*dx + dy*dy)

            if distance > 0.01:  # moved
                current_yaw = np.arctan2(dy, dx)
            elif actions is not None and i-1 >= 0:  # stationary, adjust by previous action
                action = actions[i-1]
                if action == 2:  # LEFT
                    current_yaw += np.radians(rotation_angle)
                elif action == 3:  # RIGHT
                    current_yaw -= np.radians(rotation_angle)

                current_yaw = np.arctan2(np.sin(current_yaw), np.cos(current_yaw))

            yaws.append(current_yaw)

        # Convert yaws to quaternions [w, x, y, z]
        for yaw in yaws:
            qw = np.cos(yaw / 2)
            qx = 0
            qy = 0
            qz = np.sin(yaw / 2)
            orientations.append([qw, qx, qy, qz])

        return np.array(orientations), np.array(yaws)

    def convert_trajectory(self, scene_id, goal_id, start_id, target_type):
        """
        Convert a single trajectory.
        target_type: 'fine' or 'coarse'
        Returns (episode_info, instruction_type) or (None, None) on failure.
        """
        traj_name = f"path_{goal_id}_{start_id}"
        source_scene_path = self.source_root / scene_id / self.task_dir

        # Check trajectory directory exists
        seq_path = source_scene_path / self.seq_dir / traj_name
        if not seq_path.exists():
            self._log_error(scene_id, goal_id, start_id, f"Trajectory directory not found: {seq_path}")
            return None, None

        # 1. Load raw data

        # Load RGB
        rgb_files = natsorted([f for f in seq_path.glob("rgb_*.png")], key=lambda p: p.name)
        if not rgb_files:
            self._log_error(scene_id, goal_id, start_id, "No RGB images found (rgb_*.png)")
            return None, None

        rgb_frames = []
        for rgb_file in rgb_files:
            img = Image.open(rgb_file).convert('RGB')
            img_resized = self.resize_image(img)
            rgb_frames.append(img_resized)
        rgb_array = np.array(rgb_frames)

        # Load Depth
        depth_files = natsorted([f for f in seq_path.glob("depth_*.npy")], key=lambda p: p.name)
        if not depth_files:
            self._log_error(scene_id, goal_id, start_id, "No depth maps found (depth_*.npy)")
            return None, None

        depth_frames = []
        for depth_file in depth_files:
            depth = np.load(depth_file)
            depth = np.nan_to_num(depth, nan=10.0, posinf=10.0, neginf=0.0)
            depth_resized = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_LINEAR)
            depth_frames.append(depth_resized)
        depth_array = np.array(depth_frames)
        if len(depth_array.shape) == 3:
            depth_array = np.expand_dims(depth_array, axis=-1)

        if depth_array.max() > 1.0:
            depth_array = depth_array / 100.0

        # Load Actions
        action_file = source_scene_path / "actions" / f"actions_{goal_id}_{start_id}.json"
        if not action_file.exists():
            self._log_error(scene_id, goal_id, start_id, f"Action file not found: {action_file}")
            return None, None
        try:
            with open(action_file) as f:
                actions = json.load(f)["actions"]
        except Exception as e:
            self._log_error(scene_id, goal_id, start_id, f"Failed to load action file: {e}")
            return None, None

        # Load Path NPY
        path_file = source_scene_path / "npy" / f"{traj_name}.npy"
        if not path_file.exists():
            self._log_error(scene_id, goal_id, start_id, f"Path NPY file not found: {path_file}")
            return None, None
        path_2d = np.load(path_file)

        # Verify data lengths are consistent
        num_steps = len(actions)
        if not (len(path_2d) == num_steps and len(rgb_files) == num_steps and len(depth_files) == num_steps):
            self._log_error(scene_id, goal_id, start_id,
                            f"Data length mismatch: Actions({num_steps}), Path({len(path_2d)}), RGB({len(rgb_files)}), Depth({len(depth_files)})")
            return None, None

        if num_steps < 2:
            self._log_error(scene_id, goal_id, start_id, "Trajectory too short (steps < 2), cannot compute orientation.")
            return None, None

        # Convert to 3D positions (add height=0)
        positions_3d = np.zeros((num_steps, 3))
        positions_3d[:, :2] = path_2d
        positions_3d[:, 2] = 0

        # Calculate orientation
        orientations, yaws = self.calculate_orientation(path_2d, actions, rotation_angle=15)

        # Progress values
        progress = np.linspace(0, 1, num_steps)

        # 2. Create target directory structure
        target_traj_dir = self.target_root / scene_id / f"{goal_id}_{start_id}"
        target_traj_dir.mkdir(parents=True, exist_ok=True)
        (target_traj_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (target_traj_dir / "meta").mkdir(parents=True, exist_ok=True)
        (target_traj_dir / "videos" / "chunk-000" / "observation.images.rgb").mkdir(parents=True, exist_ok=True)
        (target_traj_dir / "videos" / "chunk-000" / "observation.images.depth").mkdir(parents=True, exist_ok=True)

        # 3. Save Parquet file
        df_data = {
            'observation.camera_position': positions_3d.tolist(),
            'observation.camera_orientation': orientations.tolist(),
            'observation.camera_yaw': yaws.tolist(),
            'observation.robot_position': positions_3d.tolist(),
            'observation.robot_orientation': orientations.tolist(),
            'observation.robot_yaw': yaws.tolist(),
            'observation.progress': progress.tolist(),
            'observation.step': list(range(num_steps)),
            'observation.action': actions,
            'timestamp': list(range(num_steps)),
            'frame_index': list(range(num_steps)),
            'episode_index': [0] * num_steps,
            'index': list(range(num_steps)),
            'task_index': [0] * num_steps
        }

        df = pd.DataFrame(df_data)
        df.to_parquet(target_traj_dir / "data" / "chunk-000" / "episode_000000.parquet")

        # 4. Save image data
        np.save(target_traj_dir / "videos" / "chunk-000" / "observation.images.rgb" / "rgb.npy", rgb_array)
        np.save(target_traj_dir / "videos" / "chunk-000" / "observation.images.depth" / "depth.npy", depth_array)

        # 5. Create metadata
        goal_inst_file = source_scene_path / "goal_inst_aug_enhance.json"
        inst_seq_file = source_scene_path / "inst" / "inst_img_sequence.json"

        instruction_text = "Navigate to the goal."
        found_instruction = False
        instruction_type = None

        try:
            if target_type == 'fine':
                if inst_seq_file.exists():
                    with open(inst_seq_file, 'r', encoding='utf-8') as f:
                        inst_seq = json.load(f)
                        if traj_name in inst_seq and inst_seq[traj_name].get("instruction"):
                            instruction_text = inst_seq[traj_name]["instruction"]
                            found_instruction = True
                            instruction_type = 'fine'

                if not found_instruction:
                    self._log_error(scene_id, goal_id, start_id, "Assigned 'fine' but no fine-grained instruction found")
                    return None, None

            elif target_type == 'coarse':
                if goal_inst_file.exists():
                    with open(goal_inst_file, 'r', encoding='utf-8') as f:
                        goal_inst = json.load(f)
                        if str(goal_id) in goal_inst and goal_inst[str(goal_id)].get("instruction"):
                            instruction_text = goal_inst[str(goal_id)]["augmented_instructions"]
                            found_instruction = True
                            instruction_type = 'coarse'

                if not found_instruction:
                    self._log_error(scene_id, goal_id, start_id, "Assigned 'coarse' but no coarse-grained instruction found")
                    return None, None

        except Exception as e:
            self._log_error(scene_id, goal_id, start_id, f"JSON error while loading instructions: {e}")
            return None, None

        # Placeholder tokens
        instruction_tokens = [0] * 10

        # Check success status
        success_marker = seq_path.parent / f"{traj_name}.success"
        finish_status = "success" if success_marker.exists() else "unknown"

        episode_meta = {
            "episode_id": f"{goal_id}_{start_id}",
            "instruction_text": instruction_text,
            "instruction_tokens": instruction_tokens,
            "finish_status": finish_status,
            "fail_reason": None if finish_status == "success" else "unknown"
        }

        with open(target_traj_dir / "meta" / "episodes.jsonl", 'w') as f:
            f.write(json.dumps(episode_meta) + '\n')

        # info.json
        info = {
            "fps": 10,
            "video": True,
            "encoding": {"vcodec": "libx264"},
            "dataset": "your_dataset"
        }
        with open(target_traj_dir / "meta" / "info.json", 'w') as f:
            json.dump(info, f, indent=2)

        # tasks.jsonl
        task = {"task": "navigation", "instruction": instruction_text}
        with open(target_traj_dir / "meta" / "tasks.jsonl", 'w') as f:
            f.write(json.dumps(task) + '\n')

        episode_info = {
            "scene_id": scene_id,
            "trajectory_id": f"{goal_id}_{start_id}",
            "episode_id": f"{scene_id}_{goal_id}_{start_id}",
            "start_position": positions_3d[0].tolist(),
            "start_rotation": orientations[0].tolist(),
            "instruction_text": instruction_text,
            "instruction_tokens": instruction_tokens,
            "num_steps": num_steps,
            "full_path": positions_3d.tolist(),
            "goal_position": positions_3d[-1].tolist()
        }

        return episode_info, instruction_type

    def convert_scene(self, scene_id):
        """Convert all trajectories in a scene."""
        print(f"\nConverting scene: {scene_id}")
        print("=" * 60)

        source_scene_path = self.source_root / scene_id / self.task_dir
        if not source_scene_path.exists():
            print(f"⚠️ Scene path not found: {source_scene_path} (skipping)")
            return [], []

        seq_path = source_scene_path / self.seq_dir
        if not seq_path.exists():
            print(f"⚠️ sequence_discrete directory not found: {seq_path} (skipping)")
            return [], []

        trajectories = [d.name for d in seq_path.iterdir() if d.is_dir() and d.name.startswith("path_")]

        if not trajectories:
            print(f"⚠️ No trajectories found in scene {scene_id}")
            return [], []

        # Shuffle and split 50/50 for fine/coarse assignment
        np.random.shuffle(trajectories)
        split_point = int(len(trajectories) * 0.5)

        fine_grained_episodes = []
        coarse_grained_episodes = []

        for i, traj_name in enumerate(tqdm(trajectories, desc=f"Processing {scene_id}", unit="traj")):
            parts = traj_name.split("_")
            if len(parts) != 3:
                print(f"⚠️  SKIPPED: [Scene: {scene_id}, Traj: {traj_name}] - Reason: cannot parse goal/start ID from directory name")
                continue

            try:
                goal_id = int(parts[1])
                start_id = int(parts[2])
            except ValueError:
                print(f"⚠️  SKIPPED: [Scene: {scene_id}, Traj: {traj_name}] - Reason: goal/start ID is not an integer")
                continue

            target_type = 'fine' if i < split_point else 'coarse'

            episode_info, inst_type = self.convert_trajectory(scene_id, goal_id, start_id, target_type)

            if episode_info:
                if inst_type == 'fine':
                    fine_grained_episodes.append(episode_info)
                elif inst_type == 'coarse':
                    coarse_grained_episodes.append(episode_info)

        return fine_grained_episodes, coarse_grained_episodes

    def create_json_metadata(self, episodes, split="train"):
        """Create JSON metadata files (both .json and .json.gz)."""
        output_dir = self.raw_data_root / split
        output_dir.mkdir(parents=True, exist_ok=True)

        formatted_episodes = []
        for ep in episodes:
            formatted_ep = {
                "start_position": ep["start_position"],
                "start_rotation": ep["start_rotation"],
                "instruction": {
                    "instruction_text": ep["instruction_text"],
                    "instruction_tokens": ep["instruction_tokens"]
                },
                "reference_path": ep["full_path"],
                "goals": {
                    "position": ep["goal_position"],
                    "radius": 3.0
                },
                "info": {
                    "geodesic_distance": -1  # placeholder
                },
                "scan": ep["scene_id"],
                "episode_id": ep["episode_id"],
                "trajectory_id": ep["episode_id"],
                "scene_id": f"vlnverse/{ep['scene_id']}"
            }
            formatted_episodes.append(formatted_ep)

        output_file = output_dir / f"{split}.json"
        with open(output_file, 'w') as f:
            json.dump({"episodes": formatted_episodes}, f, indent=2)

        output_file_gz = output_dir / f"{split}.json.gz"
        with gzip.open(output_file_gz, 'wt', encoding='utf-8') as f:
            json.dump({"episodes": formatted_episodes}, f)

        print(f"✅ Created {split} metadata: {len(formatted_episodes)} episodes")


def main():
    np.random.seed(42)

    # Configure paths
    source_root = "/mnt/6t/dataset/vlnverse"
    target_root = "/data/dataset/vlnverse/traj_data/vlnverse"
    task_dir = "goalnav_discrete"
    seq_dir = "sequence_discrete"
    splits_file = "splits/scene_splits.json"

    converter = DataConverter(source_root, target_root, task_dir, seq_dir)

    # Discover scenes
    scenes_path = Path(source_root)
    if not scenes_path.exists():
        print(f"Error: source directory does not exist: {source_root}")
        return

    scenes = [d.name for d in scenes_path.iterdir() if d.is_dir()]
    if not scenes:
        print(f"Error: no scene directories found in {source_root}")
        return

    print(f"Found {len(scenes)} scenes. Starting conversion...")

    all_fine_grained_episodes = []
    all_coarse_grained_episodes = []

    for scene in scenes:
        if not is_trainval(splits_file, scene):
            continue
        fine_eps, coarse_eps = converter.convert_scene(scene)
        all_fine_grained_episodes.extend(fine_eps)
        all_coarse_grained_episodes.extend(coarse_eps)

    if all_fine_grained_episodes:
        converter.create_json_metadata(all_fine_grained_episodes, "all_fine_grained")
    else:
        print("No fine-grained trajectories converted.")

    if all_coarse_grained_episodes:
        converter.create_json_metadata(all_coarse_grained_episodes, "all_coarse_grained")
    else:
        print("No coarse-grained trajectories converted.")

    print("\n" + "=" * 80)
    print("Conversion complete!")
    print(f"Total converted (Fine-grained): {len(all_fine_grained_episodes)} trajectories")
    print(f"Total converted (Coarse-grained): {len(all_coarse_grained_episodes)} trajectories")
    print(f"Output directory: {target_root}")
    print(f"Error log: {converter.error_log_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
