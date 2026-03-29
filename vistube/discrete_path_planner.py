#!/usr/bin/env python3
"""
Discrete path planner
Implements A* algorithm path planning based on a discrete action space
Action space: forward 0.25m, turn left 15 degrees, turn right 15 degrees
"""

import numpy as np
import math
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import copy
import matplotlib.pyplot as plt
import cv2

from vistube.tube_utils import remove_initial_turns

# Action definitions
ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3

# Action parameters
FORWARD_DISTANCE = 0.25  # Forward distance (meters) - exact distance per step
TURN_ANGLE = 15  # Turn angle (degrees) - exact angle per turn
TURN_ANGLE_RAD = math.radians(TURN_ANGLE)  # Turn angle (radians)

# Path planning parameters (global thresholds)
GOAL_REACH_THRESHOLD = 0.375  # Goal reach threshold (meters) - default is 1.5x forward distance
PARTIAL_PATH_THRESHOLD = 10.0  # Maximum distance for accepting partial paths (meters)
STATE_POSITION_PRECISION = 0.125  # State space position precision (meters) - used for deduplication, does not affect actual movement distance
STATE_ANGLE_PRECISION = 15  # State space angle precision (degrees) - used for deduplication, does not affect actual turn angle

@dataclass
class DiscreteState:
    """Discrete state: contains position and orientation"""
    x: float
    y: float
    angle: float  # radians

    def __hash__(self):
        # Discretize position and angle for hashing (used for state deduplication, does not affect actual movement)
        x_discrete = round(self.x / STATE_POSITION_PRECISION)
        y_discrete = round(self.y / STATE_POSITION_PRECISION)
        angle_discrete = round(math.degrees(self.angle) / STATE_ANGLE_PRECISION)
        return hash((x_discrete, y_discrete, angle_discrete))

    def __eq__(self, other):
        if not isinstance(other, DiscreteState):
            return False
        # Check whether two states are equal (used for deduplication, does not affect actual movement)
        return (abs(self.x - other.x) < STATE_POSITION_PRECISION and
                abs(self.y - other.y) < STATE_POSITION_PRECISION and
                abs(self.angle - other.angle) < math.radians(STATE_ANGLE_PRECISION))

    def distance_to(self, other):
        """Calculate Euclidean distance to another state"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class DiscretePathPlanner:
    """Discrete path planner"""

    def __init__(self, occupancy_map, occ_world_coords, pixel_size=None):
        """
        Initialize the discrete path planner

        Args:
            occupancy_map: Occupancy grid map (1=traversable, 0=obstacle)
            occ_world_coords: World coordinate mapping for the occupancy grid
            pixel_size: Pixel size (meters/pixel)
        """
        self.occupancy_map = occupancy_map
        self.occ_map = occ_world_coords
        self.height, self.width = occupancy_map.shape

        # Calculate pixel size
        if pixel_size is None:
            self.pixel_size = abs(occ_world_coords[0, 2] - occ_world_coords[0, 1])
        else:
            self.pixel_size = pixel_size

        # Collision detection parameters
        self.robot_radius = 0.1  # Robot radius (meters), reduced appropriately to improve passability
        self.safety_margin = 0.03  # Safety margin (meters)

    def world_to_pixel(self, world_x, world_y):
        """Convert world coordinates to pixel coordinates"""
        x_idx = np.argmin(np.abs(self.occ_map[0] - world_x))
        y_idx = np.argmin(np.abs(self.occ_map[:, 0] - world_y))
        return x_idx, y_idx

    def pixel_to_world(self, pixel_x, pixel_y):
        """Convert pixel coordinates to world coordinates"""
        world_x = self.occ_map[0, pixel_x]
        world_y = self.occ_map[pixel_y, 0]
        return world_x, world_y

    def is_valid_state(self, state: DiscreteState, is_goal=False) -> bool:
        """Check whether a state is valid (no collision)

        Args:
            state: The state to check
            is_goal: Whether this is a goal state (goal states use more lenient collision detection)
        """
        px, py = self.world_to_pixel(state.x, state.y)

        # Check whether within map bounds
        if px < 0 or px >= self.width or py < 0 or py >= self.height:
            return False

        # For goal states, use a smaller collision detection radius
        if is_goal:
            check_radius = max(1, int(self.robot_radius * 0.5 / self.pixel_size))
        else:
            check_radius = int((self.robot_radius + self.safety_margin) / self.pixel_size)

        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.occupancy_map[ny, nx] == 0:  # Obstacle
                        return False

        return True

    def get_successors(self, state: DiscreteState) -> List[Tuple[DiscreteState, int, float]]:
        """
        Get all successor states of the current state

        Returns:
            List of (next_state, action, cost)
        """
        successors = []

        # Forward action - try multiple forward steps to avoid getting stuck
        for forward_mult in [1]:  # Can try [1, 2] to allow double forward
            new_x = state.x + FORWARD_DISTANCE * forward_mult * math.cos(state.angle)
            new_y = state.y + FORWARD_DISTANCE * forward_mult * math.sin(state.angle)
            forward_state = DiscreteState(new_x, new_y, state.angle)

            if self.is_valid_state(forward_state):
                # Single step forward
                if forward_mult == 1:
                    successors.append((forward_state, ACTION_FORWARD, FORWARD_DISTANCE))
                # If needed, double step forward can be added
                # else:
                #     successors.append((forward_state, ACTION_FORWARD, FORWARD_DISTANCE * forward_mult))

        # Combined turn-then-forward actions (improve reachability)
        for angle_delta in [TURN_ANGLE_RAD, -TURN_ANGLE_RAD]:
            new_angle = state.angle + angle_delta
            # Normalize angle
            while new_angle > math.pi:
                new_angle -= 2 * math.pi
            while new_angle < -math.pi:
                new_angle += 2 * math.pi

            # Only add turn actions
            if angle_delta > 0:
                turn_state = DiscreteState(state.x, state.y, new_angle)
                successors.append((turn_state, ACTION_TURN_LEFT, TURN_ANGLE_RAD * 0.1))
            else:
                turn_state = DiscreteState(state.x, state.y, new_angle)
                successors.append((turn_state, ACTION_TURN_RIGHT, TURN_ANGLE_RAD * 0.1))

        return successors

    def heuristic(self, state: DiscreteState, goal: DiscreteState) -> float:
        """Heuristic function: estimate cost to goal"""
        # Euclidean distance
        distance = state.distance_to(goal)

        # Angle difference (how many turns needed to face the goal)
        dx = goal.x - state.x
        dy = goal.y - state.y
        if abs(dx) > 0.01 or abs(dy) > 0.01:  # Avoid division by zero
            target_angle = math.atan2(dy, dx)
            angle_diff = abs(target_angle - state.angle)
            # Normalize angle difference to [0, pi]
            while angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            # Estimate number of turns needed
            turn_steps = angle_diff / TURN_ANGLE_RAD
            turn_cost = turn_steps * TURN_ANGLE_RAD * 0.1
        else:
            turn_cost = 0

        return distance + turn_cost

    def plan_discrete_path(self, start_pos: Tuple[float, float],
                          goal_pos: Tuple[float, float],
                          initial_angle: Optional[float] = None,
                          max_iterations: int = 50000) -> Optional[Tuple[List[int], List[DiscreteState]]]:
        """
        Plan a discrete path

        Args:
            start_pos: Start position (x, y) in world coordinates
            goal_pos: Goal position (x, y) in world coordinates
            initial_angle: Initial orientation (radians), None means auto-orient toward goal
            max_iterations: Maximum number of iterations

        Returns:
            (actions, states): Action sequence and state sequence, or None if no path is found
        """
        # Initialize start state
        if initial_angle is None:
            # Automatically calculate angle toward goal
            dx = goal_pos[0] - start_pos[0]
            dy = goal_pos[1] - start_pos[1]
            initial_angle = math.atan2(dy, dx)

        start_state = DiscreteState(start_pos[0], start_pos[1], initial_angle)
        goal_state = DiscreteState(goal_pos[0], goal_pos[1], 0)  # Goal angle does not matter

        # Check validity of start and goal states
        if not self.is_valid_state(start_state):
            print(f"Warning: Start state is invalid (collision)")
            return None
        if not self.is_valid_state(goal_state, is_goal=True):  # Goal uses more lenient detection
            print(f"Warning: Goal state is invalid (collision)")
            return None

        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, id(start_state), start_state, []))

        closed_set = set()
        g_scores = {start_state: 0}

        iterations = 0
        best_state = start_state
        best_distance = start_state.distance_to(goal_state)
        best_path = []

        while open_set and iterations < max_iterations:
            iterations += 1

            _, _, current_state, path = heapq.heappop(open_set)

            # Check whether the goal is reached (only consider position, not orientation)
            distance_to_goal = current_state.distance_to(goal_state)
            # Use global threshold to determine if goal is reached
            if distance_to_goal < GOAL_REACH_THRESHOLD:
                # Reconstruct path
                actions = []
                states = [start_state]

                for action, next_state in path:
                    actions.append(action)
                    states.append(next_state)

                actions.append(ACTION_STOP)  # Add stop action
                print(f"Successfully reached goal! Final distance: {distance_to_goal:.2f}m")
                return actions, states

            # Record the state closest to the goal
            if distance_to_goal < best_distance:
                best_distance = distance_to_goal
                best_state = current_state
                best_path = path

            if current_state in closed_set:
                continue

            closed_set.add(current_state)

            # Expand successor nodes
            for next_state, action, cost in self.get_successors(current_state):
                if next_state in closed_set:
                    continue

                tentative_g = g_scores[current_state] + cost

                if next_state not in g_scores or tentative_g < g_scores[next_state]:
                    g_scores[next_state] = tentative_g
                    f_score = tentative_g + self.heuristic(next_state, goal_state)

                    new_path = path + [(action, next_state)]
                    heapq.heappush(open_set, (f_score, id(next_state), next_state, new_path))

        print(f"Warning: Could not find complete path after {iterations} iterations")
        print(f"Best distance to goal: {best_distance:.2f}m")

        # Use global threshold to determine whether to accept partial path
        if best_path and best_distance < PARTIAL_PATH_THRESHOLD:
            actions = [action for action, _ in best_path]
            states = [start_state] + [state for _, state in best_path]
            actions.append(ACTION_STOP)
            print(f"Returning partial path (reached within {best_distance:.2f}m of goal)")
            return actions, states

        return None

    def visualize_discrete_path(self, actions: List[int], states: List[DiscreteState],
                               save_path: Optional[str] = None):
        """Visualize discrete path"""
        fig, ax = plt.subplots(figsize=(12, 12))

        # Display occupancy grid map
        ax.imshow(self.occupancy_map, cmap='gray', origin='lower')

        # Convert states to pixel coordinates
        pixels = []
        for state in states:
            px, py = self.world_to_pixel(state.x, state.y)
            pixels.append((px, py))

        if pixels:
            pixels = np.array(pixels)
            # Draw path
            ax.plot(pixels[:, 0], pixels[:, 1], 'g-', linewidth=2, label='Discrete Path')
            ax.plot(pixels[:, 0], pixels[:, 1], 'g.', markersize=4)

            # Draw start and end points
            ax.plot(pixels[0, 0], pixels[0, 1], 'go', markersize=10, label='Start')
            ax.plot(pixels[-1, 0], pixels[-1, 1], 'ro', markersize=10, label='End')

            # Draw orientation arrows
            arrow_interval = max(1, len(states) // 20)
            for i in range(0, len(states), arrow_interval):
                state = states[i]
                px, py = pixels[i]
                # Arrow length (pixels)
                arrow_len = 5
                dx = arrow_len * math.cos(state.angle)
                dy = arrow_len * math.sin(state.angle)
                ax.arrow(px, py, dx, dy, head_width=3, head_length=2,
                        fc='blue', alpha=0.6)

        # Add action statistics
        action_counts = {
            ACTION_FORWARD: 0,
            ACTION_TURN_LEFT: 0,
            ACTION_TURN_RIGHT: 0,
            ACTION_STOP: 0
        }
        for action in actions:
            action_counts[action] += 1

        stats_text = f'Actions: Forward={action_counts[ACTION_FORWARD]}, ' \
                    f'Left={action_counts[ACTION_TURN_LEFT]}, ' \
                    f'Right={action_counts[ACTION_TURN_RIGHT]}'
        ax.set_title(f'Discrete Path Planning\n{stats_text}')

        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()
        return fig


def convert_continuous_path_to_discrete(continuous_path: List[Tuple[float, float]],
                                       forward_distance: float = FORWARD_DISTANCE,
                                       turn_angle: float = TURN_ANGLE) -> List[int]:
    """
    Convert a continuous path to a discrete action sequence
    This is a post-processing method for discretizing an existing continuous path

    Args:
        continuous_path: List of continuous path points [(x1,y1), (x2,y2), ...]
        forward_distance: Forward distance
        turn_angle: Turn angle (degrees)

    Returns:
        List of actions
    """
    if len(continuous_path) < 2:
        return [ACTION_STOP]

    actions = []

    # Initial orientation: facing the first path segment
    dx = continuous_path[1][0] - continuous_path[0][0]
    dy = continuous_path[1][1] - continuous_path[0][1]
    current_angle = math.atan2(dy, dx)

    current_pos = list(continuous_path[0])

    for i in range(1, len(continuous_path)):
        target = continuous_path[i]

        # Keep approaching the target point
        while True:
            # Calculate distance and angle to target
            dx = target[0] - current_pos[0]
            dy = target[1] - current_pos[1]
            distance = math.sqrt(dx**2 + dy**2)

            if distance < forward_distance * 0.5:
                break  # Reached near the current target point

            # Calculate required orientation
            target_angle = math.atan2(dy, dx)

            # Calculate angle difference
            angle_diff = target_angle - current_angle
            # Normalize to [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # If turning is needed
            if abs(angle_diff) > math.radians(turn_angle * 0.5):
                if angle_diff > 0:
                    actions.append(ACTION_TURN_LEFT)
                    current_angle += math.radians(turn_angle)
                else:
                    actions.append(ACTION_TURN_RIGHT)
                    current_angle -= math.radians(turn_angle)
            else:
                # Move forward
                actions.append(ACTION_FORWARD)
                current_pos[0] += forward_distance * math.cos(current_angle)
                current_pos[1] += forward_distance * math.sin(current_angle)

    actions.append(ACTION_STOP)
    return actions


# Helper functions for integration with the existing system
def get_discrete_path(occupancy_map, occ_world_coords, start_point, end_point,
                     initial_angle=None, visualize=False):
    """
    Convenience interface for discrete path planning

    Args:
        occupancy_map: Occupancy grid map (1=traversable, 0=obstacle)
        occ_world_coords: World coordinate mapping
        start_point: Start point (x, y) in pixel coordinates
        end_point: End point (x, y) in pixel coordinates
        initial_angle: Initial orientation (degrees), None means auto
        visualize: Whether to visualize

    Returns:
        (actions, path_world): Action sequence and world coordinate path
    """
    planner = DiscretePathPlanner(occupancy_map, occ_world_coords)

    # Convert pixel coordinates to world coordinates
    start_world = planner.pixel_to_world(start_point[0], start_point[1])
    end_world = planner.pixel_to_world(end_point[0], end_point[1])

    # Convert initial angle to radians
    if initial_angle is not None:
        initial_angle = math.radians(initial_angle)

    # Plan path
    result = planner.plan_discrete_path(start_world, end_world, initial_angle)

    if result is None:
        return None, None, None

    actions, states = result

    # Convert states to world coordinate path
    path_world = [(state.x, state.y) for state in states]

    if visualize:
        planner.visualize_discrete_path(actions, states)

    return actions, np.array(path_world), states


if __name__ == "__main__":
    # Test code
    print("Discrete Path Planner Module Loaded")
    print(f"Action Space: Forward={FORWARD_DISTANCE}m, Turn={TURN_ANGLE}°")
    print("Actions: 0=Stop, 1=Forward, 2=TurnLeft, 3=TurnRight")
