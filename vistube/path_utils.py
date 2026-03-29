import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon as PolygonPatch # Renamed on import to avoid conflicts
from math import hypot
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.grid import Grid
from pathfinding.core.diagonal_movement import DiagonalMovement
import copy
from vlntube.path_finder import TimeLimitedBiAStarFinder
from vlntube.path_finder import simplify_path_with_collision_check
from scipy.interpolate import splprep, splev
from sklearn_extra.cluster import KMedoids
from scipy.ndimage import gaussian_filter1d # Gaussian smoothing
from pyquaternion import Quaternion
from PIL import Image
import cv2
from vlntube.tube_utils import rot3_from_O_to_AB, DEFAULT_CAMERA_FORWARD

O = DEFAULT_CAMERA_FORWARD

# def visualize_and_save_result(binary_map, polygon_vertices, sampled_point, filename="sampling_visualization.png"):
#     """
#     Visualize the sampling result and save it as an image.

#     Args:
#         binary_map (np.ndarray): Binary map (1=passable, 0=obstacle).
#         polygon_vertices (list): List of polygon vertices.
#         sampled_point (tuple): The sampled point (x, y), or None if not marked.
#         filename (str): Filename for saving the image.
#     """
#     height, width = binary_map.shape

#     fig, ax = plt.subplots(figsize=(8, 8))

#     # Draw binary map (0=black obstacle, 1=white passable area)
#     # Set origin='lower' to fix the upside-down issue
#     ax.imshow(binary_map, cmap='gray', origin='upper', vmin=0, vmax=1)

#     # Draw polygon area
#     polygon = PolygonPatch(polygon_vertices, facecolor='blue', alpha=0.4, label='Polygon Area')
#     ax.add_patch(polygon)

#     # Mark the sampled point
#     if sampled_point:
#         x, y = sampled_point
#         ax.scatter(x, y, c='red', s=100, edgecolors='black', zorder=5, label='Sampled Point')

#     ax.set_xlim(0, width -1)
#     ax.set_ylim(0, height -1)
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_title("Walkable Point Sampling in Polygon (1 = Walkable)")
#     ax.legend()

#     plt.savefig(filename)
#     print(f"Visualization saved to '{filename}'")
#     plt.close(fig)

def visualize_and_save_result(binary_map, polygon_vertices, sampled_point, filename="sampling_visualization.png"):
    """
    Visualize the sampling result and save it as an image.
    This version ensures the image orientation is consistent with the intuitive
    representation of the Numpy array.

    Args:
        binary_map (np.ndarray): Binary map (1=passable, 0=obstacle).
        polygon_vertices (list): List of polygon vertices.
        sampled_point (tuple): The sampled point (x, y), or None if not marked.
        filename (str): Filename for saving the image.
    """
    height, width = binary_map.shape

    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Draw binary map (0=black obstacle, 1=white passable area)
    #    We do not set the origin parameter, keeping the default 'upper'
    ax.imshow(binary_map, cmap='gray', vmin=0, vmax=1)

    # 2. Draw polygon area
    polygon = PolygonPatch(polygon_vertices, facecolor='blue', alpha=0.4, label='Polygon Area')
    ax.add_patch(polygon)

    # 3. Mark the sampled point
    from collections.abc import Iterable
    # breakpoint()
    sampled_point = np.array(sampled_point)
    # if isinstance(sampled_point,np.ndarray):
    if len(sampled_point.shape)>1:
        for i in sampled_point:
            x,y = i
            ax.scatter(x, y, c='red', s=100, edgecolors='black', zorder=5, label='Sampled Point')
    else:
        try:
            x, y = sampled_point
            # Scatter plot coordinates automatically adapt to the imshow coordinate system
            ax.scatter(x, y, c='red', s=100, edgecolors='black', zorder=5, label='Sampled Point')
        except:
            pass

    # 4. Key fix: manually flip the Y-axis display range
    #    By default, 'upper' makes the Y-axis increase top-to-bottom (0, 1, 2...)
    #    To make it look like a standard image, we manually set the Y-axis range
    #    so the bottom is height-1 and the top is 0
    ax.set_ylim(height - 1, 0)
    ax.set_xlim(0, width - 1)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Walkable Point Sampling (Image Coordinate System)")
    ax.legend()

    plt.savefig(filename)
    print(f"Visualization saved to '{filename}'")
    plt.close(fig) # Close image to free memory

def vis_gray(gray_map,name='gray_map.png'):
    '''
    Save gray map for visualization
    '''
    # Plot the grayscale map
    plt.imshow(gray_map, cmap='gray', interpolation='nearest')
    plt.colorbar(label='Intensity')
    plt.title('Gray Path Map')
    plt.axis('off')

    # Save the plot as an image file
    plt.savefig(name, bbox_inches='tight', dpi=300)

    # Close the plot explicitly
    plt.close()

def vis(data,name='binary_map.png'):
    '''
    Another visualization method in case that the points
    are very small
    '''
    cmap   = ListedColormap(["black", "red"])
    bounds = [-0.5, 0.5, 1.5]      # Everything <0.5->0-bin, >0.5->1-bin
    norm   = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6, 6))     # Bigger figure makes each cell visible
    plt.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
    plt.axis("off")

    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.close()

def sample_walkable_point_in_polygon(binary_map, polygon_vertices):
    """
    Randomly sample a passable point within a polygon area.

    Args:
        binary_map (np.ndarray): A 2D numpy array where 0 means passable and 1 means obstacle.
        polygon_vertices (list): A list of tuples, each tuple is a polygon vertex (x, y).

    Returns:
        tuple: Coordinates of a randomly sampled passable point (x, y), or None if no passable point exists.
    """
    # Get map width and height
    height, width = binary_map.shape

    # Create a grid containing all coordinate points
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    all_points = np.vstack((x.ravel(), y.ravel())).T

    # Create a polygon path
    polygon_path = Path(polygon_vertices)

    # Determine which points are inside the polygon
    is_inside_polygon = polygon_path.contains_points(all_points)

    # Reshape the boolean mask to match the map shape
    inside_mask = is_inside_polygon.reshape((height, width))

    # Find passable points inside the polygon (value is 0)
    walkable_points = np.argwhere((binary_map == 1) & (inside_mask))

    if walkable_points.size == 0:
        return None  # No passable point found

    # Randomly select a passable point
    # np.argwhere returns coordinates as (row, col), i.e. (y, x)
    random_point_yx = random.choice(walkable_points)

    # Convert to (x, y) format and return
    return (random_point_yx[1], random_point_yx[0])

def densify(path, step=5.0):
    """Insert points every `step` units; keeps originals."""
    dense = [path[0]]
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        dx, dy   = x2 - x1, y2 - y1
        length   = hypot(dx, dy)
        if length == 0:
            continue
        n = int(length // step)       # how many interior points
        for k in range(1, n + 1):
            t = k * step / length
            dense.append((x1 + dx * t, y1 + dy * t))
    dense.append(path[-1])
    return dense

def get_path(matrix, start_p, end_p):
    '''
    matrix: binary, 0 is obstacle and 1 is passable
    start_p: start point (x,y)
    end_p: end point (x,y)
    '''
    grid = Grid(matrix = matrix.tolist())
    start =grid.node(start_p[0],start_p[1])

    end = grid.node(end_p[0],end_p[1])

    # print(matrix[start.y,start.x], matrix[end.y,end.x])
    # finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    # lsh_path, runs = finder.find_path(start,end,grid)
    finder = TimeLimitedBiAStarFinder(time_limit=10, diagonal_movement=DiagonalMovement.always)
    lsh_path = finder.find_path(start,end,grid)
    return lsh_path

def smooth_path_spline(points, num_points=100, degree=3, smoothing=0.1):
    """
    Smooth a path using B-spline curves.
    - num_points: Number of points on the generated new path.
    - degree: Degree of the spline curve (usually 2 or 3).
    - smoothing: Smoothing factor; larger values produce smoother curves but may deviate more from original points.
    """
    points = np.array(points).T
    x, y = points

    # splprep (spline preparation) finds the B-spline parameters representing the curve
    tck, u = splprep([x, y], s=smoothing, k=degree)

    # Generate new, denser points using parameters between 0 and 1
    u_new = np.linspace(0, 1, num_points)

    # splev (spline evaluation) computes the coordinate points for the new parameters
    x_new, y_new = splev(u_new, tck)

    return np.array([x_new, y_new]).T

def smooth_path_average(points, window_size=3):
    """
    Smooth a path using moving average.
    Larger window_size produces a smoother path but deviates more from the original.
    """
    if window_size < 3:
        return points

    points_arr = np.array(points, dtype=float)
    new_points = np.copy(points_arr)
    half_window = window_size // 2

    for i in range(half_window, len(points_arr) - half_window):
        # Compute the average of points within the sliding window
        window = points_arr[i - half_window : i + half_window + 1]
        new_points[i] = np.mean(window, axis=0)

    return new_points

def smooth_path_conditional(points, window_size=5):
    """
    Smoothing function with conditional logic: only smooth turning points, keep straight segments unchanged.

    Args:
        points (np.array): Original path point array.
        window_size (int): Window size for moving average, must be odd.

    Returns:
        np.array: New smoothed path point array.
    """
    if len(points) < 3:
        return points

    points_arr = np.array(points, dtype=float)
    # Create a copy; we will modify the points that need smoothing on this copy
    smoothed_points = np.copy(points_arr)

    # Find indices of all points that need smoothing (i.e. "turning points")
    smooth_indices = []
    for i in range(1, len(points_arr) - 1):
        p_prev = points_arr[i-1]
        p_curr = points_arr[i]
        p_next = points_arr[i+1]

        # Check if the current point lies on a straight line
        # Condition: x-coordinates of prev/curr/next are all equal (vertical line)
        # or y-coordinates are all equal (horizontal line)
        is_on_straight_line = (p_prev[0] == p_curr[0] == p_next[0]) or \
                              (p_prev[1] == p_curr[1] == p_next[1])

        # If not on a straight line, this point needs smoothing
        if not is_on_straight_line:
            smooth_indices.append(i)

    # Apply moving average smoothing to all identified turning points
    half_window = window_size // 2
    for i in smooth_indices:
        # Ensure the window does not go out of bounds
        start = max(0, i - half_window)
        end = min(len(points_arr), i + half_window + 1)

        window = points_arr[start:end]
        smoothed_points[i] = np.mean(window, axis=0)

    return smoothed_points

def simplify_path(points, tolerance=1e-2):
    """
    Simplify a path by removing points that are nearly collinear with their neighbors.
    This effectively removes small "zigzags".
    """
    if len(points) < 3:
        return points

    simplified_points = [points[0]]
    for i in range(1, len(points) - 1):
        p_prev = np.array(simplified_points[-1])
        p_curr = np.array(points[i])
        p_next = np.array(points[i+1])

        # Compute the two vectors (forward and backward)
        vec1 = p_curr - p_prev
        vec2 = p_next - p_curr

        # Normalize the vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 > 0 and norm2 > 0:
            vec1_norm = vec1 / norm1
            vec2_norm = vec2 / norm2
            # Compute the dot product; a value close to 1 means nearly the same direction (collinear)
            dot_product = np.dot(vec1_norm, vec2_norm)
            if dot_product < 1.0 - tolerance:
                simplified_points.append(points[i])
        else:
            # If points coincide, add them as well
            simplified_points.append(points[i])

    simplified_points.append(points[-1])
    return np.array(simplified_points)

def find_representative_points(points, n_representatives):
    """
    Use the K-Medoids algorithm to find the n most representative real points from a set of points.

    Args:
        points (np.array): An Nx2 Numpy array containing all 2D point coordinates.
        n_representatives (int): The number of representative points (cluster centers) to keep, i.e. n.

    Returns:
        tuple: A tuple containing:
            - medoids (np.array): Coordinates of the n representative points (n x 2 array).
            - medoid_indices (np.array): Indices of the n representative points in the original points array.
            - labels (np.array): Cluster labels for each original point (array of length N).
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    if len(points) < n_representatives:
        print("Error: The number of data points is fewer than the number of representative points to find.")
        return None, None, None

    # 1. Create KMedoids model
    # random_state ensures reproducible results across runs
    kmedoids = KMedoids(n_clusters=n_representatives, random_state=0)

    # 2. Fit the data
    kmedoids.fit(points)

    # 3. Extract results
    # Get the n center points (Medoids), which are all original data points
    medoids = kmedoids.cluster_centers_
    # Get the indices of these n center points in the original array
    medoid_indices = kmedoids.medoid_indices_
    # Get the cluster label for each data point
    labels = kmedoids.labels_

    return medoids, medoid_indices, labels

def correct_path_jitters(points, thresholds={1, 2}):
    """
    Correct minor jitters in a path.
    If the x or y coordinate difference between adjacent points falls within the threshold set,
    align them.

    Args:
        points (list or np.array): Original path point list in format [[x1,y1], [x2,y2], ...].
        thresholds (set): A set of allowed deviation values, e.g. {1, 3}.

    Returns:
        np.array: Corrected new path point array.
    """
    if len(points) < 2:
        return np.array(points)

    # Ensure input is a Numpy array for easier manipulation
    points_arr = np.array(points)

    # Initialize the corrected path list and add the first point
    corrected_path = [points_arr[0].copy()]

    # Iterate starting from the second point
    for i in range(1, len(points_arr)):
        current_point = points_arr[i].copy() # Use a copy to avoid modifying original data
        previous_corrected_point = corrected_path[-1]

        # Compute coordinate differences with the previous corrected point
        dx = abs(current_point[0] - previous_corrected_point[0])
        dy = abs(current_point[1] - previous_corrected_point[1])

        # Apply correction rules
        if dx in thresholds:
            # Align the current point's x-coordinate to the previous point
            current_point[0] = previous_corrected_point[0]

        if dy in thresholds:
            # Align the current point's y-coordinate to the previous point
            current_point[1] = previous_corrected_point[1]

        # Add the corrected point to the new path
        corrected_path.append(current_point)

    return np.array(corrected_path)

def bresenham_line(p1, p2):
    """
    Generate all integer coordinate points between two points using Bresenham's algorithm.

    Args:
        p1 (tuple or list): Start point coordinates (x1, y1).
        p2 (tuple or list): End point coordinates (x2, y2).

    Yields:
        tuple: (x, y) for each integer point on the path.
    """
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])

    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)

    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    err = dx + dy

    while True:
        yield (x1, y1)
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x1 += sx
        if e2 <= dx:
            err += dx
            y1 += sy

def interpolate_path_with_bresenham(path,spacing):
    """
    Takes a list of path points and performs integer interpolation between adjacent points
    using Bresenham's algorithm.

    Args:
        path (list of lists/tuples): Original path, e.g. [[x1,y1], [x2,y2], ...].

    Returns:
        np.array: Dense interpolated path point array.
    """
    if len(path) < 2:
        return np.array(path)
    spacing = max(1, int(spacing))
    dense_path = []
    # Iterate over each path segment (from point i to point i+1)
    for i in range(len(path) - 1):
        start_point = path[i]
        end_point = path[i+1]

        # Use Bresenham's algorithm to generate interpolated points for this segment
        segment_points = list(bresenham_line(start_point, end_point))
        segment_points = segment_points[::spacing]
        if i > 0:
            # From the second segment onward, remove the first point as it duplicates the previous segment's endpoint
            dense_path.extend(segment_points[1:])
        else:
            # Add the first segment directly
            dense_path.extend(segment_points)

    return np.array(dense_path)

def proc_path_1(path_xy_coords,matrix,thres={1}):
    simplified_waypoints_xy = simplify_path_with_collision_check(path=path_xy_coords, epsilon=1, navigable_map=matrix)
    simplified_waypoints_xy = correct_path_jitters(simplified_waypoints_xy,thres)
    simplified_waypoints_yx = [[p[1], p[0]] for p in simplified_waypoints_xy]

    # c. Use a small step size for ultra-high-density interpolation to ensure straight paths
    # step=1.0 is a fairly robust value; adjust as needed
    # densified_float_path = densify_path(simplified_waypoints_yx, step=1.0)
    # densified_float_path = densify(simplified_waypoints_yx, step=2)
    densified_float_path =  interpolate_path_with_bresenham(simplified_waypoints_yx,spacing=2)
    if densified_float_path is not None:
        # d. Round the high-density float path to form a full pixel path
        integer_path = np.rint(np.array(densified_float_path)).astype(np.int32)

        # e. Remove duplicate pixels to get the final clean path for drawing
        final_path_pixels = integer_path
    # final_path_pixels = np.array([ [i[1],i[0]] for i in path_xy_coords]).astype(np.int32)
    # final_path_pixels = np.array([ [i[1],i[0]] for i in simplified_waypoints_xy]).astype(np.int32)
    return final_path_pixels

def proc_path_2(path_xy_coords,matrix,thres={1}):
    simplified_waypoints_xy = simplify_path_with_collision_check(path=path_xy_coords, epsilon=1, navigable_map=matrix)
    simplified_waypoints_xy = correct_path_jitters(simplified_waypoints_xy,thres)
    simplified_waypoints_yx = [[p[1], p[0]] for p in simplified_waypoints_xy]

    # c. Use a small step size for ultra-high-density interpolation to ensure straight paths
    # step=1.0 is a fairly robust value; adjust as needed
    # densified_float_path = densify_path(simplified_waypoints_yx, step=1.0)
    densified_float_path = densify(simplified_waypoints_yx, step=2)
    # densified_float_path =  interpolate_path_with_bresenham(simplified_waypoints_yx,spacing=2)
    if densified_float_path is not None:
        # d. Round the high-density float path to form a full pixel path
        integer_path = np.rint(np.array(densified_float_path)).astype(np.int32)

        # e. Remove duplicate pixels to get the final clean path for drawing
        final_path_pixels = integer_path
    # final_path_pixels = np.array([ [i[1],i[0]] for i in path_xy_coords]).astype(np.int32)
    # final_path_pixels = np.array([ [i[1],i[0]] for i in simplified_waypoints_xy]).astype(np.int32)
    return final_path_pixels

def densify_path_float(waypoints_xy, step=1.0):
    """
    Densify a path composed of (x, y) floating-point waypoints using linear interpolation.
    """
    densified_path = []
    if not waypoints_xy:
        return densified_path

    densified_path.append(list(waypoints_xy[0]))

    for i in range(len(waypoints_xy) - 1):
        p1 = np.array(waypoints_xy[i])
        p2 = np.array(waypoints_xy[i+1])

        distance = np.linalg.norm(p2 - p1)
        if distance < 1e-6: # If points are identical, skip
            continue

        num_points = int(np.ceil(distance / step))
        if num_points <= 1:
            if list(p2) != densified_path[-1]:
                densified_path.append(list(p2))
            continue

        x_coords = np.linspace(p1[0], p2[0], num_points + 1)[1:]
        y_coords = np.linspace(p1[1], p2[1], num_points + 1)[1:]

        segment_points = np.vstack([x_coords, y_coords]).T.tolist()
        densified_path.extend(segment_points)

    return densified_path

# Douglas-Peucker algorithm for path simplification
def douglas_peucker(points, epsilon):
    """
    Args:
        points (np.ndarray): N x 2 array of 2D points.
        epsilon (float): Simplification tolerance, maximum perpendicular distance from point to line.
    Returns:
        np.ndarray: Simplified point array.
    """
    if len(points) < 3:
        return points

    # Find the point farthest from the line connecting start and end points
    dists = np.abs(np.cross(points[1:-1] - points[0], points[-1] - points[0])) / np.linalg.norm(points[-1] - points[0])
    max_dist = 0
    max_idx = 0
    if len(dists) > 0: # dists is non-empty
        max_dist = np.max(dists)
        max_idx = np.argmax(dists) + 1

    if max_dist > epsilon:
        # If the farthest point exceeds the tolerance, recursively simplify
        result1 = douglas_peucker(points[:max_idx+1], epsilon)
        result2 = douglas_peucker(points[max_idx:], epsilon)
        # Remove duplicate middle point
        return np.vstack((result1[:-1], result2))
    else:
        # Otherwise, this segment is considered straight; keep only start and end points
        return np.array([points[0], points[-1]])

def world_to_pixel(world_coords_2d, occ_map):
    world_x, world_y = world_coords_2d[0], world_coords_2d[1]
    col = np.argmin(np.abs(occ_map[0, 1:] - world_x)) + 1
    row = np.argmin(np.abs(occ_map[1:, 0] - world_y)) + 1
    return [col, row]

# def find_endpoint_in_arc(face_center, normal_vector, radius, arc_degrees, max_tries, occ_map, fat_map):
#     face_center_2d, normal_vector_2d = face_center[:2], normal_vector[:2]
#     if np.linalg.norm(normal_vector_2d) < 1e-6: raise ValueError("Normal vector has zero length")
#     normalized_normal = normal_vector_2d / np.linalg.norm(normal_vector_2d)
#     cos_angle_limit = np.cos(np.deg2rad(arc_degrees / 2.0))
#     for i in range(max_tries):
#         r, theta = radius * np.sqrt(np.random.rand()), 2 * np.pi * np.random.rand()
#         random_point_2d = face_center_2d + np.array([r * np.cos(theta), r * np.sin(theta)])
#         vec_center_to_point = random_point_2d - face_center_2d
#         if np.linalg.norm(vec_center_to_point) < 1e-6: continue
#         normalized_vec_to_point = vec_center_to_point / np.linalg.norm(vec_center_to_point)
#         if np.dot(normalized_normal, normalized_vec_to_point) >= cos_angle_limit:
#             pixel_coords = world_to_pixel(random_point_2d, occ_map)
#             if 0 <= pixel_coords[1] < fat_map.shape[0] and 0 <= pixel_coords[0] < fat_map.shape[1] and fat_map[pixel_coords[1], pixel_coords[0]] == 1:
#                 print(f"Attempt {i+1}/{max_tries}: Successfully found path endpoint.")
#                 return pixel_coords
#     raise RuntimeError(f"After {max_tries} attempts, no passable endpoint was found.")

def find_endpoint_in_arc(face_center, normal_vector, min_radius, max_radius, arc_degrees, max_tries, occ_map, fat_map):
    face_center_2d, normal_vector_2d = face_center[:2], normal_vector[:2]
    if np.linalg.norm(normal_vector_2d) < 1e-6: raise ValueError("Normal vector has zero length")
    normalized_normal = normal_vector_2d / np.linalg.norm(normal_vector_2d)
    cos_angle_limit = np.cos(np.deg2rad(arc_degrees / 2.0))

    # [!!! Core modification !!!] Compute squared radius bounds for uniform area sampling
    min_r_sq = min_radius**2
    max_r_sq = max_radius**2

    for i in range(max_tries):
        # [!!! Core modification !!!] Uniformly generate random points within the annular region
        # 1. Uniformly sample between squared radii
        r_sq = min_r_sq + (max_r_sq - min_r_sq) * np.random.rand()
        # 2. Compute the final radius r
        r = np.sqrt(r_sq)
        # 3. Randomly generate angle
        theta = 2 * np.pi * np.random.rand()

        random_point_2d = face_center_2d + np.array([r * np.cos(theta), r * np.sin(theta)])

        vec_center_to_point = random_point_2d - face_center_2d
        if np.linalg.norm(vec_center_to_point) < 1e-6: continue

        normalized_vec_to_point = vec_center_to_point / np.linalg.norm(vec_center_to_point)

        if np.dot(normalized_normal, normalized_vec_to_point) >= cos_angle_limit:
            pixel_coords = world_to_pixel(random_point_2d, occ_map)
            if 0 <= pixel_coords[1] < fat_map.shape[0] and 0 <= pixel_coords[0] < fat_map.shape[1] and fat_map[pixel_coords[1], pixel_coords[0]] == 1:
                print(f"Attempt {i+1}/{max_tries}: Successfully found path endpoint within {min_radius}m to {max_radius}m annular sector.")
                return pixel_coords, random_point_2d

    raise RuntimeError(f"After {max_tries} attempts, no passable endpoint was found within {min_radius}m to {max_radius}m annular sector.")

def capture_final_scene_photo(camera, world, viewpoint_pos, target_pos, save_path):
    """Capture and save a final scene photo at the specified viewpoint, facing the target."""
    print(f"\n--- Capturing final scene photo ---")
    print(f"Camera position: {np.round(viewpoint_pos, 3)}")
    print(f"Target position: {np.round(target_pos, 3)}")
    try:
        rot_matrix = rot3_from_O_to_AB(O, viewpoint_pos, target_pos)
        orientation_quat = Quaternion(matrix=rot_matrix)
        look_at_quat_wxyz = np.array([orientation_quat.w, orientation_quat.x, orientation_quat.y, orientation_quat.z])
        camera.set_world_pose(position=viewpoint_pos, orientation=look_at_quat_wxyz)

        frame_count, max_wait_frames, image_saved = 0, 100, False
        for _ in range(20): world.step(render=True)

        while not image_saved and frame_count <= max_wait_frames:
            world.step(render=True)
            rgba_data = camera.get_rgba()
            if rgba_data is not None and rgba_data.shape[0] > 0:
                if "torch" in str(type(rgba_data)): rgba_data = rgba_data.cpu().numpy()
                if rgba_data.dtype != np.uint8: rgba_data = np.clip(rgba_data * 255.0, 0, 255).astype(np.uint8)

                image = Image.fromarray(rgba_data, "RGBA")
                image.save(save_path)
                # print(f"Final scene photo successfully saved to: {save_path}")
                print(f"==> Capturing goal object at: {save_path}")
                image_saved = True
            frame_count += 1

        if not image_saved: print(f"Warning: Failed to capture final photo, timed out without getting a valid image.")

    except Exception as e:
        print(f"Error: Exception occurred while capturing final photo: {e}")
        import traceback
        traceback.print_exc()

def get_final_camera_orientation(viewpoint_pos, target_pos, world_up_vec):
    """
    Compute the final roll-free camera orientation quaternion.

    Uses a two-step approach:
    1. Use rot3_from_O_to_AB to compute the pure horizontal rotation (Yaw).
    2. Compute and apply a pure pitch rotation (look up/down).
    """
    # Step 1: Compute pure horizontal rotation (Yaw)
    horizontal_target = np.array([target_pos[0], target_pos[1], viewpoint_pos[2]])
    horizontal_rot_matrix = rot3_from_O_to_AB(O, viewpoint_pos, horizontal_target)
    horizontal_quat = Quaternion(matrix=horizontal_rot_matrix)

    # Step 2: Compute and apply pure pitch rotation (Pitch)
    vec_to_target = target_pos - viewpoint_pos
    horizontal_dist = np.linalg.norm([vec_to_target[0], vec_to_target[1]])
    vertical_dist = vec_to_target[2]

    pitch_angle_rad = -np.arctan2(vertical_dist, horizontal_dist)

    # Pitch axis is the camera's "left" direction (local Y-axis) after completing horizontal rotation
    # We obtain its world direction by rotating the default "left" vector [0,1,0]
    pitch_axis = horizontal_rot_matrix @ np.array([0, 1, 0])

    pitch_quat = Quaternion(axis=pitch_axis, angle=pitch_angle_rad)

    # Compose rotations: apply pitch after yaw (horizontal rotation)
    final_orientation = pitch_quat * horizontal_quat

    return final_orientation

def capture_final_scene_photo_twostep(camera, world, viewpoint_pos, target_pos, world_up_vec, save_path):
    """
    [Final version] Capture and save a final scene photo at the specified viewpoint, facing the target.
    """
    print(f"\n--- Capturing final scene photo (final corrected version) ---")
    print(f"Camera position: {np.round(viewpoint_pos, 3)}")
    print(f"Target position: {np.round(target_pos, 3)}")

    try:
        # 1. Get the final correct orientation in one line
        final_orientation = get_final_camera_orientation(viewpoint_pos, target_pos, world_up_vec)

        # 2. Set camera pose and capture (Isaac Sim requires wxyz format)
        orientation_wxyz = np.array([final_orientation.w, final_orientation.x, final_orientation.y, final_orientation.z])
        camera.set_world_pose(position=viewpoint_pos, orientation=orientation_wxyz)

        # Subsequent capture and save logic remains unchanged
        for _ in range(20): world.step(render=True)

        rgba_data = camera.get_rgba()
        if rgba_data is not None and rgba_data.shape[0] > 0:
            if "torch" in str(type(rgba_data)): rgba_data = rgba_data.cpu().numpy()
            image_data = np.clip(rgba_data, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_data, "RGBA")
            image.save(save_path)
            print(f"Final scene photo successfully saved to: {save_path}")
        else:
            print(f"Warning: Failed to capture final photo, no valid image obtained.")

    except Exception as e:
        print(f"Error: Exception occurred while capturing final photo: {e}")
        import traceback
        traceback.print_exc()

def draw_semitransparent_fan(image, center_px, radius_px, normal_vec, arc_degrees, color, alpha):
    overlay = image.copy()
    center = tuple(np.rint(center_px).astype(int))
    axes = (int(round(radius_px)), int(round(radius_px)))
    angle = np.rad2deg(np.arctan2(normal_vec[1], normal_vec[0]))
    cv2.ellipse(overlay, center, axes, angle, -arc_degrees / 2, arc_degrees / 2, color, -1)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)



def proc_path_3(path_xy_coords,matrix):
    from scipy.ndimage import gaussian_filter1d # Gaussian smoothing
    from scipy.signal import savgol_filter # Savitzky-Golay filter
    simplified_waypoints_xy = simplify_path_with_collision_check(
        path=path_xy_coords,
        epsilon=20,
        navigable_map=matrix
    )
    ### original
    lsh_path = [ [i[1],i[0]] for i in simplified_waypoints_xy]
    lsh_path = densify(lsh_path,step=3)
    lsh_path = np.array(lsh_path)
    lsh_path = np.rint(lsh_path).astype(np.int32)


    # Generate high-precision float path (initial densification)
    initial_densified_path = densify_path_float(simplified_waypoints_xy, step=5.0)
    print(f"Path points after initial densification: {len(initial_densified_path)}")


    # Savitzky-Golay global smoothing
    if len(initial_densified_path) > 10:
        path_for_savgol = np.array(initial_densified_path)

        savgol_window_length = min(21, len(path_for_savgol) // 2 * 2 + 1)
        savgol_polyorder = min(3, savgol_window_length - 1)

        if savgol_window_length >= 3 and savgol_polyorder >= 1:
            try:
                smoothed_x = savgol_filter(path_for_savgol[:, 0], window_length=savgol_window_length, polyorder=savgol_polyorder)
                smoothed_y = savgol_filter(path_for_savgol[:, 1], window_length=savgol_window_length, polyorder=savgol_polyorder)
                smoothed_initial_path = np.vstack((smoothed_x, smoothed_y)).T.tolist()
                initial_densified_path = smoothed_initial_path
                print(f"Path has been Savitzky-Golay globally smoothed. window_length={savgol_window_length}, polyorder={savgol_polyorder}")
            except ValueError as e:
                print(f"Warning: Savitzky-Golay smoothing failed (path may be too short or parameters inappropriate): {e}. Skipping smoothing.")


    # Douglas-Peucker algorithm for path simplification
    # The goal is to find an epsilon that eliminates jitter without over-simplifying (causing wall clipping).
    douglas_peucker_epsilon = 1.0 # This value can be adjusted, e.g. from 0.5 to 5.0. Smaller values keep more points; larger values simplify more, making the path straighter.


    path_for_douglas_peucker = np.array(initial_densified_path)
    if path_for_douglas_peucker.shape[0] < 3: # Too few points can cause bugs; simulator may close directly
        final_simplified_key_points_path = initial_densified_path
    else:
        simplified_path_dp = douglas_peucker(path_for_douglas_peucker, douglas_peucker_epsilon)
        final_simplified_key_points_path = simplified_path_dp.tolist()

    print(f"Path points after Douglas-Peucker simplification: {len(final_simplified_key_points_path)}")


    # Densification, targeting around 50 points
    TARGET_DENSIFY_STEP = 18.0 # Higher values result in fewer points; currently targeting around 50
    final_densified_path_xy = densify_path_float(final_simplified_key_points_path, step=TARGET_DENSIFY_STEP)

    print(f"Path points after final densification: {len(final_densified_path_xy)}")

    # Secondary smoothing
    if len(final_densified_path_xy) > 5:
        path_for_final_gaussian = np.array(final_densified_path_xy)
        smoothed_x = gaussian_filter1d(path_for_final_gaussian[:, 0], sigma=1.0)
        smoothed_y = gaussian_filter1d(path_for_final_gaussian[:, 1], sigma=1.0)
        high_precision_path_xy = np.vstack((smoothed_x, smoothed_y)).T.tolist()
        print("Path has been final Gaussian smoothed.")
    else:
        high_precision_path_xy = final_densified_path_xy

    path_array = np.array(high_precision_path_xy)
    # return np.array([ [i[1],i[0]] for i in path_array.astype(int)])
    initial_densified_path_xy = np.array([ [i[1],i[0]] for i in np.array(initial_densified_path).astype(int)])

    initial_densified_path_xy = correct_path_jitters(initial_densified_path_xy,[1])
    return initial_densified_path_xy



# --- NEW: Bresenham line for collision checking ---
def bresenham_line_pixels(p1_int, p2_int):
    """
    Generates all integer pixel coordinates on a line segment using Bresenham's algorithm.
    Args:
        p1_int (tuple): Start pixel (x1, y1).
        p2_int (tuple): End pixel (x2, y2).
    Yields:
        tuple: (x, y) for each pixel on the line.
    """
    x1, y1 = p1_int
    x2, y2 = p2_int

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        yield (x1, y1)
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

# --- NEW: Line collision checker using Bresenham with radius ---
def check_line_collision(p1_float, p2_float, navigable_map_binary, check_radius_pixels=1):
    """
    Checks if a line segment between two float points collides with obstacles in a binary map.
    This version checks a small surrounding area around each pixel on the line.

    Args:
        p1_float (list/tuple): Start point (x, y) float.
        p2_float (list/tuple): End point (x, y) float.
        navigable_map_binary (np.ndarray): Binary map (0=obstacle, 1=walkable).
        check_radius_pixels (int): Radius (in pixels) around each line pixel to check.
                                   e.g., 1 for a 3x3 check, 2 for a 5x5 check.
    Returns:
        bool: True if collision detected, False otherwise.
    """
    p1_int = (int(round(p1_float[0])), int(round(p1_float[1])))
    p2_int = (int(round(p2_float[0])), int(round(p2_float[1])))

    map_height, map_width = navigable_map_binary.shape

    # Iterate through all integer pixels on the line segment
    for x_line, y_line in bresenham_line_pixels(p1_int, p2_int):
        # Check a small square area around the current line pixel
        for dy_offset in range(-check_radius_pixels, check_radius_pixels + 1):
            for dx_offset in range(-check_radius_pixels, check_radius_pixels + 1):
                x_check = x_line + dx_offset
                y_check = y_line + dy_offset

                # Check if the current check_pixel is within map bounds
                if not (0 <= y_check < map_height and 0 <= x_check < map_width):
                    return True # Path goes out of map bounds or into an unmapped area

                # Check if the current check_pixel is an obstacle (0)
                if navigable_map_binary[y_check, x_check] == 0:
                    return True # Collision detected (0 is obstacle)

    return False # No collision found along the line segment or its surrounding area

def collision_aware_smooth_path(path_list_float, navigable_map_for_check, smoothing_strength_initial=3.0, max_smoothing_attempts=20, collision_check_radius_pixels=1):
    """
    Perform collision-aware path smoothing. It iteratively tries to smooth the path,
    checking for collisions after each smoothing. If a collision occurs, the smoothing
    strength is reduced and it retries.

    Args:
        path_list_float (list): List of float [x,y] points (pixel coordinates).
        navigable_map_for_check (np.ndarray): Binary map for collision detection (0=obstacle, 1=walkable).
                                              Usually matrix_thin (thin map) or matrix_fat.
        smoothing_strength_initial (float): Initial sigma value for Gaussian smoothing.
                                            Controls the aggressiveness of initial smoothing.
        max_smoothing_attempts (int): Maximum number of smoothing attempts to avoid infinite loops.
        collision_check_radius_pixels (int): Extra pixel radius for collision detection.

    Returns:
        list: Smoothed and collision-checked path. Returns the original path if no collision-free path is found.
    """
    if len(path_list_float) < 2:
        return path_list_float

    current_path_np = np.array(path_list_float)
    best_safe_path_np = current_path_np.copy() # Store the last known safe path

    current_strength = float(smoothing_strength_initial)

    for attempt in range(max_smoothing_attempts):
        if current_strength < 0.1 and attempt > 0: # Minimum effective strength
            print(f"   - Warning: Smoothing strength too low ({current_strength:.2f}), stopping attempts.")
            break

        # Apply Gaussian smoothing
        try_path_x = gaussian_filter1d(current_path_np[:, 0], sigma=current_strength)
        try_path_y = gaussian_filter1d(current_path_np[:, 1], sigma=current_strength)
        try_path_float_list = np.vstack((try_path_x, try_path_y)).T.tolist()

        # Check if this smoothed path has collisions
        collides = False
        if len(try_path_float_list) > 1:
            for i in range(len(try_path_float_list) - 1):
                p1 = try_path_float_list[i]
                p2 = try_path_float_list[i+1]
                # Pass the new collision_check_radius_pixels
                if check_line_collision(p1, p2, navigable_map_for_check, collision_check_radius_pixels):
                    collides = True
                    break
        else: # Path too short after smoothing
            collides = True

        if not collides:
            # If no collision, this strength is safe. Update best_safe_path.
            best_safe_path_np = np.array(try_path_float_list)

            # Found a safe path, return without trying stronger smoothing
            print(f"   - Attempt {attempt+1}: Strength {current_strength:.2f} found collision-free path.")
            return best_safe_path_np.tolist()
        else:
            # If collision occurred, significantly reduce smoothing strength for next attempt
            print(f"   - Attempt {attempt+1}: Strength {current_strength:.2f} collision detected. Reducing strength.")
            current_strength *= 0.6 # Reduce strength by 40%

    # If after all attempts no collision-free smoothed path was found, return the most recent safe path
    print("Warning: Collision-aware smoothing failed to find a collision-free smoothed path. Returning most recent safe path.")
    return best_safe_path_np.tolist()

def proc_path_zerui(path_xy_coords_float,matrix_fat, matrix_thin):
    # Step 1: Initial simplification (using grutopia_extension's simplify_path_with_collision_check)
    # This roughly simplifies the path based on matrix_thin and epsilon
    try:
        simplified_waypoints_xy = simplify_path_with_collision_check(
            path=path_xy_coords_float,
            epsilon=20, # **Tunable parameter**: Initial simplification tolerance. Can be larger if the A* path is rough.
            navigable_map=matrix_thin # Use thin map for collision checking
        )
    except:
        simplified_waypoints_xy = path_xy_coords_float

    # Step 2: Initial densification
    initial_densified_path = densify_path_float(simplified_waypoints_xy, step=1.0)
    print(f"Path points after initial densification: {len(initial_densified_path)}")


    # --- Step 3: Collision-aware smoothing (core smoothing stage) ---
    print("Starting collision-aware smoothing...")
    final_path_after_cas = collision_aware_smooth_path(
        initial_densified_path,
        navigable_map_for_check=matrix_fat, # **Use fat map for collision detection here** to ensure safety margin
        smoothing_strength_initial=1.0, # **Tunable parameter**: Initial smoothing strength. Reduced to 1.0 or even 0.5.
                                    #  If still colliding, try smaller values.
        max_smoothing_attempts=50, # **Tunable parameter**: Maximum number of attempts. Increased to 50-100.
        collision_check_radius_pixels=1 # **Tunable parameter**: Extra pixel radius for collision detection. Usually 1 or 2.
    )
    print(f"Path points after collision-aware smoothing: {len(final_path_after_cas)}")


    # Step 4: Douglas-Peucker path simplification (after collision-aware smoothing, further control point count and remove minor redundancies)
    # This function now operates on a path that has already been initially smoothed and collision-checked
    douglas_peucker_epsilon = 0.5 # **Tunable parameter**: Douglas-Peucker tolerance.
                                # Goal is to balance smoothness with point count.
                                # Empirical values: 0.5 - 2.0, depending on desired straightness and point count.

    path_for_douglas_peucker = np.array(final_path_after_cas) # Input is the CAS function's result
    if path_for_douglas_peucker.shape[0] < 3:
        final_simplified_key_points_path = final_path_after_cas
    else:
        simplified_path_dp_np = douglas_peucker(path_for_douglas_peucker, douglas_peucker_epsilon)
        if simplified_path_dp_np.ndim == 1: # If only 1 point, reshape to (1,2)
            final_simplified_key_points_path = [tuple(simplified_path_dp_np)]
        else:
            final_simplified_key_points_path = [tuple(p) for p in simplified_path_dp_np]


    print(f"Path points after Douglas-Peucker simplification: {len(final_simplified_key_points_path)}")


    # Step 5: Forced final densification
    # TARGET_DENSIFY_STEP = 18.0 # Target around 50 final points # 10 for kujiale
    TARGET_DENSIFY_STEP = 10.0 # Target around 50 final points # 10 for kujiale
    final_densified_path_xy = densify_path_float(final_simplified_key_points_path, step=TARGET_DENSIFY_STEP)

    print(f"Path points after final densification: {len(final_densified_path_xy)}")

    # Step 6: Lightweight secondary Gaussian smoothing (remove interpolation zigzags, maintain smoothness)
    # This step is now optional, used to remove interpolation zigzags from forced densification
    if len(final_densified_path_xy) > 5 and False:
        path_for_final_gaussian = np.array(final_densified_path_xy)
        smoothed_x = gaussian_filter1d(path_for_final_gaussian[:, 0], sigma=0.5) # **Tunable sigma**: 0.5 - 1.0
        smoothed_y = gaussian_filter1d(path_for_final_gaussian[:, 1], sigma=0.5)
        high_precision_path_xy = [(float(x), float(y)) for x,y in np.vstack((smoothed_x, smoothed_y)).T]
        # print("Path has been final Gaussian smoothed.")
    else:
        high_precision_path_xy = final_densified_path_xy

    # --- Use the final path ---
    path_array = np.array(high_precision_path_xy)

    path_array = np.rint(path_array).astype(np.int32)
    final_path_pixels = [ [p[1],p[0]] for p in path_array]
    # final_path_pixels =  interpolate_path_with_bresenham(final_path_pixels,spacing=3) #kujiale
    final_path_pixels =  interpolate_path_with_bresenham(final_path_pixels,spacing=5)
    print(f"Path points after final interpolation: {len(final_path_pixels)}")
    return final_path_pixels


def segments_intersect(p1, q1, p2, q2):
    """
    Determine whether line segments p1q1 and p2q2 intersect.
    p1, q1 are the endpoints of the first segment.
    p2, q2 are the endpoints of the second segment.
    """
    def orientation(p, q, r):
        """
        Determine the orientation of the triplet (p, q, r).
        Returns:
        0 --> p, q, r are collinear
        1 --> Clockwise
        2 --> Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - \
              (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or counterclockwise

    def on_segment(p, q, r):
        """
        Determine whether point q lies on segment pr (given that the three points are collinear).
        """
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    # Compute four orientations
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # --- General case ---
    # If (p1,q1,p2) and (p1,q1,q2) have different orientations,
    # and (p2,q2,p1) and (p2,q2,q1) also have different orientations, they must intersect.
    if o1 != o2 and o3 != o4:
        return True

    # --- Special case: point lies on the other segment (collinear case) ---
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False # All other cases: no intersection

# --- Main function: determine whether a path intersects a line segment ---
def path_intersects_segment(path, line_segment):
    """
    Determine whether a path intersects a given line segment.

    Args:
        path (list of lists/tuples): Path point list [[x1,y1], ...].
        line_segment (list of lists/tuples): Two endpoints of the line segment [[u1,v1], [u2,v2]].

    Returns:
        bool: True if they intersect, False otherwise.
    """
    line_p1, line_p2 = line_segment

    # Iterate over each sub-segment of the path
    for i in range(len(path) - 1):
        path_p1 = path[i]
        path_p2 = path[i+1]

        # Check if the path sub-segment intersects the given line segment
        if segments_intersect(path_p1, path_p2, line_p1, line_p2):
            print(f"Intersection detected: path segment {path_p1}-{path_p2} intersects the line.")
            return True

    return False

def simplify_path_by_intersection(path, line_segment):
    """
    If the path intersects the line segment, return a new path consisting of the path start point,
    the line center point, and the path end point.
    Otherwise, return the original path.

    Args:
        path (list of lists/tuples): Path point list [[x1,y1], ...].
        line_segment (list of lists/tuples): Two endpoints of the line segment [[u1,v1], [u2,v2]].

    Returns:
        np.array: Simplified new path if intersection exists, otherwise the original path.
    """
    line_p1, line_p2 = np.array(line_segment[0]), np.array(line_segment[1])

    # Iterate over each sub-segment of the path
    for i in range(len(path) - 1):
        path_p1 = path[i]
        path_p2 = path[i+1]

        # Check if the path sub-segment intersects the given line segment
        if segments_intersect(path_p1, path_p2, line_p1, line_p2):
            print(f"Intersection detected: path segment {path_p1}-{path_p2} intersects the line.")

            # Compute the center point Q of the line
            center_point_q = (line_p1 + line_p2) / 2.0

            # Get the path start point A and end point E
            start_point_A = path[0]
            end_point_E = path[-1]

            # Build new path [A, Q, E]
            new_path = [start_point_A, center_point_q.tolist(), end_point_E]

            return np.array(new_path)

    # If the loop ends without finding any intersection
    print("No intersection detected, path remains unchanged.")
    return np.array(path)

def segment_path_by_all_intersections(path, line_segments):
    """
    Find all intersections between a path and a set of line segments, and reconstruct
    the path based on the order of intersections along the path.

    Args:
        path (list of lists/tuples): Path point list [[x1,y1], ...].
        line_segments (list of lists): A list containing multiple line segments.

    Returns:
        np.array: Segmented new path if intersections exist, otherwise the original path.
    """
    intersections = []

    # 1. Find all intersections and record which path segment they occur on
    for line_index, line_segment in enumerate(line_segments):
        line_p1, line_p2 = np.array(line_segment[0]), np.array(line_segment[1])

        for path_index in range(len(path) - 1):
            path_p1 = path[path_index]
            path_p2 = path[path_index+1]

            if segments_intersect(path_p1, path_p2, line_p1, line_p2):
                center_point_q = (line_p1 + line_p2) / 2.0
                # Record (path segment index, line index, line center point)
                intersections.append((path_index, line_index, center_point_q.tolist()))
                # print(f"Intersection detected: path segment {path_index} intersects line {line_index+1}.")

    # 2. If no intersections found, return the original path
    if not intersections:
        print("No intersections detected, path remains unchanged.")
        return False, np.array(path)

    # 3. Sort intersections by their order of appearance along the path
    #    Primary sort by path segment index, secondary by line index (if multiple lines intersect the same segment)
    intersections.sort()

    # 4. Build new path
    start_point_A = path[0]
    end_point_E = path[-1]

    # Extract all sorted intersection points Q
    q_points = [q for _, _, q in intersections]

    new_path = [start_point_A] + q_points + [end_point_E]

    return True, np.array(new_path)

# def get_front_face_info(min_points, max_points, object_center):
#     '''
#     Consider the biggest 2D face
#     '''
#     min_p, max_p, obj_center = np.array(min_points), np.array(max_points), np.array(object_center)
#     dimensions = max_p - min_p
#     areas = {"XZ": dimensions[0]*dimensions[2], "XY": dimensions[0]*dimensions[1], "YZ": dimensions[1]*dimensions[2]}
#     largest_face_axes = max(areas, key=areas.get)
#     box_center = (min_p + max_p) / 2
#     face_centers, normal_vectors = [], []
#     if largest_face_axes == "XZ":
#         face_centers.extend([np.array([box_center[0], min_p[1], box_center[2]]), np.array([box_center[0], max_p[1], box_center[2]])])
#         normal_vectors.extend([np.array([0, -1, 0]), np.array([0, 1, 0])])
#     elif largest_face_axes == "YZ":
#         face_centers.extend([np.array([min_p[0], box_center[1], box_center[2]]), np.array([max_p[0], box_center[1], box_center[2]])])
#         normal_vectors.extend([np.array([-1, 0, 0]), np.array([1, 0, 0])])
#     else: # XY
#         face_centers.extend([np.array([box_center[0], box_center[1], min_p[2]]), np.array([box_center[0], box_center[1], max_p[2]])])
#         normal_vectors.extend([np.array([0, 0, -1]), np.array([0, 0, 1])])
#     vec_face0_to_center = obj_center - face_centers[0]
#     front_face_center, front_normal = (face_centers[0], normal_vectors[0]) if np.dot(vec_face0_to_center, normal_vectors[0]) < 0 else (face_centers[1], normal_vectors[1])
#     return front_face_center, front_normal

def get_front_face_info(min_points, max_points, object_center):
    """
    Analyze a bounding box, find the object's "front face" (ignoring top and bottom faces),
    and return its center and normal.
    """
    min_p, max_p, obj_center = np.array(min_points), np.array(max_points), np.array(object_center)
    dimensions = max_p - min_p
    # Only search for the largest face among side faces (YZ and XZ planes), ignoring top/bottom (XY plane).
    side_areas = {
        "YZ": dimensions[1] * dimensions[2], # YZ plane area
        "XZ": dimensions[0] * dimensions[2]  # XZ plane area
    }

    # If the object's height or some side dimension is too small, fall back to original logic to avoid errors
    if not side_areas or max(side_areas.values()) < 1e-6:
         print("Warning: Object's side face area is too small, falling back to original largest face search logic.")
         areas = { "XZ": dimensions[0]*dimensions[2], "XY": dimensions[0]*dimensions[1], "YZ": dimensions[1]*dimensions[2] }
         largest_face_axes = max(areas, key=areas.get)
    else:
        largest_face_axes = max(side_areas, key=side_areas.get)

    print(f"Found largest side face for navigation parallel to: {largest_face_axes} plane")

    box_center = (min_p + max_p) / 2
    face_centers, normal_vectors = [], []

    if largest_face_axes == "XZ":
        face_centers.extend([np.array([box_center[0], min_p[1], box_center[2]]), np.array([box_center[0], max_p[1], box_center[2]])])
        normal_vectors.extend([np.array([0, -1, 0]), np.array([0, 1, 0])])
    elif largest_face_axes == "YZ":
        face_centers.extend([np.array([min_p[0], box_center[1], box_center[2]]), np.array([max_p[0], box_center[1], box_center[2]])])
        normal_vectors.extend([np.array([-1, 0, 0]), np.array([1, 0, 0])])
    else: # XY (only possible in fallback logic)
        face_centers.extend([np.array([box_center[0], box_center[1], min_p[2]]), np.array([box_center[0], box_center[1], max_p[2]])])
        normal_vectors.extend([np.array([0, 0, -1]), np.array([0, 0, 1])])

    vec_face0_to_center = obj_center - face_centers[0]
    front_face_center, front_normal = (face_centers[0], normal_vectors[0]) if np.dot(vec_face0_to_center, normal_vectors[0]) < 0 else (face_centers[1], normal_vectors[1])

    return front_face_center, front_normal

def get_opposing_faces_info(min_points, max_points):
    """
    Analyze a bounding box, find the two largest opposing side faces,
    and return their respective center points and normals.
    Ignores top and bottom faces.
    """
    min_p, max_p = np.array(min_points), np.array(max_points)

    dimensions = max_p - min_p

    side_areas = {
        "YZ": dimensions[1] * dimensions[2],
        "XZ": dimensions[0] * dimensions[2]
    }

    if not side_areas or max(side_areas.values()) < 1e-6:
        # If side face area is too small, provide a default fallback (unlikely to trigger)
        largest_face_axes = "YZ"
    else:
        largest_face_axes = max(side_areas, key=side_areas.get)

    box_center = (min_p + max_p) / 2

    if largest_face_axes == "XZ":
        # Face parallel to XZ plane, normal along Y-axis
        face1_center = np.array([box_center[0], min_p[1], box_center[2]])
        face1_normal = np.array([0, -1, 0])
        face2_center = np.array([box_center[0], max_p[1], box_center[2]])
        face2_normal = np.array([0, 1, 0])
    else: # YZ
        # Face parallel to YZ plane, normal along X-axis
        face1_center = np.array([min_p[0], box_center[1], box_center[2]])
        face1_normal = np.array([-1, 0, 0])
        face2_center = np.array([max_p[0], box_center[1], box_center[2]])
        face2_normal = np.array([1, 0, 0])

    return (
        {"center": face1_center, "normal": face1_normal},
        {"center": face2_center, "normal": face2_normal}
    )



def get_opposing_faces_info_yxz(min_points, max_points):
    """
    Analyze a bounding box, find the two largest opposing side faces,
    and return their respective center points and normals.
    Ignores top and bottom faces.
    """
    # min_p, max_p = np.array(min_points), np.array(max_points)
    # kujiale, yxz -> xyz
    min_points_xyz = np.array(min_points)[[1, 0, 2]]
    max_points_xyz = np.array(max_points)[[1, 0, 2]]
    min_p, max_p = min_points_xyz, max_points_xyz # Use the converted points
    dimensions = max_p - min_p

    side_areas = {
        "YZ": dimensions[1] * dimensions[2],
        "XZ": dimensions[0] * dimensions[2]
    }

    if not side_areas or max(side_areas.values()) < 1e-6:
        # If side face area is too small, provide a default fallback (unlikely to trigger)
        largest_face_axes = "YZ"
    else:
        largest_face_axes = max(side_areas, key=side_areas.get)

    box_center = (min_p + max_p) / 2

    if largest_face_axes == "XZ":
        # Face parallel to XZ plane, normal along Y-axis
        face1_center = np.array([box_center[0], min_p[1], box_center[2]])
        face1_normal = np.array([0, -1, 0])
        face2_center = np.array([box_center[0], max_p[1], box_center[2]])
        face2_normal = np.array([0, 1, 0])
    else: # YZ
        # Face parallel to YZ plane, normal along X-axis
        face1_center = np.array([min_p[0], box_center[1], box_center[2]])
        face1_normal = np.array([-1, 0, 0])
        face2_center = np.array([max_p[0], box_center[1], box_center[2]])
        face2_normal = np.array([1, 0, 0])

    # return (
    #     {"center": face1_center, "normal": face1_normal},
    #     {"center": face2_center, "normal": face2_normal}
    # )
    face1_yxz = {
        "center": face1_center[[1, 0, 2]],
        "normal": face1_normal[[1, 0, 2]]
    }
    face2_yxz = {
        "center": face2_center[[1, 0, 2]],
        "normal": face2_normal[[1, 0, 2]]
    }

    return (face1_yxz, face2_yxz)

def calculate_proximity_risk_score(face_info, radius, arc_degrees, occ_map, distance_transform_map):
    """
    Calculate the "proximity risk score" for a given fan-shaped region.
    The score is the sum of the inverse squared distances from each point to the nearest obstacle.
    A low score means the direction is very safe with almost no nearby obstacles.
    """
    map_height, map_width = distance_transform_map.shape
    mask = np.zeros((map_height, map_width), dtype=np.uint8)

    # pixel_size = occ_map[0, 2] - occ_map[0, 1]
    pixel_size = np.abs(occ_map[0, 2] - occ_map[0, 1]) # robust?
    radius_px = radius / pixel_size

    center_px = world_to_pixel(face_info["center"][:2], occ_map)
    center_px_cv = (int(center_px[0]), int(center_px[1]))

    normal_vec = face_info["normal"][:2]
    angle = np.rad2deg(np.arctan2(-normal_vec[1], normal_vec[0]))

    cv2.ellipse(mask, center_px_cv, (int(round(radius_px)), int(round(radius_px))),
                int(round(angle)), -arc_degrees / 2, arc_degrees / 2, 1, -1)

    fan_locations = (mask == 1)

    if not np.any(fan_locations):
        # If the fan is entirely outside the map, assign an extremely high risk value
        return float('inf')

    fan_area_on_dist_map = distance_transform_map[fan_locations]

    # To avoid division by zero (when a point is on an obstacle, distance d=0), add a small epsilon
    epsilon = 1e-6

    # Compute each point's risk value: inverse of squared distance. Closer distance means higher risk.
    risk_values = 1.0 / (fan_area_on_dist_map**2 + epsilon)

    # Return the sum of all risk values as the final risk score for this direction
    return np.sum(risk_values)

def find_containing_room(point, polygons):
    """
    Determine which polygon a point falls inside.

    Args:
        point (list or tuple): Coordinates of the point to check [px, py].
        polygons (list of lists): A list containing multiple polygons,
                                 each polygon is a list of vertices.

    Returns:
        int: Index of the first polygon containing the point. Returns -1 if not found.
    """
    # Iterate over each polygon and its index
    for i, polygon_vertices in enumerate(polygons):
        # Create a Path object
        path = Path(polygon_vertices)

        # Use the contains_point method to check
        if path.contains_point(point):
            # If the point is inside the polygon, return its index immediately
            return i

    # If no containing polygon found after iterating all, return -1
    return -1
