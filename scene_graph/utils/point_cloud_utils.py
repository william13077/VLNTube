import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy


def is_point_in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, scipy.spatial.Delaunay):
        hull = scipy.spatial.Delaunay(hull)

    return hull.find_simplex(p)>=0

def is_inside(src_pts, target_pts, thresh=0.5):
    try:
        hull = scipy.spatial.ConvexHull(target_pts)
    except:
        return False
    # print("vertices of hull: ", np.array(hull.vertices).shape)
    hull_vertices = np.array([[0,0,0]])
    for v in hull.vertices:
        try:
            hull_vertices = np.vstack((hull_vertices, np.array([target_pts[v,0], target_pts[v,1], target_pts[v,2]])))
        except:
            import pdb;pdb.set_trace()
    hull_vertices = hull_vertices[1:]

    num_src_pts = len(src_pts)
    # Don't want threshold to be too large (specially with more objects, like 4, 0.9*thresh becomes too large)
    thresh_obj_particles = thresh * num_src_pts
    src_points_in_hull = is_point_in_hull(src_pts, hull_vertices)
    # print("src pts in target, thresh: ", src_points_in_hull.sum(), thresh_obj_particles)
    if src_points_in_hull.sum() > thresh_obj_particles:
        return True
    else:
        return False
    
def calculate_distance_between_two_point_clouds(point_cloud_a, point_cloud_b):
    nn = NearestNeighbors(n_neighbors=1).fit(point_cloud_a)
    distances, _ = nn.kneighbors(point_cloud_b)
    res = np.min(distances)
    
    return res