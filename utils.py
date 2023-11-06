#-*-coding:utf-8-*-
"""
    Utility functions
    @author: Qianyue He
    @date: 2022-12-13
"""

import scipy.io
import numpy as np
from scipy.spatial.transform import Rotation as Rot

__all__ = ['skew_symmetric_transform', 'world_frame_ray_dir', 'get_arrow', 'np_rotation_between']

# ================================ Mathematical utilities ====================================
def skew_symmetric_transform(t: np.ndarray):
    """ Skew Symmetric Transform for Rodrigues' rotation formula """
    x, y, z = t
    return np.float32([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

def world_frame_ray_dir(K_inv: np.ndarray, pix: np.ndarray, Rs: np.ndarray):
    """
        Inverse projection via intrinsic matrix
        - Transform pixel coordinates (non-homogeneous) to world frame ray direction    
        - Logic should be checked (matplotlib plot3d)
        - pix of shape (N, 2, 1) N = number of covisible frames
    """
    num_covi = pix.shape[0]
    homo_pix = np.concatenate((pix, np.ones((num_covi, 1, 1))), axis = 1)      # shape (N, 3, 1)
    camera_coords = K_inv @ homo_pix                                            # K_inv for inverse projection
    camera_coords = camera_coords / np.linalg.norm(camera_coords, axis = 1)[:, None, :]     # normalize along axis 0
    return Rs @ camera_coords                # normalized ray direction in the world frame

def np_rotation_between(fixed: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
        Transform parsed from xml file is merely camera orientation (numpy CPU version)
        INPUT arrays [MUST] be normalized
        Orientation should be transformed to be camera rotation matrix
        Rotation from <fixed> vector to <target> vector, defined by cross product and angle-axis
    """
    axis = np.cross(fixed, target)
    dot  = (fixed * target).sum(axis = -1, keepdims = True)
    ill_mask = abs(dot) > 1. - 1e-5
    axis /= np.linalg.norm(axis, axis = -1, keepdims = True)
    axis *= np.arccos(dot)
    euler_vec = Rot.from_rotvec(axis).as_euler('zxy')
    euler_vec[:, 0] = 0                                                # eliminate roll angle
    results = Rot.from_euler('zxy', euler_vec).as_matrix()
    return results

# ============================ open3d utils ==============================
# Draw arrow from https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def get_arrow(origin, end, length, scale=1, color = [1, 0, 0]):
    import open3d as o3d
    assert(not np.all(end == origin))
    vec = (end - origin) * length
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
        cone_height=size*0.2 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*(1 - 0.2*scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    mesh.paint_uniform_color(color)
    return (mesh)

def load_from_mat():
    pass

if __name__ == "__main__":
    print("This script is not to be executed.")