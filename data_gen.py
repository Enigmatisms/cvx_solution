#-*-coding:utf-8-*-
"""
    Generate data to be of used
    @author: Qianyue He
    @date: 2023-11-6
"""

import json
import numpy as np
import open3d as o3d
import scipy.io as sio
import open3d.visualization.gui as gui

from scipy.stats import qmc
from utils import get_arrow, np_rotation_between

RANDOM_SEED = 114515        # such a smelly number!

K = np.float32([
    [1500, 0, 400], 
    [0, 1500, 400],
    [0,   0,   1],
])

# 8 / 16 / 32
def quasi_uniform_eighth_spherical_sampling(sample_base = 4):
    """ Quasi Monte Carlo sampling for generating points on a eighth-sphere """
    sampler = qmc.Sobol(d=2, scramble=True, seed=RANDOM_SEED)
    sample = sampler.random_base2(m = sample_base)
    cos_theta = sample[:, 0]
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    phi = 0.5 * np.pi * sample[:, 1] 
    spherical_samples = np.stack([
        np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, cos_theta
    ], axis = -1)
    return spherical_samples

def radial_perturb(samples: np.ndarray, scale = 1.5):
    """ Add perturbation in radial direction """
    np.random.seed(RANDOM_SEED)
    distance_samples = (np.random.rayleigh(scale, samples.shape[0]) - 0.5).clip(-0.5, 1.5)
    samples + samples * distance_samples[:, None]
    return (samples + samples * distance_samples[:, None]) * 2

def visualize_spherical_samples(samples):
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - Spherical sample visualize", 1024, 768)
    vis.set_background((0.3, 0.3, 0.3, 1), None)
    vis.show_settings = True
    vis.show_skybox(False)
    vis.show_axes = True

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(samples)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.float32([0, 0, 1]), (samples.shape[0], 1)))
    vis.add_geometry('sample pos', pcd)
    for i, v in enumerate(samples):
        arrow = get_arrow(v, np.zeros(3, dtype = np.float32), length = 0.2, scale=0.8, color = (1, 0, 0))
        vis.add_geometry(f"arrow {i+1}", arrow)
   
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()
    app.quit()

def get_all_rotation(samples, verbose_check = False):
    normed_dir = samples / np.linalg.norm(samples, axis = -1, keepdims = True)
    z_axis = np.tile(np.float32([0, 0, 1]), (samples.shape[0], 1))
    Rs = np_rotation_between(z_axis, -normed_dir)
    if verbose_check:
        print(Rs.shape)
        for R in Rs:            # check orthogonality
            print(R @ R.T)
        rot_samples = Rs @ np.tile(np.float32([0, 0, 1]).reshape(3, 1), (samples.shape[0], 1, 1))
        print((rot_samples.squeeze() * normed_dir).sum(axis = -1))
    return Rs

def get_3d_projection(pos_3d: np.ndarray, Rs: np.ndarray, ts: np.ndarray):
    """ 3D to 2D projection 
        uniform hemispherical sampling
    """
    diff_t_world = (pos_3d.reshape(1, 3) - ts)[..., None]
    ts_cam = (np.transpose(Rs, (0, 2, 1)) @ diff_t_world).squeeze()
    ts_cam /= ts_cam[:, -1:]         # normalized by Z
    return (K[None, ...] @ ts_cam[..., None]).squeeze()[:, :-1].astype(np.int32)

def perturbed_projection():
    """ Noisy projection (noise for 3D position and pose) """
    pass

def output(Rs: np.ndarray, ts: np.ndarray, gt_pos: np.ndarray, pix_pos: np.ndarray):
    Ts = np.concatenate((Rs, ts[..., None]), axis = -1)
    json_file = {
        "camera-tranforms": Ts.tolist(),
        "camera-intrinsic": K.tolist(),
        "pixel-positions": pix_pos.tolist(),
        "gt": gt_pos.tolist()
    }
    matlab_file = {
        "camera-tranforms": Ts,
        "camera-intrinsic": K,
        "pixel-positions": pix_pos,
        "gt": gt_pos
    }
    with open("./data1.json", 'w', encoding = 'utf-8') as file:
        json.dump(json_file, file, indent = 4)
    sio.savemat("./data1.mat", matlab_file)

if __name__ == "__main__":
    sh_samples = quasi_uniform_eighth_spherical_sampling()
    # sh_samples are T
    sh_samples = radial_perturb(sh_samples)
    # rotation matrices
    Rs = get_all_rotation(sh_samples, True)
    gt_pos = np.float32([-0.5, 0.2, -0.2])
    all_proj = get_3d_projection(gt_pos, Rs, sh_samples)
    output(Rs, sh_samples, gt_pos, all_proj)
    visualize_spherical_samples(sh_samples)