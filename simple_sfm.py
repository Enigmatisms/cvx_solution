#-*-coding:utf-8-*-
"""
    Python test for SfM using the given feature points
    If this module succeeds in doing the things I want, it can be scaled into our model
    Co-visibility SfM feature point position optimization
    @author: Qianyue He
    @date: 2022-12-13
"""

import scipy
import scipy.io
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

from utils import *

def solve_3d_position(wf_origins: np.ndarray, wf_ray_dir: np.ndarray, verbose = False):
    As = np.eye(3) - wf_ray_dir @ np.transpose(wf_ray_dir, [0, 2, 1])
    p = np.sum(As @ wf_origins, axis = 0)
    A = np.sum(As, axis = 0)
    u, s, vt = np.linalg.svd(A)
    abs_last_v = abs(s[-1])
    if abs_last_v < 1e-3:
        if verbose:
            print(f"The last eigen value is too small: {abs_last_v}. Ill-conditioned matrix.") 
        s[-1] = s[-1] / abs_last_v * 1e-3 
    inv_A = vt.T @ np.diag(1 / s) @ u.T
    return inv_A @ p 