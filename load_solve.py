#-*-coding:utf-8-*-
"""
    Executable file for solving the problem
    @author: Qianyue He
    @date: 2023-11-7
"""
import json
import numpy as np
import scipy.io as sio
import configargparse
from solver import *

def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument("-i", "--input",  type = str, default = "./data1.json", help = "Path to the input data.")
    parser.add_argument("-s", "--solver", type = str, default = "3d",   choices = ["analytical", "3d", "2d"],  
                                            help = "Solver to use (3 choices)")
    parser.add_argument("--max_iter",     type = int, default = 8000,   
                                            help = "Max number of iteration (currently useless)")
    parser.add_argument("--huber_param",  type = float, default = -1.0, 
                                            help = "Huber Loss parameter. Value less than 0.01 means no Huber loss.")
    parser.add_argument("-v", "--verbose", default = False, action = "store_true", help = "Output some intermediate information")
    return parser.parse_args()

def load_json(path):
    """ load data from json file 
        Generate ray (from 2D pixel and camera poses)
    """
    with open(path, 'r') as file:
        data = json.load(file)
    Ts  = np.float32(data['camera-tranforms'])
    K   = np.float32(data['camera-intrinsic'])
    pix = np.float32(data['pixel-positions'])
    gt  = np.float32(data['gt'])
    homo_pix = np.concatenate((pix, np.ones((pix.shape[0], 1), dtype = pix.dtype)), axis = -1)
    cf_dir = np.linalg.inv(K)[None, ...] @ homo_pix[..., None]
    wf_ray_dir = Ts[..., :3] @ cf_dir
    wf_ray_dir = wf_ray_dir.squeeze()
    wf_ray_dir /= np.linalg.norm(wf_ray_dir, axis = -1, keepdims = True)
    return Ts, K, pix, gt, wf_ray_dir

def visualize_rays(wf_ray_dir: np.ndarray, ray_o: np.ndarray):
    import open3d as o3d
    import open3d.visualization.gui as gui
    from utils import get_arrow, np_rotation_between

    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - Spherical sample visualize", 1024, 768)
    vis.set_background((0.3, 0.3, 0.3, 1), None)
    vis.show_settings = True
    vis.show_skybox(False)
    vis.show_axes = True

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ray_o)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.float32([0, 0, 1]), (ray_o.shape[0], 1)))
    vis.add_geometry('sample pos', pcd)
    for i, (po, v) in enumerate(zip(ray_o, wf_ray_dir)):
        arrow = get_arrow(po, po + v * 2, length = 0.2, scale=0.8, color = (1, 0, 0))
        vis.add_geometry(f"arrow {i+1}", arrow)
   
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()
    app.quit()

if __name__ == "__main__":
    print("This is a drill.")
    opts = parse_args()
    Ts, K, pix, gt, wf_ray_dir = load_json(opts.input)

    # visualize_rays(wf_ray_dir, Ts[..., -1])

    if opts.solver == '3d':
        solver = Solver3DSpaceDistance(Ts[..., -1], wf_ray_dir, opts.max_iter, opts.huber_param)
    elif opts.solver == 'analytical':
        solver = AnalyitcalSolver(Ts[..., -1], wf_ray_dir)
    else:
        solver = Solver2DReprojectionErr(pix, Ts, K, opts.max_iter, opts.huber_param)

    solution = solver.solve(opts.verbose)
    print(f"The solution is {solution}")
    print(f"GT is {gt}")


