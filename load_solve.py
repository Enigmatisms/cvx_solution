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
    pix = np.float32(data['pixel-position'])
    gt  = np.float32(data['gt'])
    wf_ray_dir = Ts[..., :3] @ np.tile(np.float32([[0], [0], [1]]), (Ts.shape[0], 1, 1))
    wf_ray_dir = wf_ray_dir.squeeze()
    wf_ray_dir /= np.linalg.norm(wf_ray_dir, axis = -1, keepdims = True)
    return Ts, K, pix, gt, wf_ray_dir

if __name__ == "__main__":
    print("This is a drill.")
    opts = parse_args()
    Ts, K, pix, gt, wf_ray_dir = load_json(opts.input)

    if opts.solver == '3d':
        solver = Solver3DSpaceDistance(Ts[..., -1], wf_ray_dir, opts.max_iter, opts.huber_param)
    elif opts.solver == 'analytical':
        solver = AnalyitcalSolver(Ts[..., -1], wf_ray_dir)
    else:
        solver = Solver2DReprojectionErr(pix, Ts, K, opts.max_iter, opts.huber_param)

    solution = solver.solve(opts.verbose)
    print(f"The solution is {solution}")
    print(f"GT is {gt}")


