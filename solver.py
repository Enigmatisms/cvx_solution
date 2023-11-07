#-*-coding:utf-8-*-
"""
    3D point position optimization, serves as baseline method
    @author: Qianyue He
    @date: 2023-11-6
"""

import time
import cvxpy as cp
import numpy as np

__all__ = ['AnalyitcalSolver', 'Solver3DSpaceDistance', 'Solver2DReprojectionErr']

class AnalyitcalSolver:
    def __init__(self, wf_origins: np.ndarray, wf_ray_dir: np.ndarray):
        self.wf_origins = wf_origins
        self.wf_ray_dir = wf_ray_dir

    def solve(self, verbose = False):
        """ Analytical solution to the 3D point estimation problem
            wf_origins: world frame ray origins
            wf_ray_dir: world frame ray directions
        """
        As = np.eye(3) - self.wf_ray_dir @ np.transpose(self.wf_ray_dir, [0, 2, 1])
        p = np.sum(As @ self.wf_origins, axis = 0)
        A = np.sum(As, axis = 0)
        u, s, vt = np.linalg.svd(A)
        abs_last_v = abs(s[-1])
        if abs_last_v < 1e-3:
            if verbose:
                print(f"The last eigen value is too small: {abs_last_v}. Ill-conditioned matrix.") 
            s[-1] = s[-1] / abs_last_v * 1e-3 
        inv_A = vt.T @ np.diag(1 / s) @ u.T
        return inv_A @ p 
    
class SolverBase:
    def __init__(self, max_iter, huber_param = -1.0) -> None:
        self.huber_param = huber_param
        self.max_iter = max_iter
        self.pos = cp.Variable(3)    # 3D position

    def solve(self, verbose = False):
        raise NotImplementedError("Solver base does not implement anything yet.")

class Solver3DSpaceDistance(SolverBase):
    """ This is the solver for minizing 3D space point-to-line distance
        The model is convex
    """
    def __init__(self, wf_origins: np.ndarray, wf_ray_dir: np.ndarray, max_iter, huber_param = -1.0) -> None:
        super().__init__(max_iter, huber_param)
        self.wf_origins = wf_origins
        self.wf_ray_dir = wf_ray_dir
        self.pos = cp.Variable((1, 3))

    def solve(self, verbose = False):
        """ The cvxpy 3D space solver """
        pos_diff    = self.wf_origins - self.pos
        proj_length = cp.multiply(self.wf_ray_dir, pos_diff).sum(axis = 1)             # stupid CVXPY does not recognize axis = -1, what a joke
        print(proj_length)
        distance2   = cp.multiply(pos_diff, pos_diff).sum(axis = 1) - proj_length ** 2
        if self.huber_param > 1e-2:         # valid huber param
            loss = 0
            for item in distance2:
                loss += cp.huber(item, self.huber_param)
            problem = cp.Problem(cp.Minimize(loss))
        else:
            problem = cp.Problem(cp.Minimize(cp.sum(distance2)))
        start_time = time.time()
        if verbose:
            print(f"Start solving... ray num: {self.wf_origins.shape[0]}. Huber Loss Used = [{self.huber_param > 1e-2}]")

        problem.solve()
        end_time = time.time()
        solution = self.pos.value.ravel()
        if verbose:
            print(f"Problem solved. Time consumption: {end_time - start_time:.3f}")
            print("The optimal value is", problem.value)
            print("Optimal solution:", solution.ravel())
        return solution
    
class Solver2DReprojectionErr(SolverBase):
    """ This is the solver for minizing 2D space reprojection error
        This method is potentially non-convex (since we are solving things in a 2D space)
        Projection is a non-linear transform
    """
    def __init__(self, pix_pos: np.ndarray, Ts: np.ndarray, K: np.ndarray, max_iter, huber_param = -1.0) -> None:
        super().__init__(max_iter, huber_param)
        self.pix_pos = pix_pos
        self.Rs      = Ts[:, :, :-1]    # rotation
        self.ts      = Ts[:, :, -1:]    # translation
        self.K       = K
        self.pos = cp.Variable((3, 1))

    def solve(self, verbose = False):
        """ The cvxpy 2D space solver (This might not be convex)
            self.pos is projected onto the 2D screen space and the distance to the pixel pos is to be minimized
        """
        # stupid CVXPY does not support Atom with more than 2 dimension! So I have to for-loop the following computation, what a joke.
        loss = 0
        for R, t, pix in zip(self.Rs, self.ts, self.pix_pos):
            cf_pos = R.T @ (self.pos - t)
            pix_coords = (self.K @ cf_pos)[:2, :]
            if self.huber_param > 1e-2:
                loss += cp.huber(pix.astype(np.float32) - pix_coords, self.huber_param)
            else:
                loss += ((pix.astype(np.float32) - pix_coords) ** 2).sum()

        problem = cp.Problem(cp.Minimize(loss))
        start_time = time.time()
        if verbose:
            print(f"Start solving... pixel num: {self.pix_pos.shape[0]}. Huber Loss Used = [{self.huber_param > 1e-2}]")

        problem.solve()
        end_time = time.time()
        solution = self.pos.value.ravel()
        if verbose:
            print(f"Problem solved. Time consumption: {end_time - start_time:.3f}")
            print("The optimal value is", problem.value)
            print("Optimal solution:", solution.ravel())
        return solution
    
if __name__ == "__main__":
    print("This script is not to be executed.")
