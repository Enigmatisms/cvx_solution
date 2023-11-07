#-*-coding:utf-8-*-
"""
    3D point position optimization, serves as baseline method
    @author: Qianyue He
    @date: 2023-11-6
"""

import time
import torch
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
        # `As` is of shape (N, 3, 3)
        As = np.tile(np.eye(3, dtype = np.float32), (self.wf_ray_dir.shape[0], 1, 1)) - self.wf_ray_dir[..., None] @ self.wf_ray_dir[:, None, :]
        p = np.sum(As @ self.wf_origins[..., None], axis = 0)       # of shape (3, 1)
        A = np.sum(As, axis = 0)                                    # shape (3, 3)
        print(A)
        print(p)
        return (np.linalg.inv(A) @ p).ravel()
    
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
        self.pos = cp.Variable(3)

    def solve(self, verbose = False):
        """ The cvxpy 3D space solver """
        mat_A  = np.zeros((3, 3), dtype = np.float32)
        mat_Ap = np.zeros((1, 3), dtype = np.float32)
        if self.huber_param >= 1e-2:
            loss = 0
            diff = self.wf_origins - self.pos[None, :]
            sqrt_diff = cp.sqrt(diff)
            mat = diff / sqrt_diff - self.wf_ray_dir
            items = cp.sum(cp.multiply(mat, diff), axis = 1)
            for item in items:
                loss += cp.huber(item, self.huber_param)
            problem = cp.Problem(cp.Minimize(loss))
        else:
            for ray_o, ray_d in zip(self.wf_origins, self.wf_ray_dir):
                A  = np.eye(3) - ray_d[:, None] @ ray_d[None, :]
                Ap = ray_o[None, :] @ A 
                mat_A  += A
                mat_Ap += Ap
            problem = cp.Problem(cp.Minimize(cp.quad_form(self.pos, mat_A) - 2 * mat_Ap @ self.pos))
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
    
class Solver2DReprojectionErr(torch.nn.Module, SolverBase):
    """ This is the solver for minizing 2D space reprojection error
        This method is potentially non-convex (since we are solving things in a 2D space)
        Projection is a non-linear transform
    """
    def __init__(self, pix_pos: np.ndarray, Ts: np.ndarray, K: np.ndarray, max_iter, huber_param = -1.0) -> None:
        torch.nn.Module.__init__(self)
        SolverBase.__init__(self, max_iter, huber_param)
        self.pix_pos = torch.from_numpy(pix_pos).to(torch.float32).cuda()
        self.Rs      = torch.from_numpy(Ts[:, :, :-1]).cuda()    # rotation
        self.ts      = torch.from_numpy(Ts[:, :, -1:]).cuda()    # translation
        self.K       = torch.from_numpy(K).cuda()
        self.pos = torch.nn.Parameter(torch.normal(0, 0.2, (3,), dtype = torch.float32).cuda())

    def solve(self, verbose = False):
        """ The torch 2D space solver (This might not be convex)
            self.pos is projected onto the 2D screen space and the distance to the pixel pos is to be minimized
        """
        torch.autograd.set_detect_anomaly(True)
        opt = torch.optim.Adam(self.parameters(), lr = 1e-3)
        for i in range(self.max_iter):
            opt.zero_grad()
            cf_pos = torch.transpose(self.Rs, 2, 1) @ (self.pos[None, :, None] - self.ts)     # shape (N, 3, 1)
            pix_coords = self.K[None, ...] @ cf_pos
            pix_coords = pix_coords / pix_coords[:, -1:, :]
            pix_coords = pix_coords.squeeze()[:, :2]        # (N, 2)
            loss = ((pix_coords - self.pix_pos) ** 2).mean()
            loss.backward()
            opt.step()
            if i % 20 == 0:
                print(f"Step {i + 1:4d} / {self.max_iter}. Loss = {loss.item():.5f}")

        solution = self.pos.detach().cpu().numpy()
        return solution
    
class Solver3DHuberDistance(SolverBase):
    """ This solver can be implemented via Pytorch, but not cvxpy (DCP Error)"""
    pass

class SolverMultiple3DPoints(SolverBase):
    """ Multiple points problem"""
    pass
    
if __name__ == "__main__":
    print("This script is not to be executed.")
