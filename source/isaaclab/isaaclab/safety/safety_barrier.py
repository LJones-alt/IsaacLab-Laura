import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from .fk_engine import Panda7DOF_DH, Q_MIN, Q_MAX, SPHERE_RADIUS, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX


T_MAX = 0.01  # seconds — trajectory horizon

class Sin(nn.Module):
    def forward(self, x): return torch.sin(x)

class JP_Mod(nn.Module):
    def __init__(self, device):
        super(JP_Mod, self).__init__()
        self.device = device
        self.fk_engine=Panda7DOF_DH(device)
        self.net = nn.Sequential(
            nn.Linear(25, 512), Sin(),
            nn.Linear(512, 512), Sin(),
            nn.Linear(512, 512), Sin(),
            nn.Linear(512, 1)
        ).to(self.device)

    def forward(self, state):
        """
        Args:
            state (torch.Tensor): Combined input batch [batch_size, 18]

        Returns:
            torch.Tensor: Safety Value V(state) [batch_size, 1]
        """
        # unpack variables directly from state tensor
        q     = state[:, :7]       # joint angles
        t     = state[:, 14:15]    # time
        walls = state[:, 15:21]    # fume hood boundaries
        obs   = state[:, 21:]  

        # compute smooth whole-arm SDF clearance l(x) inside forward pass
        l_x = get_whole_arm_sdf(q, walls, obs, self.fk_engine, training=False)

        # compute neural network residual output
        nn_out = self.net(state).view(-1, 1)

        # enforce Exact Boundary Condition: V(x, T_MAX) = l(x)
        return l_x + (T_MAX - t) * nn_out


def compute_sdf(pos, walls, obs, k=15.0):
    px, py, pz = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
    xmin, xmax = walls[:, 0:1], walls[:, 1:2]
    ymin, ymax = walls[:, 2:3], walls[:, 3:4]
    zmin, zmax = walls[:, 4:5], walls[:, 5:6]

    d_walls = torch.cat([
        px - xmin - SPHERE_RADIUS,
        xmax - px - SPHERE_RADIUS,
        py - ymin - SPHERE_RADIUS,
        ymax - py - SPHERE_RADIUS,
        pz - zmin - SPHERE_RADIUS,
        zmax - pz - SPHERE_RADIUS,
    ], dim=1)

    ox, oy, oz, r = obs[:, 0:1], obs[:, 1:2], obs[:, 2:3], obs[:, 3:4]
    d_obs = torch.sqrt((px - ox)**2 + (py - oy)**2 + (pz - oz)**2 + 1e-6) - r - SPHERE_RADIUS

    all_boundaries = torch.cat([d_walls, d_obs], dim=1)
    return torch.min(all_boundaries, dim=1, keepdim=True)[0]

def get_whole_arm_sdf(q, walls, obs, fk_engine, active_links=None, k=15.0, training=True):
    sphere_positions = fk_engine.get_arm_spheres(q, active_links=active_links)
    n_spheres = sphere_positions.shape[1]

    l_x_list = [compute_sdf(sphere_positions[:, s, :], walls, obs, k=k) for s in range(n_spheres)]
    all_sphere_dists = torch.cat(l_x_list, dim=1)

    if training:
        # Softmin — gradients flow through all spheres during training
        return -(1.0 / k) * torch.logsumexp(-k * all_sphere_dists, dim=1, keepdim=True)
    else:
        # Hard min — accurate SDF for boundary term and inference
        return torch.min(all_sphere_dists, dim=1, keepdim=True)[0]

class J_Vals:
    def __init__(self,vals):
        self.vals=vals

class JP_APF():
    def __init__(self, device,obst):
        self.device=device
        self.obst=obst
        self.x_min=-0.2
        self.x_max=0.40
        self.y_min=-0.73
        self.y_max=0.73 
        self.z_min=0.0 
        self.z_max=0.9
        self.obst_loc = obst # location of the obstacle in cartesian
        self.model_path = "docs/apf/25_dims_tmax_001.pth"
        self.model=JP_Mod(self.device)
        self._load_model()
    
    def _load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.net.eval()

    def get_joint_vals(self, q, q_dot, t_val):#-> J_Vals:
        #]q_dot=q_dot.unsqueeze(0).requires_grad_(True)
        # for debug
        #q_dot = q_dot[:, :-2]
        #print(f"[DEBUG] q_dot  :{q_dot}")
        q=q.unsqueeze(0).requires_grad_(True)
        #q_dot=q_dot.unsqueeze(0).requires_grad_(True)
        #print(f"[DEBUG] q: {q}")
        # build input tensor
        walls= torch.tensor([[self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max]], device=self.device).requires_grad_(True).float()
        #print(f"[DEBUG] walls: {walls}")
        test_obs = torch.tensor([[0.8, 0.0, 0.9, 0.1]], device = self.device).requires_grad_(True).float()
        #print(f"[DEBUG] test_obs: {test_obs}")
        t_tensor_0 = torch.tensor([[t_val]], device=self.device).requires_grad_(True).float()
        #q_tensor= torch.tensor(q, device=self.device,requires_grad = False).float()
        #print(f"[DEBUG] t_tensor_0: {t_tensor_0}")
       # print(f"[DEBUG] q: {q}")
        #input=torch.cat([q,q_dot, t_tensor_0, walls, test_obs], dim=-1)
        input=torch.cat([q, q_dot, t_tensor_0, walls, test_obs], dim=-1)
        #print(f"[DEBUG] input shape: {input.shape}")
        #print(f"[DEBUG] input: {input}")
        #input=input.unsqueeze(0)

        output_min = None
       # outputs = self.model(input)
        #pass into model?
        with torch.enable_grad():
            outputs = self.model(input)
            #print(f"[DEBUG] outputs: {outputs}")
            #outputs=outputs.detach().cpu().flatten()
       # print(f"[DEBUG] outputs: {outputs}")
        if output_min is None:
            output_min=outputs
        else :
            output_min= np.minimum(output_min, outputs)
        
        grads=self._get_grads(q, outputs)
        #vals=J_Vals(outputs)


       
        return outputs,grads

    def _get_grads(self,q, outputs):
        with torch.enable_grad():
           
           # print(f"[DEBUG] : q_clone : {q_clone}")
            #print(f"[DEBUG] : output_clone : {outputs_clone}")
            grads = torch.autograd.grad(
                outputs=outputs,
                inputs=q,
                grad_outputs=torch.ones_like(outputs),
                create_graph=False,
                allow_unused=True
            )
        #print(f"[DEBUG] gradient of value: {grads}")
        return grads
            



        

        
