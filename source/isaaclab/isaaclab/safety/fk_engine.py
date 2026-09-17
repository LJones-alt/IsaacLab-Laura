import torch
import torch.nn as nn

# fume hood geometry Bounds (metres)
X_MIN, X_MAX = -0.73, 0.73
Y_MIN, Y_MAX = -0.2,  0.4
Z_MIN, Z_MAX =  0.0,  0.9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# franka panda joint limits (7 Joints in radians)
Q_MIN = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], device=DEVICE)
Q_MAX = torch.tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], device=DEVICE)

# Sphere radius surrounding robot link segments 
SPHERE_RADIUS = 0.04 


class Panda7DOF_DH(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # Official Franka Panda DH Table Parameters [a, d, alpha]
        self.register_buffer('dh_params', torch.tensor([
            [0.0,     0.333, -torch.pi/2], # Link 1
            [0.0,     0.0,    torch.pi/2], # Link 2
            [0.0825,  0.316,  torch.pi/2], # Link 3
            [-0.0825, 0.0,   -torch.pi/2], # Link 4
            [0.0,     0.384,  torch.pi/2], # Link 5
            [0.088,   0.0,    torch.pi/2], # Link 6
            [0.0,     0.107,  0.0        ]  # Link 7 (EE)
        ], device=device))

    def get_ee_position(self, q):
        # Computes 3D end-effector position from joint configurations q
        batch_size = q.shape[0]
        T_curr = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)

        for i in range(7):
            a = self.dh_params[i, 0]
            d = self.dh_params[i, 1]
            alpha = self.dh_params[i, 2]
            theta = q[:, i]

            ct, st = torch.cos(theta), torch.sin(theta)
            ca, sa = torch.cos(alpha), torch.sin(alpha)
            zeros, ones = torch.zeros_like(ct), torch.ones_like(ct)

            row0 = torch.stack([ct, -st*ca,  st*sa, a*ct], dim=1)
            row1 = torch.stack([st,  ct*ca, -ct*sa, a*st], dim=1)
            row2 = torch.stack([zeros, sa.expand_as(ct), ca.expand_as(ct), d.expand_as(ct)], dim=1)
            row3 = torch.stack([zeros, zeros, zeros, ones], dim=1)

            mat = torch.stack([row0, row1, row2, row3], dim=1)
            T_curr = torch.bmm(T_curr, mat)

        return T_curr[:, :3, 3]

    def forward(self, q):
        return self.get_ee_position(q)

    def get_arm_spheres(self, q, active_links=None, spheres_per_link=3):
        if active_links is None:
            active_links = {
                'link_3': True,
                'link_4': True,
                'link_5': True,
                'link_6': True,
                'link_7': True
            }

        batch_size = q.shape[0]
        T_curr = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        joint_positions = [T_curr[:, :3, 3]]

        # Forward Kinematics Chain
        for i in range(7):
            a = self.dh_params[i, 0]
            d = self.dh_params[i, 1]
            alpha = self.dh_params[i, 2]
            theta = q[:, i]

            ct, st = torch.cos(theta), torch.sin(theta)
            ca, sa = torch.cos(alpha), torch.sin(alpha)
            zeros, ones = torch.zeros_like(ct), torch.ones_like(ct)

            row0 = torch.stack([ct, -st*ca,  st*sa, a*ct], dim=1)
            row1 = torch.stack([st,  ct*ca, -ct*sa, a*st], dim=1)
            row2 = torch.stack([zeros, sa.expand_as(ct), ca.expand_as(ct), d.expand_as(ct)], dim=1)
            row3 = torch.stack([zeros, zeros, zeros, ones], dim=1)

            mat = torch.stack([row0, row1, row2, row3], dim=1)
            T_curr = torch.bmm(T_curr, mat)
            joint_positions.append(T_curr[:, :3, 3])

        link_map = {
            'link_3': 3,
            'link_4': 4,
            'link_5': 5,
            'link_6': 6,
            'link_7': 7
        }

        spheres = []
        for name, joint_idx in link_map.items():
            if active_links.get(name, False):
                p_start = joint_positions[joint_idx - 1]
                p_end   = joint_positions[joint_idx]

                for k in range(spheres_per_link):
                    t = k / float(spheres_per_link - 1 if spheres_per_link > 1 else 1)
                    spheres.append(p_start * (1 - t) + p_end * t)

        return torch.stack(spheres, dim=1)

