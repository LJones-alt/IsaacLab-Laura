import torch

class ForwardsDynamics:
    def __init__(self):
        self.cartsian_ee = None
    

    def get_tf_mat(self, a, d, alpha, q):
        cos_q = torch.cos(q)
        sin_q = torch.sin(q)
        cos_alpha = torch.cos(alpha)
        sin_alpha = torch.sin(alpha)

        # Mathematically correct Modified DH (Craig's convention)
        row0 = torch.stack([cos_q, -sin_q, torch.zeros_like(q), a])
        row1 = torch.stack([sin_q * cos_alpha, cos_q * cos_alpha, -sin_alpha, -sin_alpha * d])
        row2 = torch.stack([sin_q * sin_alpha, cos_q * sin_alpha, cos_alpha, cos_alpha * d])
        row3 = torch.tensor([0.0, 0.0, 0.0, 1.0], device=q.device)

        return torch.stack([row0, row1, row2, row3])


    def get_fk_solution(self, joint_angles):
        device = joint_angles.device
        print(f"got joint angles: {joint_angles}")
        # Pre-define constant tensors on the target device
        zero = torch.tensor(0.0, device=device)
        pi = torch.tensor(torch.pi, device=device)
        half_pi = pi / 2

        # DH parameters structured as: (a, d, alpha, theta)
        dh_params = [
            (zero, torch.tensor(0.333, device=device), zero, joint_angles[0]),
            (zero, zero, -pi/2, joint_angles[1]),
            (zero, torch.tensor(0.316, device=device), pi/2, joint_angles[2]),
            (torch.tensor(0.0825, device=device), zero, pi/2, joint_angles[3]),
            (torch.tensor(-0.0825, device=device), torch.tensor(0.384, device=device), -pi/2, joint_angles[4]),
            (zero, zero, pi/2, joint_angles[5]),
            (torch.tensor(0.088, device=device), zero, pi/2, joint_angles[6]),
            # Flange and tool static transformations
            (zero, torch.tensor(0.107, device=device), zero, zero),
            (zero, zero, zero, -pi/4),
            (zero, torch.tensor(0.1034, device=device), zero, zero)
        ]

        T = torch.eye(4, device=device)
        
        # Loop through all 10 transformation steps
        for a, d, alpha, q in dh_params:
            T_i = self.get_tf_mat(a, d, alpha, q)
            T = T @ T_i
            
        self.cartsian_ee =T
        print(f"got T matrix: {T}")
        self.cartesian_ee_7d = self.matrix_to_pose_7d(T)
        return self.cartesian_ee_7d

    def matrix_to_pose_7d(self, T):
        """
        Converts a 4x4 homogenous transformation matrix into a 7D pose tensor 
        formatted precisely for Isaac Lab's DifferentialIKController.
        Layout: [X, Y, Z, qw, qx, qy, qz]
        """
        device = T.device
        
        # 1. Position extraction including the physical asset offset
        x_fixed = T[0, 3] - 0.0200
        y_fixed = T[1, 3] - 0.0200
        z_fixed = T[2, 3] + 0.0534
        position = torch.stack([x_fixed, y_fixed, z_fixed])
        
        # 2. Linearize Rotation Matrix
        R = T[:3, :3]
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        
        if tr > 0:
            S = torch.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        # Standard Isaac Lab [W, X, Y, Z] representation
        quaternion = torch.stack([qw, qx, qy, qz])
        
        print(f"calculated  EE : {position}, {quaternion}")
        return torch.cat([position, quaternion])