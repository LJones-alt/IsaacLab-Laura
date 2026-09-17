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
        
        zero = torch.tensor(0.0, device=device)
        pi = torch.tensor(torch.pi, device=device)

        # DH parameters structured as: (a, d, alpha, theta)
        dh_params = [
            (zero, torch.tensor(0.333, device=device), zero, joint_angles[0]),        # Joint 1
            (zero, zero, -pi/2, joint_angles[1]),                                    # Joint 2
            (zero, torch.tensor(0.316, device=device), pi/2, joint_angles[2]),       # Joint 3
            (torch.tensor(0.0825, device=device), zero, pi/2, joint_angles[3]),      # Joint 4
            (torch.tensor(-0.0825, device=device), torch.tensor(0.384, device=device), -pi/2, joint_angles[4]), # Joint 5
            (zero, zero, pi/2, joint_angles[5]),                                    # Joint 6
            (torch.tensor(0.088, device=device), zero, pi/2, joint_angles[6]),       # Joint 7
            # Flange and tool static transformations (Indices 7, 8, 9)
            (zero, torch.tensor(0.107, device=device), zero, zero),
            (zero, zero, zero, -pi/4),
            (zero, torch.tensor(0.1034, device=device), zero, zero)
        ]

        T = torch.eye(4, device=device)
        joint_positions = []
        
        # Loop through all transformation steps
        for i, (a, d, alpha, q) in enumerate(dh_params):
            T_i = self.get_tf_mat(a, d, alpha, q)
            T = T @ T_i
            
            # Only record the origin for the first 7 actuated joint frames
            if i < 7:
                joint_positions.append(T[:3, 3])
            
        # Shape: (7, 3) representing the absolute 3D position of each joint frame
        self.joint_positions = torch.stack(joint_positions)

        self.cartsian_ee = T
        print(f"got T matrix: {T}")
        self.cartesian_ee_7d = self.matrix_to_pose_7d(T)
        return self.cartesian_ee_7d
    
    def get_joint_positions(self, joint_angles=None):
        """
        Returns the absolute 3D Cartesian coordinates [X, Y, Z] for each frame/joint 
        along the robot's kinematic chain as a PyTorch tensor of shape (N, 3).
        """
        if joint_angles is not None or self.joint_positions is None:
            if joint_angles is None:
                raise ValueError("No cached joint positions available. Pass 'joint_angles' to compute FK.")
            self.get_fk_solution(joint_angles)
        print(f"got joint positions: {self.joint_positions}")
        return self.joint_positions

    def matrix_to_pose_7d(self, T):
        """
        Converts a 4x4 homogenous transformation matrix into a 7D pose tensor 
        formatted precisely for Isaac Lab's DifferentialIKController.
        Layout: [X, Y, Z, qw, qx, qy, qz]
        """
        device = T.device
        
        #  Position extraction including the physical asset offset
        x_fixed = T[0, 3] - 0.0200
        y_fixed = T[1, 3] - 0.0200
        z_fixed = T[2, 3] + 0.0534
        position = torch.stack([x_fixed, y_fixed, z_fixed])
        
        # Linearize Rotation Matrix
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