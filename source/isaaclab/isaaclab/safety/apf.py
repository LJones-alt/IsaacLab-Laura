import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
#region CONFIG

class Hazard:
    def __init__(self, x, y, z, r, name='obstacle'):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.name = name

class Sin(nn.Module):
    def forward(self, x): return torch.sin(x)

class ExactBCNet(nn.Module):
    def __init__(self, device, t_max):
        super().__init__()
        # Input: [px, py, pz, t, xmin...zmax, ox, oy, oz, r] = 14 dims
        self.device = device
        self.t_max = t_max
        # 4 layers
        self.net = nn.Sequential(
            nn.Linear(14, 512), Sin(),
            nn.Linear(512, 512), Sin(),
            nn.Linear(512, 512), Sin(),
            nn.Linear(512, 1)
        ).to(self.device)
    
    def target_function(self, x, walls, obs_params):
        # x: [N, 3], walls: [N, 6], obs_params: [N, 4]
        px, py, pz = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        xmin, xmax, ymin, ymax, zmin, zmax = walls.unbind(1)
        # Ensure walls are [N, 1] for subtraction
        xmin, xmax, ymin, ymax, zmin, zmax = xmin.view(-1,1), xmax.view(-1,1), ymin.view(-1,1), ymax.view(-1,1), zmin.view(-1,1), zmax.view(-1,1)
        
        d_walls = torch.cat([px - xmin, xmax - px, py - ymin, ymax - py, pz - zmin, zmax - pz], dim=1)
        min_d_walls = torch.min(d_walls, dim=1, keepdim=True)[0]
        
        ox, oy, oz, r = obs_params[:, 0:1], obs_params[:, 1:2], obs_params[:, 2:3], obs_params[:, 3:4]
        d_sphere = torch.sqrt((px-ox)**2 + (py-oy)**2 + (pz-oz)**2 + 1e-6) - r
        
        return torch.min(min_d_walls, d_sphere)

    def forward(self, state_aug):
        # state_aug: [N, 14]
        x, t = state_aug[:, :3], state_aug[:, 3:4]
        walls, obs = state_aug[:, 4:10], state_aug[:, 10:]
        
        l_x = self.target_function(x, walls, obs)
        # Force the neural network output to be [N, 1]
        nn_out = self.net(state_aug).view(-1, 1)
        
        return l_x + (self.t_max - t) * nn_out

class APF():
    def __init__(self, hazards: list[Hazard], x_min:float, x_max:float, y_min:float, y_max:float, z_min:float, z_max:float, plot_graph : bool = False, name:str="apf", retrain:bool=True):
        self.hazards = hazards
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.velocity_limit = 2.0
        self.t_max = 2.0 # max time from now
        self.lr = 2e-5
        self.batch_size = 6500
        self.pretrain_iters = 20000
        self.train_iters = 20000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ExactBCNet(self.device, self.t_max)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.name = name
        # now run the functions
        if retrain:
            self.train()

        if plot_graph:
            self.plot_with_obstacle(self.net, 0.02, self.hazards, 0.2)    

    def compute_safe_velocity(self, current_pos, nominal_velocity, safety_margin=0.01, repulsion_gain=2.0):
        """
        Compute a safe velocity by modifying the nominal velocity using the APF gradient.
        
        Args:
            current_pos: torch.Tensor or array-like [3] (EE position)
            nominal_velocity: torch.Tensor or array-like [3] (Desired EE velocity/delta)
            safety_margin: float, distance at which repulsion starts
            repulsion_gain: float, gain for the repulsive force
            
        Returns:
            safe_velocity: torch.Tensor [3]
        """
        if not isinstance(current_pos, torch.Tensor):
            current_pos = torch.tensor(current_pos, dtype=torch.float32, device=self.device)
        if not isinstance(nominal_velocity, torch.Tensor):
            nominal_velocity = torch.tensor(nominal_velocity, dtype=torch.float32, device=self.device)
            
        px, py, pz = current_pos[0], current_pos[1], current_pos[2]
        # Create pts_tensor [1, 4] with t=0
        pts_tensor = torch.tensor([[px, py, pz, 0.0]], dtype=torch.float32, device=self.device).requires_grad_(True)
        
        # Construct walls tensor [1, 6] based on the bounds provided to APF
        walls_tensor = torch.tensor(
            [[self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max]],
            dtype=torch.float32, device=self.device
        )
        
        v_vals = []
        # Query the network for EVERY obstacle independently
        for haz in self.hazards:
            obs_tensor = torch.tensor([[haz.x, haz.y, haz.z, haz.r]], dtype=torch.float32, device=self.device)
            
            # Combine state, walls, and this specific obstacle into 14 dims
            state_aug = torch.cat([pts_tensor, walls_tensor, obs_tensor], dim=1)
            
            # Get Value Function for this one obstacle
            v = self.net(state_aug)
            v_vals.append(v)
            
        if not v_vals:
            # Revert to standard wall-only safety if no hazards are defined
            # (Though in your case hazards are defined in the teleop script)
            return nominal_velocity
        #print(f"V_vals: {v_vals}")
        # Take the minimum over all obstacle outputs (approximate SDF)
        V_min = torch.min(torch.stack(v_vals))
        
        # Compute analytical gradient of V_min with respect to position
        grads = torch.autograd.grad(outputs=V_min, inputs=pts_tensor)[0]
        
        grad_xyz = grads[0, :3].detach()
        V_val = V_min.item()
        
        safe_velocity = nominal_velocity.clone()
        # Push away from closest obstacle/wall if within safety margin
        if V_val < safety_margin:
            # The gradient points towards safer regions (increasing distance)
            # We scale it by the gain and add to nominal velocity
            repulsive_velocity = repulsion_gain * grad_xyz
            safe_velocity += repulsive_velocity
            
        return safe_velocity

    def sample_data(self, batch_size, t_min=0.0):
        pts = (torch.rand(batch_size, 3, device=self.device) - 0.5) * 2.5
        t = torch.rand(batch_size, 1, device=self.device) * (self.t_max - t_min) + t_min
        
        w = torch.tensor([self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max], device=self.device)
        walls = w.repeat(batch_size, 1) + (torch.rand(batch_size, 6, device=self.device) - 0.5) * 0.1
        
        o_pos = (torch.rand(batch_size, 3, device=self.device) - 0.5) * 0.6
        o_r = torch.rand(batch_size, 1, device=self.device) * 0.05 + 0.1 
        obs = torch.cat([o_pos, o_r], dim=1)
        
        return torch.cat([pts, t, walls, obs], dim=1)
    
    def train(self):
        wandb.init(
            name=self.name,
            project="place_environment_apf",
            config={
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "pretrain_iters": self.pretrain_iters,
                "train_iters": self.train_iters,
                "velocity_limit": self.velocity_limit,
                "t_max": self.t_max
            }
        )
        for i in tqdm(range(self.pretrain_iters), desc="Pretraining"):
            data = self.sample_data(self.batch_size, t_min=self.t_max)
            v = self.net.forward(data)
            l_x = self.net.target_function(data[:, :3], data[:, 4:10], data[:, 10:])
            loss = torch.mean((v - l_x)**2)
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            
            if i % 100 == 0:
                wandb.log({"pretrain_loss": loss.item(), "epoch": i})

        for i in tqdm(range(self.train_iters), desc="HJB Training"):
            t_start = max(0.0, self.t_max - (i/self.train_iters)*self.t_max)
            data = self.sample_data(self.batch_size, t_min=t_start).requires_grad_(True)
            v = self.net.forward(data)
            grads = torch.autograd.grad(v.sum(), data, create_graph=True)[0]
            dv_dx, dv_dt = grads[:, :3], grads[:, 3:4]
            ham = torch.norm(dv_dx, p=2, dim=1, keepdim=True) * self.velocity_limit
            l_x = self.net.target_function(data[:, :3], data[:, 4:10], data[:, 10:])
            loss = torch.mean(torch.min(dv_dt + ham, l_x - v)**2)
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            
            if i % 100 == 0:
                wandb.log({"hjb_loss": loss.item(), "t_start": t_start, "epoch": self.pretrain_iters + i})
        
        wandb.finish()
        torch.save(self.net.state_dict(), "/workspace/isaaclab/docs/apf/safety_wall_model.pth")

    def load_model(self, model_path):
        self.net.load_state_dict(torch.load(model_path))

    def plot_with_obstacle(self,model, epsilon, hazards_list, z_slice=0.5):
        res = 150
        # Create 1D arrays
        x_line = np.linspace(self.x_min, self.x_max, res)
        y_line = np.linspace(self.y_min, self.y_max, res)
        X, Y = np.meshgrid(x_line, y_line)
        num_pts = res**2
        
        # Building tensors carefully
        pts = np.zeros((num_pts, 4))
        pts[:, 0] = X.flatten()
        pts[:, 1] = Y.flatten()
        pts[:, 2] = z_slice
        pts[:, 3] = 0.0 # t=0
        
        walls_np = np.tile([self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max], (num_pts, 1))
        
        model.eval()
        
        v_vals_min = None
        for haz in hazards_list:
            obs_np = np.tile([haz.x, haz.y, haz.z, haz.r], (num_pts, 1))
            
            # Concatenate in numpy first
            input_np = np.hstack([pts, walls_np, obs_np])
            input_torch = torch.from_numpy(input_np).float().to(self.device)
            
            with torch.no_grad():
                v_vals = model(input_torch)
                v_vals = v_vals.detach().cpu().numpy().flatten()
                
            if v_vals_min is None:
                v_vals_min = v_vals
            else:
                v_vals_min = np.minimum(v_vals_min, v_vals)
                
        # Fallback if list is empty
        if v_vals_min is None:
            v_vals_min = np.zeros(num_pts)
            
        v_vals = v_vals_min.reshape(res, res)

        

        plt.figure(figsize=(10, 8))
        # Safe/Unsafe
        plt.contourf(X, Y, v_vals, levels=[-10, 0, 10], colors=['#ffcccc', '#e1f5fe'])
        plt.contourf(X, Y, v_vals - epsilon, levels=[-10, 0, 10], colors=['#ffcccc', '#ffffff'], alpha=0.6)
        
        # Boundary Decorations
        rect = plt.Rectangle((self.x_min, self.y_min), self.x_max-self.x_min, self.y_max-self.y_min, linewidth=3, edgecolor='black', fill=False, label="Walls")
        plt.gca().add_patch(rect)
        
        for haz in hazards_list:
            dist_to_center_z = abs(z_slice - haz.z)
            if dist_to_center_z < haz.r:
                r_slice = np.sqrt(haz.r**2 - dist_to_center_z**2)
                circ = plt.Circle((haz.x, haz.y), r_slice, color='red', alpha=0.4, label=f"Obstacle ({haz.name})")
                plt.gca().add_patch(circ)

        # Custom legend patches for safe zones
        safe_patch = mpatches.Patch(color='#e1f5fe', label='Safe Zone')
        vsafe_patch = mpatches.Patch(color='#ffffff', label='99% Verified Safe')
        unsafe_patch = mpatches.Patch(color='#ffcccc', label='Unsafe Zone')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Use dictionary to prevent duplicate obstacle labels
        by_label = dict(zip(labels, handles))
        by_label['Safe Zone'] = safe_patch
        by_label['99% Verified Safe'] = vsafe_patch
        by_label['Unsafe Zone'] = unsafe_patch
        
        plt.title(f"Franka Safe Workspace (Slice z={z_slice}m)\\nWhite = 99% Verified Safe")
        plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.axis('equal')
        plt.xlim(self.x_min, self.x_max); plt.ylim(self.y_min, self.y_max)
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig("docs/apf/reachable_tubes.png", bbox_inches="tight", transparent=False)
        
        # === Heatmap Plot ===
        plt.figure(figsize=(10, 8))
        c = plt.pcolormesh(X, Y, v_vals, cmap='viridis', shading='auto')
        plt.colorbar(c, label='Neural Network Value Function (Signed Distance)')
        plt.contour(X, Y, v_vals, levels=[0], colors='white', linewidths=2, linestyles='solid')
        plt.gca().add_patch(plt.Rectangle((self.x_min, self.y_min), self.x_max-self.x_min, self.y_max-self.y_min, linewidth=2, edgecolor='k', fill=False, label='Walls'))
        
        for haz in hazards_list:
            dist_to_center_z = abs(z_slice - haz.z)
            if dist_to_center_z < haz.r:
                r_slice = np.sqrt(haz.r**2 - dist_to_center_z**2)
                plt.gca().add_patch(plt.Circle((haz.x, haz.y), r_slice, color='red', alpha=0.6, label=f"Obstacle ({haz.name})"))

        hm_handles, hm_labels = plt.gca().get_legend_handles_labels()
        by_label_hm = dict(zip(hm_labels, hm_handles))
        
        by_label_hm['0-Level Safe Boundary'] = Line2D([0], [0], color='white', lw=2)
        plt.legend(by_label_hm.values(), by_label_hm.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(f"Value Function Heatmap (Slice z={z_slice}m)")
        plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.axis('equal')
        plt.xlim(self.x_min, self.x_max); plt.ylim(self.y_min, self.y_max)
        plt.savefig("docs/apf/value_heatmap.png", bbox_inches="tight", transparent=False)
        plt.close()


#region PLOT 3D
    def plot_safe_space_3d(self, model, epsilon, hazards_list):
        res = 40
        x_line = np.linspace(self.x_min, self.x_max, res)
        y_line = np.linspace(self.y_min, self.y_max, res)
        z_line = np.linspace(self.z_min, self.z_max, res)
        X, Y, Z = np.meshgrid(x_line, y_line, z_line, indexing='ij')
        num_pts = res**3
        pts = np.zeros((num_pts, 4))
        pts[:, 0] = X.flatten()
        pts[:, 1] = Y.flatten()
        pts[:, 2] = Z.flatten()
        pts[:, 3] = 0.0 # t=0
        
        walls_np = np.tile([self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max], (num_pts, 1))
        
        model.eval()
        v_vals_min = None
        for haz in hazards_list:
            obs_np = np.tile([haz.x, haz.y, haz.z, haz.r], (num_pts, 1))
            input_np = np.hstack([pts, walls_np, obs_np])
            input_torch = torch.from_numpy(input_np).float().to(self.device)
            
            with torch.no_grad():
                v_vals = model(input_torch).cpu().numpy().flatten()
                
            if v_vals_min is None:
                v_vals_min = v_vals
            else:
                v_vals_min = np.minimum(v_vals_min, v_vals)
                
        if v_vals_min is None: v_vals_min = np.zeros(num_pts)
        
        is_safe = v_vals_min > 0
        safe_pts = pts[is_safe]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(safe_pts[:,0], safe_pts[:,1], safe_pts[:,2], alpha=0.03, color='blue', s=5, label='Safe Space')
        
        for haz in hazards_list:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            xs = haz.r * np.outer(np.cos(u), np.sin(v)) + haz.x
            ys = haz.r * np.outer(np.sin(u), np.sin(v)) + haz.y
            zs = haz.r * np.outer(np.ones(np.size(u)), np.cos(v)) + haz.z
            ax.plot_surface(xs, ys, zs, color='red', alpha=0.5)
            
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_zlim(self.z_min, self.z_max)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        plt.title('Franka Safe Workspace (3D Volume)')
        plt.savefig('safe_space_3d.png', bbox_inches='tight')
        plt.close()
    #endregion


    #region PLOT VECTOR FIELD
    def plot_vector_field_3d(self, model, hazards_list):
        res = 12  # Coarse resolution for readable quiver plot
        x_line = np.linspace(self.x_min, self.x_max, res)
        y_line = np.linspace(self.y_min, self.y_max, res)
        z_line = np.linspace(self.z_min, self.z_max, res)
        X, Y, Z = np.meshgrid(x_line, y_line, z_line, indexing='ij')
        
        num_pts = res**3
        pts = np.zeros((num_pts, 4))
        pts[:, 0] = X.flatten()
        pts[:, 1] = Y.flatten()
        pts[:, 2] = Z.flatten()
        pts[:, 3] = 0.0 # t=0
        
        walls_np = np.tile([self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max], (num_pts, 1))
        
        pts_tensor = torch.from_numpy(pts).float().to(self.device).requires_grad_(True)
        walls_tensor = torch.from_numpy(walls_np).float().to(self.device)
        
        v_vals_all = []
        for haz in hazards_list:
            obs_tensor = torch.tensor([[haz.x, haz.y, haz.z, haz.r]], dtype=torch.float32, device=self.device).repeat(num_pts, 1)
            input_torch = torch.cat([pts_tensor, walls_tensor, obs_tensor], dim=1)
            v = model(input_torch)
            v_vals_all.append(v)
            
        if v_vals_all:
            v_vals_min = torch.min(torch.stack(v_vals_all, dim=1), dim=1)[0]
        else:
            # Fallback if no hazards
            dummy_obs = torch.zeros((num_pts, 4), device=self.device)
            input_torch = torch.cat([pts_tensor, walls_tensor, dummy_obs], dim=1)
            v_vals_min = model(input_torch)
            
        # Compute spatial gradient (which represents the optimal pushing direction)
        grads = torch.autograd.grad(outputs=v_vals_min.sum(), inputs=pts_tensor)[0]
        
        # Extract X, Y, Z gradients
        U = grads[:, 0].detach().cpu().numpy().reshape(res, res, res)
        V_grad = grads[:, 1].detach().cpu().numpy().reshape(res, res, res)
        W = grads[:, 2].detach().cpu().numpy().reshape(res, res, res)
        
        # Prepare figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Normalize vectors for consistent arrow length
        norm = np.sqrt(U**2 + V_grad**2 + W**2) + 1e-6
        length_scale = 0.04
        U_norm = U / norm * length_scale
        V_norm_vec = V_grad / norm * length_scale
        W_norm = W / norm * length_scale
        
        # Optional filtering to declutter: we plot everywhere to see the global optimal field
        ax.quiver(X, Y, Z, U_norm, V_norm_vec, W_norm, length=1.0, color='dodgerblue', alpha=0.6, arrow_length_ratio=0.3)
        
        # Draw transparent obstacles
        for haz in hazards_list:
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            xs = haz.r * np.outer(np.cos(u), np.sin(v)) + haz.x
            ys = haz.r * np.outer(np.sin(u), np.sin(v)) + haz.y
            zs = haz.r * np.outer(np.ones(np.size(u)), np.cos(v)) + haz.z
            ax.plot_surface(xs, ys, zs, color='red', alpha=0.5)
            
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_zlim(self.z_min, self.z_max)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        plt.title('Optimal Avoidance Vector Field (Gradient of Value Function)')
        plt.savefig('vector_field_3d.png', bbox_inches='tight')
        plt.close()





