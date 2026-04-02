import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wall Boundaries - fume hood internal dims
X_MIN, X_MAX = -0.45, 0.45
Y_MIN, Y_MAX = -0.1, 0.535
Z_MIN, Z_MAX =  0.0, 0.9
VELOCITY_LIMIT = 2.0
T_MAX = 2.0 # max time from now

class Sin(nn.Module):
    def forward(self, x): return torch.sin(x)

class Hazard:
    def __init__(self, x, y, z, r, name='obstacle'):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.name = name

class APF(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [px, py, pz, t, xmin...zmax, ox, oy, oz, r] = 14 dims

        # 4 layers
        self.net = nn.Sequential(
            nn.Linear(14, 512), Sin(),
            nn.Linear(512, 512), Sin(),
            nn.Linear(512, 512), Sin(),
            nn.Linear(512, 1)
        ).to(DEVICE)

        self.walls_tensor = torch.tensor(
            [[X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX]],
            dtype=torch.float32, device=DEVICE
        )
        self.hazards = [
            Hazard(x=0.2, y=0.2, z=0.2, r=0.0, name='obstacle1'),
            Hazard(x=-0.2, y=-0.05, z=0.2, r=0.0, name='obstacle2')
        ]
        self.obs_tensor = self.create_obs()
    
    def create_obs(self):
        obs_list = []
        for h in self.hazards:
            obs_list.append([h.x, h.y, h.z, h.r])
        return torch.tensor(obs_list, dtype=torch.float32, device=DEVICE)
    
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
        
        return l_x + (T_MAX - t) * nn_out

    def compute_safe_velocity(current_pos, nominal_velocity, safety_margin=0.01, repulsion_gain=2.0):
        px, py, pz = current_pos
            
        pts_tensor = torch.tensor([[px, py, pz, 0.0]], dtype=torch.float32, device=DEVICE).requires_grad_(True)
        
        v_vals = []
        
        Query the network for EVERY obstacle independently
        for haz in hazards_list:
            obs_tensor = torch.tensor([[haz['x'], haz['y'], haz['z'], haz['r']]], dtype=torch.float32, device=DEVICE)
            
            # Combine state, walls, and this specific obstacle into 14 dims
            state_aug = torch.cat([pts_tensor, walls_tensor, obs_tensor], dim=1)
            
            # Get Value Function for this one obstacle
            v = model(state_aug)
            v_vals.append(v)
            
        # Take the minimum over all obstacle outputs. 
        # NN approx. of Value Function (SDF)
        V_min = torch.min(torch.stack(v_vals))
        
        # analytical gradient of V_min. 
        grads = torch.autograd.grad(outputs=V_min, inputs=pts_tensor)[0]
        
        grad_xyz = grads[0, :3].detach().cpu().numpy()
        V_val = V_min.item()
        
        safe_velocity = nominal_velocity.copy()
        # push away from closest obstacle
        if V_val < safety_margin:
            repulsive_velocity = repulsion_gain * grad_xyz
            safe_velocity += repulsive_velocity
            
        return safe_velocity