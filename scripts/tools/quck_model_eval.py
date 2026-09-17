import os
import torch
import sys
from isaaclab.safety.safety_barrier import JP_APF
from isaaclab.safety import Panda7DOF_DH

print("TESTING MODEL")
test_obst = [-0.6, -0.1, 0.8, 0.0]

joint_apf=JP_APF(device=torch.device("cuda:0"),obst=test_obst)
# print("Model's state_dict:")
# for param_tensor in joint_apf.model.state_dict():
#     print(param_tensor, "\t", joint_apf.model.state_dict()[param_tensor].size())

joint_apf.model.eval()


q_dot = torch.tensor([ -0.0247,  0.0147,  0.0154, -0.0559, -0.0533,  0.0228,  0.0526, 0.0, 0.0], device='cuda:0').requires_grad_(True)
q_dot=q_dot.unsqueeze(0).requires_grad_(True)
joint_pos = torch.tensor([ 0.2808, -0.3024, -0.2034, -2.9972, -0.8395,  2.7396,  1.7181], device='cuda:0').requires_grad_(True)
#print(f"Joint position: {joint_pos}")
with torch.enable_grad():
    safety_val, joint_vels=joint_apf.get_joint_vals(joint_pos,q_dot, 0.0)
    print(f"Joint velocities_t0: {joint_vels[0].cpu().squeeze().tolist()}")
    safety_val, joint_vels=joint_apf.get_joint_vals(joint_pos,q_dot, 0.1)
    print(f"Joint velocities_t_max: {joint_vels[0].cpu().squeeze().tolist()}")

# nasty little data recording 
# with open("docs/apf/joint_pos_vals.csv", "a") as f:
#     # get_joint_vals now returns (safety_val, grads_tuple)
#     # Extract the gradient tensor from the tuple
#     if isinstance(joint_vels, tuple):
#         joint_vels_tensor = joint_vels[0]
#     else:
#         joint_vels_tensor = joint_vels
#     # Ensure tensors are 1-D for easy indexing
#     joint_pos_flat = joint_pos.squeeze()
#     joint_vels_flat = joint_vels_tensor.squeeze()
#     safety_val_flat = safety_val.squeeze()
#     # Write each joint position, its corresponding velocity, and the safety value
#     for i in range(joint_pos_flat.shape[0]):
#         f.write(f"{joint_pos_flat[i].item()}, {joint_vels_flat[i].item()}, ")
#     f.write(f"{safety_val_flat.item()}")
#     f.write("\n")

# with torch.enable_grad():
#     q_clone = joint_pos.clone()
#     outputs_clone = joint_vels.clone() 
            
#     print(f"[DEBUG] : inout q_clone : {q_clone}, shape : {q_clone.shape}")
#     print(f"[DEBUG] : output output_clone : {outputs_clone}, shape : {outputs_clone.shape}")
    
#     grads = torch.autograd.grad(
#         outputs=joint_vels,
#         inputs=joint_pos,
#         grad_outputs=torch.ones_like(joint_vels),
#         create_graph=False,
#         allow_unused=False
#     )
#     grads = grads[0]
# print(f"[DEBUG] gradient of value: {grads}")
# print(f"type {type(grads)}")
        





