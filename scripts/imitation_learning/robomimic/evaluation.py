# run this script: ./isaaclab.sh -p scripts/imitation_learning/robomimic//evaluation.py --checkpoint ./docs/Models/model_epoch_5000.pth

import argparse
import copy
import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

import time


# add argparse arguments
# parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
# parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
# parser.add_argument("--seed", type=int, default=101, help="Random seed.")



# args_cli = parser.parse_args()

def ensemble_uncertainty(ensemble, obs):
    start_time = time.time()
    outputs = [torch.from_numpy(policy(obs)) for policy in ensemble]

    outputs = torch.stack(outputs)
    
    mean_action = torch.mean(outputs, dim=0)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    median_action = torch.median(outputs, dim=0).values
    

    min_action = torch.min(outputs, dim=0)
    max_action = torch.max(outputs, dim=0)

    std = torch.std(outputs, dim=0)
    variance = std.pow(2)

    # std per policy (overall std across joints for each policy's action)
    per_policy_std = torch.std(outputs, dim=1) 
    min_std_index = torch.argmin(per_policy_std)
    min_std_action = outputs[min_std_index]  

    return {
        'mean': mean_action,
        'median': median_action,
        'min': min_action,
        'max': max_action,
        'std': std,
        'variance': variance,
        'min_std_action': min_std_action,
        'time_taken': time_taken
    }

def MC_dropout_uncertainty(policy, obs, niters=50):
    actions = [torch.from_numpy(policy(obs)) for i in range(niters)]
    actions = torch.stack(actions)
    
    mean = torch.mean(actions, dim=0)
    std = torch.std(actions, dim=0)
    variance = torch.sqrt(std)

    return {
        'mean': mean,
        'std': std,
        'variance': variance
    }


def mc_dropout_uncertainty_eval(policy, obs, niters=50):
    policy.policy.nets['policy'].train()


    actions = []
    with torch.no_grad():
        for _ in range(niters):
            action = torch.from_numpy(policy(obs))
            actions.append(action)
    
    actions = torch.stack(actions)

    mean = torch.mean(actions, dim=0)
    std = torch.std(actions, dim=0)
    variance = torch.var(actions, dim=0)

    policy.policy.nets['policy'].eval()

    return {
        'mean': mean,
        'std': std,
        'variance': variance
    }




def remove_dropout_layers(hooks):
    for hook in hooks:
        hook.remove()    

def inject_dropout_layers(policy, probability=0.5):
    modules = list(policy.policy.nets['policy'].named_modules())
    #print(f"Total modules found: {len(modules)}")

    def dropout_hook(module, _, output):
        #print("applying dropout")
        return F.dropout(output, p=probability, training=True) # setting training=True forces dropout to happen regardless of whether the model is in train or evaluation mode

    hooks = []
    for name, module in modules:
        if isinstance(module, nn.Linear):
            hook = module.register_forward_hook(dropout_hook)
            hooks.append(hook)
    
    return hooks

def inject_dropout_layers_for_training(model, probability=0.5):
    modules = list(model.nets['policy'].named_modules())
    #print(f"Total modules found: {len(modules)}")

    def dropout_hook(module, _, output):
        #print("applying dropout")
        return F.dropout(output, p=probability, training=True) # setting training=True forces dropout to happen regardless of whether the model is in train or evaluation mode

    hooks = []
    for name, module in modules:
        if isinstance(module, nn.Linear):
            hook = module.register_forward_hook(dropout_hook)
            hooks.append(hook)
    
    return hooks

def prepare_observations(obs_dict: dict):
    # Prepare observations
    obs = copy.deepcopy(obs_dict["policy"])
    sub_obs = copy.deepcopy(obs_dict["subtask_terms"])

    for ob in obs:
        obs[ob] = torch.squeeze(obs[ob])

    for subob in sub_obs:
        sub_obs[subob] = torch.squeeze(sub_obs[subob])
    
    return obs, sub_obs


# device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)


# obs_dict = {'policy': {
#             'joint_pos': torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.]], device=device), 
#             'joint_vel': torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.]], device=device), 
#             'object_position': torch.tensor([[0.4975, 0.0483, 0.0203]], device=device), 
#             'target_object_position': torch.tensor([[ 0.4000, -0.3000,  0.0800,  1.0000,  0.0000,  0.0000,  0.0000]], device=device), 
#             'actions': torch.tensor([[0., 0., 0., 0., 0., 0., 0.]], device=device), 
#             'object_to_target': torch.tensor([False], device=device), 
#             'eef_pos': torch.tensor([[ 0.3764, -0.0050,  0.0891]], device=device), 
#             'eef_quat': torch.tensor([[ 0.6433,  0.0366,  0.7636, -0.0415]], device=device), 
#             'gripper_pos': torch.tensor([[ 0.0400, -0.0400]], device=device)
#         }, 
#     'subtask_terms': {
#             'appr': torch.tensor([False], device=device), 
#             'grasp': torch.tensor([False], device=device), 
#             'lift': torch.tensor([False], device=device), 
#             'appr_goal': torch.tensor([False], device=device)
#         }
#     }

# inject_dropout_layers(policy=policy, probability=0.7)

# policy.start_episode()
# obs, sub_obs = prepare_observations(obs_dict=obs_dict)


# start_time = time.time()
# results_dict = MC_dropout_uncertainty(policy=policy, obs=obs, niters=100)
# end_time = time.time()

# print(results_dict)
# print(f"time taken: {end_time - start_time}")

# ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
# --device cuda --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --num_rollouts 50 \
# --checkpoint /logs/docs/Models/bc/model1/Isaac-Stack-Cube-Franka-IK-Rel-v0/bc_rnn_low_dim_franka_stack/20250715152224/models/model_epoch_2000.pth


"""
train a model that:

takes as input joint positions (so a combination of joint positions and their values).

outputs whether that specific combination of joint positions is safe or not.

for the training data use joint position input of the current policy and see if it resulted in task success or not.

"""