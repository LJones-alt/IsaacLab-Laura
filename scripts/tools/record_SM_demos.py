# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import argparse
import contextlib

# Third-party imports
import gymnasium as gym
import numpy as np
import os
import time
import torch

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Omniverse logger
import omni.log

if not args_cli.headless:
    import omni.ui as ui
    from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions
    from isaaclab.envs.ui import EmptyWindow


import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

if args_cli.enable_pinocchio:
    from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import GR1T2Retargeter
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab.devices.openxr.retargeters.manipulator import GripperRetargeter, Se3AbsRetargeter, Se3RelRetargeter
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from scripts.imitation_learning.robomimic.backup_controller_handler import BackupController


#rom isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
#from scripts.environments.state_machine.stack_lab_sm import PickAndLiftSm
#from scripts.environments.state_machine.weigh_lab_sm import PickAndLiftSm
#from scripts.environments.state_machine.pour_lab_sm import PickAndLiftSm
#from scripts.environments.state_machine.stack_weigh_lab_sm import PickAndLiftSm


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            if not args_cli.headless:
                env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration

def init_state_machine(env, device):
    """Create PickAndLiftSm once and capture initial TCP orientation."""

    backup_controller  = BackupController(env, device, tasktype="insert_top")
    state_guess = 0

    # Read current ee frame and capture a fixed orientation (per env)
    

    return backup_controller

def state_machine_step(env, backup_controller, fixed_tcp_rot):

    action, state_guess = backup_controller.get_controller_action(state_guess, 0)
    """Compute a single-step action from the persistent state machine (does NOT call env.step)."""
    
    return action

def reset_env_and_sm(env, backup_controller, reset_indices=None):
    """Reset specified environments (or all if None) along with the state machine.

    Args:
        env: The environment instance
        pick_sm: The state machine
        reset_indices: Optional list/1D tensor of env indices to reset. If None, resets all.
    Returns:
        fixed_tcp_rot: (num_envs, 4) tensor of fixed TCP rotations
        actions: action buffer tensor
    """
    if reset_indices is None:
        # full reset
        env.sim.reset()
        env.reset()
        reset_indices = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
    else:
        # partial reset
        env.reset(reset_indices)

    # reset state machine for these envs
    backup_controller.reset()

    # get new tcp rotation for *all* envs (keep shape consistent)
    # ee_frame = env.unwrapped.scene["ee_frame"]
    # fixed_tcp_rot = ee_frame.data.target_quat_w[..., 0, :].clone()

    # reset action buffer (neutral gripper open)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    
    return actions

def main():
    """Collect demonstrations from the environment using teleop interfaces."""

    if args_cli.headless:
        rate_limiter = None
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # prepare dataset directory
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    num_envs = 1
    success_step_counts = [0 for _ in range(num_envs)]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    # success_term = None
    # if hasattr(env_cfg.terminations, "success_term"):
    #     success_term = env_cfg.terminations.success_term
    #     env_cfg.terminations.success_term = None
    # else:
    #     omni.log.warn("No success termination term found — cannot mark demos as successful.")

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    # Set seed9
    torch.manual_seed(args_cli.seed)
    env.seed(args_cli.seed)
    # control flags
    should_reset_recording_instance = False
    running_recording_instance = True

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False

    # reset everything at start
    env.sim.reset()
    env.reset()

    current_recorded_demo_count = 0
    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."

    if not args_cli.headless:
        instruction_display = InstructionDisplay(args_cli.teleop_device)
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)
    else:
        demo_label = None
        subtask_label = None
        print(label_text)

    # init state machine + reset all envs
    pick_sm= init_state_machine(env, args_cli.device)
    actions = reset_env_and_sm(env, pick_sm)
    state_guess=0
    while simulation_app.is_running():
        if running_recording_instance:
            # standard Gymnasium step return
            actions, state_guess = pick_sm.get_controller_action(state_guess, 0)
            obs, rewards, dones, infos, _ = env.step(actions)

            # compute next actions
            #action, state_guess = pick_sm.get_controller_action(state_guess, 0)
            if dones.any():
                done_mask = dones.detach().cpu().numpy().astype(bool) if isinstance(dones, torch.Tensor) else np.asarray(dones, dtype=bool)

                # success mask from env infos (matches training!)
                if "success_term" in infos:
                    success_mask = infos["success_term"].detach().cpu().numpy().astype(bool)
                else:
                    success_mask = np.zeros(env.unwrapped.num_envs, dtype=bool)

                exported_indices = []
                for idx in range(env.unwrapped.num_envs):
                    if not done_mask[idx]:
                        continue  # only process finished envs

                    if success_mask[idx]:
                        success_step_counts[idx] += 1
                        print(f"[INFO] Env {idx} success streak: {success_step_counts[idx]}")
                    else:
                        if success_step_counts[idx] != 0:
                            print(f"[INFO] Env {idx} failed, resetting streak")
                        success_step_counts[idx] = 0

                    # reached threshold → export demo
                    if success_step_counts[idx] >= args_cli.num_success_steps:
                        print(f"[INFO] Exporting successful demo for env {idx}")
                        env.recorder_manager.record_pre_reset([idx], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [idx],
                            torch.tensor([[True]], dtype=torch.bool, device=env.device),
                        )
                        env.recorder_manager.export_episodes([idx])
                        exported_indices.append(idx)
                        success_step_counts[idx] = 0

                if exported_indices:
                    print(f"[INFO] Exported env indices: {exported_indices}")
                    pick_sm.reset()
                    fixed_tcp_rot, actions = reset_env_and_sm(
                        env, pick_sm, torch.tensor(exported_indices, device=env.device)
                    )
                else:
                    pick_sm.reset()
                    reset_indices = [idx for idx in range(env.unwrapped.num_envs) if done_mask[idx]]
                    if reset_indices:
                        print(f"[INFO] Resetting env indices: {reset_indices}")
                        fixed_tcp_rot, actions = reset_env_and_sm(
                            env, pick_sm, torch.tensor(reset_indices, device=env.device)
                        )
                        for idx in reset_indices:
                            success_step_counts[idx] = 0

            else:
                if not args_cli.headless:
                    env.sim.render()

        # update demo count label
        if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
            current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
            label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
            print(label_text)

        # stop when enough demos are recorded
        if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
            print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
            break

        if env.sim.is_stopped():
            break

        if rate_limiter:
            rate_limiter.sleep(env)


    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()