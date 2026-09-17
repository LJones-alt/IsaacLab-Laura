# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record hybrid (Human + State Machine) demonstrations for Isaac Lab environments into a single HDF5 dataset.
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record hybrid (Human + State Machine) demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="spacemouse", help="Device for human teleoperation (keyboard, spacemouse, gamepad).")
parser.add_argument("--dataset_file", type=str, default="docs/datasets/hybrid_dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--step_hz", type=int, default=50, help="Environment stepping rate in Hz.")
parser.add_argument("--num_human_demos", type=int, default=1, help="Number of human teleoperated demonstrations to record.")
parser.add_argument("--num_sm_demos", type=int, default=1, help="Number of State Machine controller demonstrations to record.")
parser.add_argument("--num_success_steps", type=int, default=10, help="Number of continuous steps with task success for concluding a demo.")
parser.add_argument("--gripper_orient", type=int, default=0, help="Gripper orientation mode for state machine controller.")
parser.add_argument("--enable_pinocchio", action="store_true", default=False, help="Enable Pinocchio for IK solvers.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.task is None:
    parser.error("--task is required")

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import sys
import time
from collections.abc import Callable

import gymnasium as gym
import torch

import omni.log

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.imitation_learning.robomimic.test_controller import TestController
from scripts.imitation_learning.robomimic.test_controller_top import TestController as TestControllerTop


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(output_dir: str, output_file_name: str) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        omni.log.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn("No success termination term found in environment.")

    if getattr(args_cli, "xr", False):
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # Configure recorder to save all successful episodes into the target HDF5 file
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def setup_teleop_device(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, callbacks: dict[str, Callable]) -> object:
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
        else:
            omni.log.warn(f"No teleop device '{args_cli.teleop_device}' found in env config. Creating fallback device.")
            dev_type = args_cli.teleop_device.lower()
            if dev_type == "keyboard":
                teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            elif dev_type == "spacemouse":
                teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            elif dev_type == "gamepad":
                teleop_interface = Se3Gamepad(Se3GamepadCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            else:
                omni.log.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                exit(1)

            for key, cb in callbacks.items():
                try:
                    teleop_interface.add_callback(key, cb)
                except Exception:
                    pass
    except Exception as e:
        omni.log.error(f"Failed to create teleop device: {e}")
        exit(1)

    return teleop_interface


def process_success_condition(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("[INFO] Goal reached! Demonstration exported.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def run_hybrid_collection(
    env: gym.Env,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
    success_term: object | None,
    rate_limiter: RateLimiter | None,
):
    current_human_count = 0
    current_sm_count = 0
    total_target = args_cli.num_human_demos + args_cli.num_sm_demos

    active_phase = "HUMAN" if args_cli.num_human_demos > 0 else "SM"
    success_step_count = 0
    should_reset_recording_instance = False
    running_recording_instance = True

    # Initialize State Machine controller based on gripper orientation selection
    if args_cli.gripper_orient == 0:
        sm_controller = TestControllerTop(args_cli.device, env, 0)
    else:
        sm_controller = TestController(args_cli.device, env, 1)

    # Teleoperation callbacks
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("[INFO] Reset triggered by user.")

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True
        print("[INFO] Recording active.")

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("[INFO] Recording paused.")

    callbacks = {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }

    teleop_interface = setup_teleop_device(env_cfg, callbacks)

    # Reset initial state
    env.sim.reset()
    env.reset()
    teleop_interface.reset()
    sm_controller.reset()

    print("\n" + "=" * 70)
    print("HYBRID DEMONSTRATION RECORDER INITIALIZED")
    print(f"Target Human Demos:         {args_cli.num_human_demos}")
    print(f"Target State Machine Demos: {args_cli.num_sm_demos}")
    print(f"Total Combined Demos:       {total_target}")
    print(f"Current Phase:              [{active_phase}]")
    print("=" * 70 + "\n")

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running():
            # Get action based on active collection phase
            if active_phase == "HUMAN":
                teleop_action = teleop_interface.advance()
                actions = teleop_action.repeat(env.num_envs, 1).to(env.device)
            else:
                sm_action = sm_controller.get_action()
                actions = sm_action.repeat(env.num_envs, 1).to(env.device)

            if running_recording_instance:
                obs = env.step(actions)
            else:
                env.sim.render()

            # Check task success criteria
            success_step_count, success_reset_needed = process_success_condition(
                env, success_term, success_step_count
            )
            if success_reset_needed:
                should_reset_recording_instance = True

            # Check exported demo progress
            total_exported = env.recorder_manager.exported_successful_episode_count
            if total_exported > (current_human_count + current_sm_count):
                if active_phase == "HUMAN":
                    current_human_count += 1
                    print(f"[HUMAN DEMO] Recorded {current_human_count}/{args_cli.num_human_demos}")
                    if current_human_count >= args_cli.num_human_demos:
                        active_phase = "SM"
                        print("\n" + "#" * 70)
                        print(f"[PHASE TRANSITION] Completed {current_human_count} human demos.")
                        print("Switching control to State Machine Controller for remaining demos...")
                        print("#" * 70 + "\n")
                else:
                    current_sm_count += 1
                    print(f"[SM DEMO] Recorded {current_sm_count}/{args_cli.num_sm_demos}")

            # Check if all targeted demonstrations have been collected
            if total_exported >= total_target and total_target > 0:
                print("\n" + "=" * 70)
                print(f"[COMPLETE] Recorded all {total_exported} demonstrations successfully!")
                print("=" * 70 + "\n")
                break

            # Reset environment when requested or episode ends
            if should_reset_recording_instance:
                env.sim.reset()
                env.recorder_manager.reset()
                env.reset()
                sm_controller.reset()
                teleop_interface.reset()
                success_step_count = 0
                should_reset_recording_instance = False

            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)


def main():
    rate_limiter = None if getattr(args_cli, "xr", False) else RateLimiter(args_cli.step_hz)

    output_dir, output_file_name = setup_output_directories()
    env_cfg, success_term = create_environment_config(output_dir, output_file_name)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    run_hybrid_collection(env, env_cfg, success_term, rate_limiter)

    env.close()
    print(f"[INFO] HDF5 Dataset written to: {os.path.abspath(args_cli.dataset_file)}")


if __name__ == "__main__":
    main()
    simulation_app.close()