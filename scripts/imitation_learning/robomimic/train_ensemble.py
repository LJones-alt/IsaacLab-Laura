# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MIT License
#
# Copyright (c) 2021 Stanford Vision and Learning Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
The main entry point for training policies from pre-collected data.

This script loads dataset(s), creates a model based on the algorithm specified,
and trains the model. It supports training on various environments with multiple
algorithms from robomimic.

Args:
    algo: Name of the algorithm to run.
    task: Name of the environment.
    name: If provided, override the experiment name defined in the config.
    dataset: If provided, override the dataset path defined in the config.
    log_dir: Directory to save logs.
    normalize_training_actions: Whether to normalize actions in the training data.

This file has been modified from the original robomimic version to integrate with IsaacLab.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Standard library imports
import argparse

# Third-party imports
import gymnasium as gym
import h5py
import json
import numpy as np
import os
import shutil
import sys
import time
import torch
import traceback
import subprocess
import wandb

from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import psutil

# Robomimic imports
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import algo_factory
from robomimic.config import Config, config_factory
from robomimic.utils.log_utils import DataLogger, PrintLogger

# Isaac Lab imports (needed so that environment is registered)
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from evaluation import inject_dropout_layers_for_training, remove_dropout_layers

def normalize_hdf5_actions(config: Config, log_dir: str) -> str:
    """Normalizes actions in hdf5 dataset to [-1, 1] range.

    Args:
        config: The configuration object containing dataset path.
        log_dir: Directory to save normalization parameters.

    Returns:
        Path to the normalized dataset.
    """
    base, ext = os.path.splitext(config.train.data)
    normalized_path = base + "_normalized" + ext

    # Copy the original dataset
    print(f"Creating normalized dataset at {normalized_path}")
    shutil.copyfile(config.train.data, normalized_path)

    # Open the new dataset and normalize the actions
    with h5py.File(normalized_path, "r+") as f:
        dataset_paths = [f"/data/demo_{str(i)}/actions" for i in range(len(f["data"].keys()))]

        # Compute the min and max of the dataset
        dataset = np.array(f[dataset_paths[0]]).flatten()
        for i, path in enumerate(dataset_paths):
            if i != 0:
                data = np.array(f[path]).flatten()
                dataset = np.append(dataset, data)

        max = np.max(dataset)
        min = np.min(dataset)

        # Normalize the actions
        for i, path in enumerate(dataset_paths):
            data = np.array(f[path])
            normalized_data = 2 * ((data - min) / (max - min)) - 1  # Scale to [-1, 1] range
            del f[path]
            f[path] = normalized_data

        # Save the min and max values to log directory
        with open(os.path.join(log_dir, "normalization_params.txt"), "w") as f:
            f.write(f"min: {min}\n")
            f.write(f"max: {max}\n")

    return normalized_path


#

def train(config: Config, device: str, log_dirs: list[str], ckpt_dirs: list[str], video_dirs: list[str], use_config_seed: bool = True, ensemble_size: int = 1):
    """Train an ensemble of models using the algorithm specified in config. If ensemble_size is not specified, a single model will be trained.

    Args:
        config: Configuration object.
        device: PyTorch device to use for training.
        log_dirs: List of directories to save logs for each network in the ensemble.
        ckpt_dirs: List of directories to save checkpoints for each network in the ensemble.
        video_dirs: List of directories to save videos for each network in the ensemble.
    """
    # first set seeds
    if use_config_seed:
        np.random.seed(config.train.seed)
        torch.manual_seed(config.train.seed)
    

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")


    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset at provided path {dataset_path} not found!")

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data, all_obs_keys=config.all_obs_keys, verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            envs[env.name] = env
            print(envs[env.name])

    print("")


    # load training data
    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    
    # reduce size of the dataset
    trainset = keep_data_percentage(trainset, args.dataset_percentage)

    #train_sampler = trainset.get_dataset_sampler()
    if isinstance(trainset, Subset):
        train_sampler = trainset.dataset.get_dataset_sampler()
    else:
        train_sampler = trainset.get_dataset_sampler()


    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # maybe retrieve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        valid_loader = None

    # main training loop
    #best_valid_loss = None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    #config.algo.transformer.num_layers = 1
    #config.algo.transformer.num_heads = 1
    for model_num in range(ensemble_size):
        # setup a wand run
        wand_experiment_name = f"{config.experiment.name}-model{model_num}"
        run=wandb.init(
            project=config.experiment.logging.wandb_proj_name,
            name=wand_experiment_name
        )
        #run.define_metric("train reward", step_metric="episode")
       # run.define_metric("episode error", step_metric="episode")
        # reset best valid loss
        best_valid_loss = None

        log_dir = log_dirs[model_num]
        ckpt_dir = ckpt_dirs[model_num]
        video_dir = video_dirs[model_num]

        # setup logger stuff
        # logger = Logger(os.path.join(log_dir, "log.txt"))
        # sys.stdout = logger
        # sys.stderr = logger
        data_logger = DataLogger(log_dir, config=config, log_tb=config.experiment.logging.log_tb, log_wandb=config.experiment.logging.log_wandb)
        with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
            json.dump(config, outfile, indent=4)

        # initialise current network in the ensemble
        model = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=shape_meta["all_shapes"],
            ac_dim=shape_meta["ac_dim"],
            device=device,
        )
        print("\n============= Model Summary =============")
        print(model)  # print model summary
        print("")
        
        for epoch in range(1, config.train.num_epochs + 1):  # epoch numbers start at 1
            step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps)
            model.on_epoch_end(epoch)

            # setup checkpoint path
            epoch_ckpt_name = f"model_epoch_{epoch}"

            # check for recurring checkpoint saving conditions
            should_save_ckpt = False
            if config.experiment.save.enabled:
                time_check = (config.experiment.save.every_n_seconds is not None) and (
                    time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
                )
                epoch_check = (
                    (config.experiment.save.every_n_epochs is not None)
                    and (epoch > 0)
                    and (epoch % config.experiment.save.every_n_epochs == 0)
                )
                epoch_list_check = epoch in config.experiment.save.epochs
                should_save_ckpt = time_check or epoch_check or epoch_list_check
            ckpt_reason = None
            if should_save_ckpt:
                last_ckpt_time = time.time()
                ckpt_reason = "time"

            print(f"Train Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Train/{k}", v, epoch)

            # Evaluate the model on validation set
            if config.experiment.validate:
                with torch.no_grad():
                    step_log = TrainUtils.run_epoch(
                        model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps
                    )
                for k, v in step_log.items():
                    if k.startswith("Time_"):
                        data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                    else:
                        data_logger.record(f"Valid/{k}", v, epoch)

                print(f"Validation Epoch {epoch}")
                print(json.dumps(step_log, sort_keys=True, indent=4))

                # save checkpoint if achieve new best validation loss
                valid_check = "Loss" in step_log
                if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                    best_valid_loss = step_log["Loss"]
                    if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                        epoch_ckpt_name += f"_best_validation_{best_valid_loss}"
                        should_save_ckpt = True
                        ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

            # Save model checkpoints based on conditions (success rate, validation loss, etc)

            if should_save_ckpt:
                TrainUtils.save_model(
                    model=model,
                    config=config,
                    env_meta=env_meta,
                    shape_meta=shape_meta,
                    ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                    obs_normalization_stats=obs_normalization_stats,
                )

            # Finally, log memory usage in MB
            process = psutil.Process(os.getpid())
            mem_usage = int(process.memory_info().rss / 1000000)
            data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
            print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")

        #config.algo.transformer.num_layers = config.algo.transformer.num_layers + 1
        #config.algo.transformer.num_heads = config.algo.transformer.num_heads + 1
       # run.finish()
    # terminate logging
    data_logger.close()

def keep_data_percentage(trainset, percentage):
    og_len = len(trainset)
    # keep only x% of the samples 
    total_size = len(trainset)
    num_samples = int(total_size * (percentage))
    indices = np.random.choice(total_size, num_samples, replace=False)
    
    trainset = Subset(trainset, indices)
    
    print(f"size of trainset after reduction is {len(trainset)}")
    
    new_len = len(trainset)
    retained_percent = (new_len / og_len) * 100
    
    print(f"Retained: {retained_percent:.2f}% of original dataset")

    return trainset



def main(args: argparse.Namespace):
    """Train a model on a task using a specified algorithm.

    Args:
        args: Command line arguments.
    """
    # load config
    if args.task is not None:
        # obtain the configuration entry point
        cfg_entry_point_key = f"robomimic_{args.algo}_cfg_entry_point"

        print(f"Loading configuration for task: {args.task}")
        print(gym.envs.registry.keys())
        print(" ")
        cfg_entry_point_file = gym.spec(args.task).kwargs.pop(cfg_entry_point_key)
        # check if entry point exists
        if cfg_entry_point_file is None:
            raise ValueError(
                f"Could not find configuration for the environment: '{args.task}'."
                f" Please check that the gym registry has the entry point: '{cfg_entry_point_key}'."
            )

        with open(cfg_entry_point_file) as f:
            ext_cfg = json.load(f)
            config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        raise ValueError("Please provide a task name through CLI arguments.")

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name
    # little fix for WANDB
    for stream in (sys.stdout, sys.stderr):
        if not hasattr(stream, "isatty"):
            stream.isatty = lambda: False
    os.environ["WANDB_MODE"] = "offline"
    run=wandb.init(
        project=config.experiment.logging.wandb_proj_name,
        name=config.experiment.name
    )
    # run.define_metric("train reward", step_metric="episode")
    # run.define_metric("episode error", step_metric="episode")

    # change location of experiment directory
    config.train.output_dir = os.path.abspath(os.path.join("./logs", args.log_dir, args.task))
    original_output_dir = config.train.output_dir
    
    log_dirs, ckpt_dirs, video_dirs = [], [], []
    
    for i in range(args.ensemble_size):
        new_output_dir = f"{original_output_dir}/model{i}/"
        config.train.output_dir = new_output_dir
        log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

        log_dirs.append(log_dir)
        ckpt_dirs.append(ckpt_dir)
        video_dirs.append(video_dir)

    print("=============== Ensemble Paths =============== ")
    print(f"log_dirs: {log_dirs}")
    print(f"ckpt_dirs: {ckpt_dirs}")
    print(f"video_dirs: {video_dirs}")

    if args.normalize_training_actions:
        config.train.data = normalize_hdf5_actions(config, log_dir)

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        #train(config, device, log_dir, ckpt_dir, video_dir)
        train(
            config=config, 
            device=device, 
            log_dirs=log_dirs, ckpt_dirs=ckpt_dirs, video_dirs=video_dirs, 
            use_config_seed=True, 
            ensemble_size=args.ensemble_size
        )
        
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--algo", type=str, default=None, help="Name of the algorithm.")
    parser.add_argument("--log_dir", type=str, default="robomimic", help="Path to log directory")
    parser.add_argument("--normalize_training_actions", action="store_true", default=False, help="Normalize actions")
    
    parser.add_argument("--dataset_percentage", type=float, default=1.0, help="Percentage of samples from the dataset to use")
    parser.add_argument("--ensemble_size", type=int, default=15, help="Defines the number of networks in the ensemble.")
    
    args = parser.parse_args()

    # run training
    main(args)
    # close sim app
    simulation_app.close()



    # ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --algo bc --dataset ./docs/training_data/generated_dataset_split.hdf5 --log_dir ./logs/docs/Models/bc/bcc_rnn_gmm_stack_cube_model1/