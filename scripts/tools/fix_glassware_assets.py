#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to regenerate all glassware assets from their config.yaml.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import yaml
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.dict import print_dict

def load_mesh_converter_cfg_from_yaml(yaml_path: str) -> MeshConverterCfg:
    with open(yaml_path, "r") as f:
        # Some YAMLs might have typos like double colons or python/tuple tags
        content = f.read().replace("hull_vertex_limit::", "hull_vertex_limit:")
        cfg_dict = yaml.load(content, Loader=yaml.FullLoader)
    
    # Start with a default config
    cfg = MeshConverterCfg(
        asset_path=cfg_dict["asset_path"],
        usd_dir=cfg_dict["usd_dir"],
        usd_file_name=cfg_dict["usd_file_name"],
        force_usd_conversion=True,
        make_instanceable=cfg_dict.get("make_instanceable", True),
        translation=tuple(cfg_dict.get("translation", (0.0, 0.0, 0.0))),
        rotation=tuple(cfg_dict.get("rotation", (1.0, 0.0, 0.0, 0.0))),
        scale=tuple(cfg_dict.get("scale", (1.0, 1.0, 1.0))),
    )

    # Handle mass props
    if cfg_dict.get("mass_props"):
        cfg.mass_props = schemas_cfg.MassPropertiesCfg(**{k: v for k, v in cfg_dict["mass_props"].items() if v is not None})
    
    # Handle rigid props
    if cfg_dict.get("rigid_props"):
        cfg.rigid_props = schemas_cfg.RigidBodyPropertiesCfg(**{k: v for k, v in cfg_dict["rigid_props"].items() if v is not None})

    # Handle collision props
    if cfg_dict.get("collision_props"):
        cfg.collision_props = schemas_cfg.CollisionPropertiesCfg(**{k: v for k, v in cfg_dict["collision_props"].items() if v is not None})

    # Handle mesh collision props
    if cfg_dict.get("mesh_collision_props"):
        mcp_dict = cfg_dict["mesh_collision_props"]
        if "max_convex_hulls" in mcp_dict:
            cfg.mesh_collision_props = schemas_cfg.ConvexDecompositionPropertiesCfg(**{k: v for k, v in mcp_dict.items() if v is not None})
        else:
            cfg.mesh_collision_props = schemas_cfg.ConvexHullPropertiesCfg(**{k: v for k, v in mcp_dict.items() if v is not None})
    
    return cfg

def main():
    root_dir = "/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/glassware"
    
    print(f"Searching for config.yaml in {root_dir}...")
    
    for root, dirs, files in os.walk(root_dir):
        if "config.yaml" in files:
            yaml_path = os.path.join(root, "config.yaml")
            print("-" * 80)
            print(f"Processing: {yaml_path}")
            
            try:
                cfg = load_mesh_converter_cfg_from_yaml(yaml_path)
                
                # Create Mesh converter and import the file
                mesh_converter = MeshConverter(cfg)
                print(f"✓ Success generated: {mesh_converter.usd_path}")
            except Exception as e:
                print(f"✗ Failed to process {yaml_path}: {e}")
            
    print("-" * 80)
    print("Asset regeneration complete!")

if __name__ == "__main__":
    main()
    simulation_app.close()
