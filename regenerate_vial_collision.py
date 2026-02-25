#!/usr/bin/env python3
"""Script to regenerate USD files for vial rack and capped vial with proper collision meshes."""

from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.dict import print_dict

def regenerate_vial_rack():
    """Regenerate vial rack USD with convex decomposition."""
    print("=" * 80)
    print("Regenerating Vial Rack USD with Convex Decomposition")
    print("=" * 80)
    
    mesh_converter_cfg = MeshConverterCfg(
        asset_path="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/glassware/vial_rack/vial_rack.obj",
        usd_dir="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/glassware/vial_rack",
        usd_file_name="vial_rack.usd",
        force_usd_conversion=True,
        make_instanceable=True,
        mass_props=schemas_cfg.MassPropertiesCfg(mass=0.1),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
        mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(
            max_convex_hulls=64,
            voxel_resolution=1000000,
            hull_vertex_limit=64,
            min_thickness=0.0005
        ),
        translation=(0.0, 0.0, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )
    
    print("Config:")
    print_dict(mesh_converter_cfg.to_dict(), nesting=0)
    
    mesh_converter = MeshConverter(mesh_converter_cfg)
    print(f"✓ Generated: {mesh_converter.usd_path}")
    print()

def regenerate_capped_vial():
    """Regenerate capped vial USD with convex hull."""
    print("=" * 80)
    print("Regenerating Capped Vial USD with Convex Hull")
    print("=" * 80)
    
    mesh_converter_cfg = MeshConverterCfg(
        asset_path="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/glassware/vial_with_cap/vial_with_cap.obj",
        usd_dir="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/glassware/vial_with_cap",
        usd_file_name="vial_with_cap.usd",
        force_usd_conversion=True,
        make_instanceable=True,
        mass_props=schemas_cfg.MassPropertiesCfg(mass=0.01),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
        mesh_collision_props=schemas_cfg.ConvexHullPropertiesCfg(
            hull_vertex_limit=32,
            min_thickness=0.0005
        ),
        translation=(0.0, 0.0, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )
    
    print("Config:")
    print_dict(mesh_converter_cfg.to_dict(), nesting=0)
    
    mesh_converter = MeshConverter(mesh_converter_cfg)
    print(f"✓ Generated: {mesh_converter.usd_path}")
    print()

if __name__ == "__main__":
    regenerate_vial_rack()
    regenerate_capped_vial()
    print("=" * 80)
    print("✓ All USD files regenerated successfully!")
    print("=" * 80)
    simulation_app.close()
