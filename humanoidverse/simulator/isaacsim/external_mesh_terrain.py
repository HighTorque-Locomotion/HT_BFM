"""Isaac Lab sub-terrain adapter for motion-matched STL meshes."""

from dataclasses import MISSING
from pathlib import Path

import numpy as np
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def external_stl_terrain(_difficulty: float, cfg: "ExternalStlTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    mesh_path = Path(cfg.mesh_path).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Motion-matched terrain mesh does not exist: {mesh_path}")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = mesh.copy()
    bounds = mesh.bounds
    mesh_size = bounds[1, :2] - bounds[0, :2]
    if np.any(mesh_size > np.asarray(cfg.size) + 1e-5):
        raise ValueError(f"{mesh_path} bounds {mesh_size.tolist()} exceed terrain tile size {cfg.size}.")

    # Isaac Lab sub-terrains use [0, size] coordinates. Generated motion-matched
    # meshes use a motion-local frame centered at (0, 0), so shift the mesh while
    # retaining the tile center as the motion/reset origin.
    center_xy = 0.5 * np.asarray(cfg.size)
    mesh.apply_translation((center_xy[0], center_xy[1], 0.0))
    origin = np.array((center_xy[0], center_xy[1], cfg.origin_z), dtype=np.float64)
    return [mesh], origin


@configclass
class ExternalStlTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a checked, motion-local STL terrain."""

    function = external_stl_terrain
    mesh_path: str = MISSING
    origin_z: float = 0.0
