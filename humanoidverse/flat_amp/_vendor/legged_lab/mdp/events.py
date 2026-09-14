"""Project-specific domain-randomization events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import (
    _randomize_prop_by_op,
    randomize_joint_parameters as IsaacRandomizeJointParameters,
)
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class randomize_joint_parameters(IsaacRandomizeJointParameters):
    """Isaac Lab joint randomization with independent dynamic and viscous friction ranges."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        dynamic_friction_distribution_params: tuple[float, float] | None = None,
        viscous_friction_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ) -> None:
        super().__call__(
            env=env,
            env_ids=env_ids,
            asset_cfg=asset_cfg,
            friction_distribution_params=friction_distribution_params,
            armature_distribution_params=armature_distribution_params,
            lower_limit_distribution_params=lower_limit_distribution_params,
            upper_limit_distribution_params=upper_limit_distribution_params,
            operation=operation,
            distribution=distribution,
        )

        if dynamic_friction_distribution_params is None and viscous_friction_distribution_params is None:
            return

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        joint_ids: slice | Sequence[int] | torch.Tensor
        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        if dynamic_friction_distribution_params is not None:
            dynamic_friction_coeff = _randomize_prop_by_op(
                self.asset.data.default_joint_dynamic_friction_coeff.clone(),
                dynamic_friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_dynamic_friction_coefficient_to_sim(
                dynamic_friction_coeff[env_ids[:, None], joint_ids],
                joint_ids=joint_ids,
                env_ids=env_ids,
            )

        if viscous_friction_distribution_params is not None:
            viscous_friction_coeff = _randomize_prop_by_op(
                self.asset.data.default_joint_viscous_friction_coeff.clone(),
                viscous_friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_viscous_friction_coefficient_to_sim(
                viscous_friction_coeff[env_ids[:, None], joint_ids],
                joint_ids=joint_ids,
                env_ids=env_ids,
            )


class SharedBodyFrictionRandomizer(ManagerTermBase):
    """Assign one shared friction coefficient to all selected shapes per environment."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]
        if not isinstance(self.asset, Articulation):
            raise TypeError(f"SharedBodyFrictionRandomizer requires an articulation, got {type(self.asset).__name__}.")

        self.num_shapes_per_body: list[int] = []
        for link_path in self.asset.root_physx_view.link_paths[0]:
            link_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore[attr-defined]
            self.num_shapes_per_body.append(link_view.max_shapes)

        parsed_shape_count = sum(self.num_shapes_per_body)
        expected_shape_count = self.asset.root_physx_view.max_shapes
        if parsed_shape_count != expected_shape_count:
            raise RuntimeError(
                "Failed to map articulation bodies to PhysX collision shapes: "
                f"parsed {parsed_shape_count}, expected {expected_shape_count}."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        friction_range: tuple[float, float],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        del asset_cfg

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        friction = torch.empty(len(env_ids), device="cpu").uniform_(*friction_range)
        shared_friction = torch.stack((friction, friction), dim=-1)

        materials = self.asset.root_physx_view.get_material_properties()
        body_ids = self.asset_cfg.body_ids
        if isinstance(body_ids, slice):
            body_ids = list(range(self.asset.num_bodies))

        for body_id in body_ids:
            start_idx = sum(self.num_shapes_per_body[:body_id])
            end_idx = start_idx + self.num_shapes_per_body[body_id]
            materials[env_ids, start_idx:end_idx, :2] = shared_friction[:, None, :].expand(-1, end_idx - start_idx, -1)

        self.asset.root_physx_view.set_material_properties(materials, env_ids)


def randomize_joint_encoder_bias(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    bias_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample a persistent per-joint motor encoder zero offset."""

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)

    if not hasattr(env, "joint_encoder_bias"):
        raise AttributeError("The environment must allocate joint_encoder_bias before startup events are applied.")

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = torch.arange(asset.num_joints, device=asset.device, dtype=torch.long)
    else:
        joint_ids = torch.as_tensor(joint_ids, device=asset.device, dtype=torch.long)

    samples = torch.empty((len(env_ids), len(joint_ids)), device=asset.device).uniform_(*bias_range)
    env.joint_encoder_bias[env_ids[:, None], joint_ids] = samples
