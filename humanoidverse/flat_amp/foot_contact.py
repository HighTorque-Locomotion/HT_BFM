"""Sample persistent per-environment foot compliance and bind it before physics starts."""

import math

import numpy as np


def sample_foot_contact_parameters(num_envs, cfg, *, seed, stiffness=0.0, damping=0.0):
    """Return [stiffness, damping] per environment and group IDs (rigid/reproduction/random/fixed).

    Group membership is sampled independently with the configured probabilities.
    A local generator keeps sampling reproducible without perturbing other randomization.
    """
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if not cfg.enabled:
        if not all(math.isfinite(v) and v >= 0 for v in (stiffness, damping)):
            raise ValueError("Fixed foot contact stiffness/damping must be finite and non-negative")
        return np.tile([stiffness, damping], (num_envs, 1)), np.full(num_envs, 3, dtype=np.int64)

    probabilities = np.array([cfg.rigid_fraction, cfg.reproduction_fraction], dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or probabilities.sum() > 1:
        raise ValueError("Foot contact group fractions must be non-negative and sum to at most one")
    for name, bounds, positive in (
        ("stiffness_range", cfg.stiffness_range, True),
        ("damping_ratio_range", cfg.damping_ratio_range, False),
    ):
        if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds):
            raise ValueError(f"Invalid foot contact {name}")
        low, high = bounds
        if low > high or low < 0 or (positive and low == 0):
            raise ValueError(f"Invalid foot contact {name}")
    if not math.isfinite(cfg.reference_mass) or cfg.reference_mass <= 0:
        raise ValueError("Foot contact reference_mass must be positive and finite")
    if not math.isfinite(cfg.reproduction_stiffness) or cfg.reproduction_stiffness <= 0:
        raise ValueError("Foot contact reproduction_stiffness must be positive and finite")

    rng = np.random.default_rng(seed)
    groups = rng.choice(3, size=num_envs, p=[*probabilities, 1.0 - probabilities.sum()])
    parameters = np.zeros((num_envs, 2), dtype=np.float64)
    parameters[groups == 1, 0] = cfg.reproduction_stiffness
    random_mask = groups == 2
    count = int(random_mask.sum())
    k = np.exp(rng.uniform(*np.log(cfg.stiffness_range), size=count))
    zeta = rng.uniform(*cfg.damping_ratio_range, size=count)
    parameters[random_mask, 0] = k
    parameters[random_mask, 1] = 2.0 * zeta * np.sqrt(cfg.reference_mass * k)
    return parameters, groups


def bind_foot_contact_materials(stage, env_prim_paths, parameters):
    """Give each environment its own material, shared by its two feet only.

    Call after heterogeneous scene creation (replicate_physics=False), before sim.reset().
    Parent binding overrides collider descendants, including instanced collision geometry.
    """
    import isaaclab.sim as sim_utils
    from pxr import UsdShade

    if parameters.shape != (len(env_prim_paths), 2):
        raise ValueError("Expected one stiffness/damping pair per environment")
    for env_path, (stiffness, damping) in zip(env_prim_paths, parameters):
        material_path = f"{env_path}/FootContactMaterial"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            compliant_contact_stiffness=float(stiffness),
            compliant_contact_damping=float(damping),
        )
        material_cfg.func(material_path, material_cfg)
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        for foot_name in ("l_ankle_roll_link", "r_ankle_roll_link"):
            foot_path = f"{env_path}/Robot/{foot_name}"
            foot_prim = stage.GetPrimAtPath(foot_path)
            if not foot_prim.IsValid():
                raise RuntimeError(f"Missing foot rigid body: {foot_path}")
            UsdShade.MaterialBindingAPI.Apply(foot_prim).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
