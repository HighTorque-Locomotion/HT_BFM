"""Read-only signal adapters for the IsaacLab playback window (no Kit imports)."""

import torch


class PlaySignals:
    """Expose policy inputs and physical transitions without recomputing observations.

    Sampling happens once per control step, before automatic resets. In particular,
    reading the panel must never append observation history or advance commands.
    """

    def __init__(self, env):
        self.env = env
        self.raw_actions = torch.zeros_like(env.action)
        self.joint_targets = env.robot.data.default_joint_pos.clone()

    def terms(self, category, env_idx):
        """Return (name, values, legends) for one environment only."""
        if not 0 <= env_idx < self.env.num_envs:
            raise ValueError(f"Environment index out of range: {env_idx}")
        return [
            (name, value.detach().flatten().cpu().tolist(), legends)
            for name, value, legends in getattr(self, category)(env_idx)
        ]

    def actions(self, i):
        env = self.env
        joints = list(env.robot.joint_names)
        yield "policy_actions", self.raw_actions[i], joints
        yield "applied_actions", env.action[i], joints
        yield "joint_position_target (rad)", self.joint_targets[i], joints
        yield "applied_torque (Nm)", env.robot.data.applied_torque[i], joints

    def observations(self, i):
        env = self.env
        obs = env.extras["observations"]
        joints = list(env.robot.joint_names)
        layout = [
            ("base_ang_vel", ["x", "y", "z"]),
            ("projected_gravity", ["x", "y", "z"]),
            ("velocity_commands", ["vx", "vy", "yaw"]),
            ("joint_pos", joints),
            ("joint_vel", joints),
            ("last_action", joints),
            ("phase", ["sin", "cos"]),
        ]
        # This is the exact current frame used by act_inference, including scaling/noise.
        actor = obs["current_actor_obs_add_nosie"][i]
        offset = 0
        for name, legends in layout:
            yield f"policy/{name}", actor[offset:offset + len(legends)], legends
            offset += len(legends)
        if offset != actor.numel():
            raise ValueError(f"Unsupported flat AMP observation layout: {actor.numel()} (expected {offset})")
        critic = obs["current_critic_obs"][i]
        yield "critic/base_lin_vel", critic[offset:offset + 3], ["x", "y", "z"]
        feet_names = [env.contact_sensor.body_names[j] for j in env.feet_cfg.body_ids]
        yield "critic/feet_contact", critic[offset + 3:], feet_names

    def commands(self, i):
        env = self.env
        yield "base_velocity", env.command_generator.command[i], ["vx", "vy", "yaw"]
        yield "measured_linear_velocity (m/s)", env.robot.data.root_lin_vel_b[i], ["x", "y", "z"]
        yield "measured_angular_velocity (rad/s)", env.robot.data.root_ang_vel_b[i], ["x", "y", "z"]

    def rewards(self, i):
        for name, values in self.env.reward_manager.get_active_iterable_terms(i):
            yield name, torch.as_tensor(values), [name]

    def termination(self, i):
        for name, values in self.env.reset_reason_buf.items():
            yield name, values[i], [name]

    def curriculum(self, i):
        terrain = self.env.scene.terrain
        if terrain is not None and terrain.cfg.terrain_type == "generator":
            yield "terrain_level", terrain.terrain_levels[i], ["level"]

    def contact_forces(self, i):
        for name in ("left_foot_force", "right_foot_force"):
            sensor = self.env.scene.sensors[name]
            # Ground-filtered normal + tangential force, matching HT_lab's panel.
            # Sanitize each collision pair before summing so NaNs do not discard
            # a valid force from another collision pair.
            normal = torch.nan_to_num(sensor.data.force_matrix_w[i, 0], nan=0.0).sum(dim=0)
            friction = torch.nan_to_num(sensor.data.friction_forces_w[i, 0], nan=0.0).sum(dim=0)
            yield f"{sensor.body_names[0]} (N, world)", normal + friction, ["Fx", "Fy", "Fz"]
