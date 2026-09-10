import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from humanoidverse.amp_stage2 import (
    MIMICLITE_LOCOMOTION_WEIGHTS,
    AMPDiscriminator,
    CommandEncoderPolicy,
    LoRALinear,
    MimicLiteLocomotionRewardState,
    _command_category_ids,
    _hydra_robot_name,
    _load_piplus_robot_contract,
    _motion_qpos,
    _policy_dof_from_motion,
    _resample_done_command_targets,
    _sample_commands,
    _stage2_prepare_pre_reset_transition,
    _stage2_update_reset_buf,
    _torchrun_command,
    _transition_core,
    _validate_stage1_robot_contract,
    baseline_normalized_linvel_reward,
    bfm_lora_parameters,
    command_tracking_metrics,
    compute_gae,
    encoder_input_scale,
    flatten_encoder_observation,
    inject_bfm_actor_lora,
    load_command_encoder_policy_state,
    ppo_update,
    project_latent,
    restore_discriminator_optimizer,
    restore_policy_optimizer,
)
from humanoidverse.amp_stage2_amp018_speedtrack_nofilter_20260909 import (
    FEET_DIAGNOSTIC_KEYS as AMP018_FEET_DIAGNOSTIC_KEYS,
)
from humanoidverse.amp_stage2_amp018_speedtrack_nofilter_20260909 import (
    FOOT_MIN_SEPARATION as AMP018_FOOT_MIN_SEPARATION,
)
from humanoidverse.amp_stage2_amp018_speedtrack_nofilter_20260909 import (
    MimicLiteLocomotionRewardState as Amp018MimicLiteLocomotionRewardState,
)
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase


class AmpStage2Test(unittest.TestCase):
    def test_h0w_robot_contract_matches_stage1_checkpoint(self):
        contract = _load_piplus_robot_contract(
            "humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H0W.yaml"
        )
        validated = _validate_stage1_robot_contract(
            contract,
            "huiying/bfmzero-piplus-h0w-isaac-20260730_102224(1)/checkpoint",
        )
        self.assertEqual(_hydra_robot_name("PiPlus_S_12L8A0G2H0W"), "robot=piplus/PiPlus_S_12L8A0G2H0W")
        self.assertTrue(validated["validated"])
        self.assertEqual(contract.robot_type, "PiPlus_S_12L8A0G2H0W")
        self.assertEqual(len(contract.policy_joint_names), 22)

    def test_lora_linear_starts_as_frozen_base_and_has_trainable_residual(self):
        base = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        expected = base(x).detach()
        layer = LoRALinear(base, rank=2, alpha=4.0)
        actual = layer(x)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertFalse(layer.base.weight.requires_grad)
        self.assertTrue(layer.lora_A.requires_grad)
        self.assertTrue(layer.lora_B.requires_grad)

    def test_inject_bfm_actor_lora_replaces_standard_linears(self):
        actor = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
        count = inject_bfm_actor_lora(actor, rank=2, alpha=4.0)
        self.assertEqual(count, 2)
        parameters = bfm_lora_parameters(actor)
        self.assertEqual(len(parameters), 4)
        self.assertTrue(all(parameter.requires_grad for parameter in parameters))
        self.assertTrue(all(not parameter.requires_grad for name, parameter in actor.named_parameters() if "lora_" not in name))

    def test_sample_commands_lateral_only_probability(self):
        commands = _sample_commands(
            512,
            torch.device("cpu"),
            torch.tensor([-0.8, -0.5, -0.8]),
            torch.tensor([0.8, 0.5, 0.8]),
            stand_prob=0.0,
            turn_prob=0.0,
            lateral_prob=1.0,
        )
        self.assertTrue(torch.allclose(commands[:, 0], torch.zeros(512)))
        self.assertTrue(torch.allclose(commands[:, 2], torch.zeros(512)))
        self.assertTrue(torch.all(commands[:, 1].abs() <= 0.5))
        self.assertGreater(float(commands[:, 1].abs().mean()), 0.1)

    def test_command_tracking_metrics_report_lateral_response(self):
        commands = torch.tensor(
            [[0.0, -0.4, 0.0], [0.0, -0.2, 0.0], [0.0, 0.2, 0.0], [0.0, 0.4, 0.0]],
            dtype=torch.float32,
        )
        achieved = commands.clone()
        metrics = command_tracking_metrics(commands, achieved, torch.zeros_like(achieved))
        self.assertAlmostEqual(metrics["tracking/nonzero_vy_response_slope"], 1.0, places=5)
        self.assertAlmostEqual(metrics["tracking/nonzero_vy_command_correlation"], 1.0, places=5)
        self.assertAlmostEqual(metrics["tracking/vy_bin_left_mae"], 0.0, places=5)
        self.assertAlmostEqual(metrics["tracking/vy_bin_right_mae"], 0.0, places=5)

    def test_amp_command_categories(self):
        commands = torch.tensor(
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.4]]
        )
        self.assertTrue(torch.equal(_command_category_ids(commands), torch.tensor([0, 1, 2, 3, 4])))

    def test_piplus_robot_and_motion_joint_contract(self):
        contract = _load_piplus_robot_contract(
            "humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H1W_LSE.yaml"
        )
        joint_names = contract.policy_joint_names
        frame = np.arange(len(joint_names), dtype=np.float64)
        motion = {
            "root_trans_offset": np.zeros((1, 3), dtype=np.float64),
            "root_rot": np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            "dof": frame[None],
            "joint_names": list(reversed(joint_names)),
        }
        motion["dof"] = motion["dof"][:, ::-1]
        policy_dof = _policy_dof_from_motion(motion, joint_names)
        model = mujoco.MjModel.from_xml_path(str(contract.robot.xml_path))
        qpos = _motion_qpos(motion, joint_names, policy_dof, 0, model)

        self.assertEqual(len(joint_names), 23)
        self.assertEqual(policy_dof.shape, (1, 23))
        for index, joint_name in enumerate(joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.assertEqual(qpos[int(model.jnt_qposadr[joint_id])], float(index))

    def test_command_encoder_shapes_and_latent_projection(self):
        policy = CommandEncoderPolicy(input_dim=19, z_dim=8, hidden_dim=32)
        raw_z, log_prob, value = policy.sample(torch.randn(5, 19))

        self.assertEqual(raw_z.shape, (5, 8))
        self.assertEqual(log_prob.shape, (5,))
        self.assertEqual(value.shape, (5,))
        projected = project_latent(raw_z)
        self.assertTrue(torch.allclose(projected.norm(dim=-1), torch.full((5,), 8.0**0.5), atol=1.0e-5))

    def test_ppo_update_early_stops_and_reports_averaged_metrics(self):
        torch.manual_seed(3)
        policy = CommandEncoderPolicy(input_dim=5, z_dim=4, hidden_dim=16)
        features = torch.randn(8, 5)
        raw_z, log_prob, values = policy.sample(features)
        rollout = SimpleNamespace(
            encoder_features=features.reshape(2, 4, 5),
            raw_z=raw_z.reshape(2, 4, 4),
            old_log_prob=(log_prob + 1.0).reshape(2, 4),
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-4)
        metrics = ppo_update(
            policy,
            rollout,
            torch.ones(2, 4),
            values.detach().reshape(2, 4),
            epochs=4,
            minibatch_size=8,
            clip_ratio=0.2,
            value_coef=0.5,
            entropy_coef=0.001,
            optimizer=optimizer,
            max_grad_norm=1.0,
            target_kl=0.01,
        )
        self.assertEqual(metrics["ppo_updates"], 1.0)
        self.assertEqual(metrics["ppo_expected_updates"], 4.0)
        self.assertEqual(metrics["ppo_update_fraction"], 0.25)
        self.assertEqual(metrics["ppo_early_stop"], 1.0)
        self.assertTrue(np.isfinite(metrics["ratio_max"]))
        self.assertGreaterEqual(metrics["clip_fraction"], 0.0)

    def test_encoder_input_scaling_and_legacy_policy_migration_are_equivalent(self):
        torch.manual_seed(4)
        obs = {
            "state": torch.randn(3, 10),
            "last_action": torch.randn(3, 2),
            "history_actor": torch.randn(3, 48),
        }
        commands = torch.randn(3, 3)
        raw_features = torch.cat((commands, obs["state"], obs["last_action"], obs["history_actor"]), dim=-1)
        scaled_features = flatten_encoder_observation(obs, commands)
        scale = encoder_input_scale(obs, commands)
        self.assertTrue(torch.allclose(scaled_features, raw_features * scale))
        self.assertTrue(torch.equal(scale[5:7], torch.full((2,), 0.05)))
        self.assertEqual(int((scale == 0.05).sum()), 10)

        lateral_scaled = encoder_input_scale(obs, commands, command_scale=(1.25, 2.5, 1.25))
        self.assertAlmostEqual(float(lateral_scaled[0]), 1.25)
        self.assertAlmostEqual(float(lateral_scaled[1]), 2.5)
        custom_features = flatten_encoder_observation(obs, commands, command_scale=(1.25, 2.5, 1.25))
        self.assertTrue(torch.allclose(custom_features, raw_features * lateral_scaled))

        legacy_policy = CommandEncoderPolicy(raw_features.shape[-1], z_dim=4, hidden_dim=16)
        expected = legacy_policy(raw_features)
        migrated_policy = CommandEncoderPolicy(raw_features.shape[-1], z_dim=4, hidden_dim=16)
        checkpoint = {"policy": legacy_policy.state_dict(), "metadata": {}}
        migrated = load_command_encoder_policy_state(migrated_policy, checkpoint, scale)
        actual = migrated_policy(scaled_features)
        self.assertTrue(migrated)
        for expected_value, actual_value in zip(expected, actual, strict=True):
            self.assertTrue(torch.allclose(expected_value, actual_value, atol=1.0e-6))

    def test_backward_command_scale_only_changes_negative_vx(self):
        obs = {
            "state": torch.zeros(2, 10),
            "last_action": torch.zeros(2, 2),
            "history_actor": torch.zeros(2, 48),
        }
        commands = torch.tensor([[-0.4, 0.2, 0.1], [0.4, 0.2, 0.1]])
        baseline = flatten_encoder_observation(obs, commands, command_scale=(1.0, 1.0, 1.0))
        boosted = flatten_encoder_observation(
            obs,
            commands,
            command_scale=(1.0, 1.0, 1.0),
            backward_command_scale=1.4,
        )
        self.assertAlmostEqual(float(boosted[0, 0]), -0.56)
        self.assertAlmostEqual(float(boosted[1, 0]), 0.4)
        self.assertTrue(torch.equal(boosted[:, 1:], baseline[:, 1:]))
        with self.assertRaises(ValueError):
            flatten_encoder_observation(obs, commands, backward_command_scale=0.0)

    def test_optimizer_resume_overrides_checkpoint_learning_rate(self):
        policy = CommandEncoderPolicy(input_dim=5, z_dim=4, hidden_dim=16)
        old_optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-5)
        checkpoint = {"policy_optimizer": old_optimizer.state_dict()}
        optimizer = torch.optim.Adam(policy.parameters(), lr=9.0e-4)
        status = restore_policy_optimizer(
            optimizer,
            checkpoint,
            learning_rate=1.0e-4,
            reset_for_input_migration=False,
        )
        self.assertEqual(status, "loaded_with_lr_override")
        self.assertTrue(all(group["lr"] == 1.0e-4 for group in optimizer.param_groups))

    def test_discriminator_optimizer_resume_overrides_checkpoint_learning_rate(self):
        discriminator = AMPDiscriminator(feature_dim=5, hidden_dims=(16, 8))
        old_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2.0e-4)
        checkpoint = {"discriminator_optimizer": old_optimizer.state_dict()}
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=9.0e-4)
        status = restore_discriminator_optimizer(optimizer, checkpoint, learning_rate=1.0e-4)
        self.assertEqual(status, "loaded_with_lr_override")
        self.assertTrue(all(group["lr"] == 1.0e-4 for group in optimizer.param_groups))

    def test_signed_linvel_reward_removes_zero_speed_baseline(self):
        commands = torch.tensor([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
        achieved = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]])
        reward = baseline_normalized_linvel_reward(achieved, commands)
        self.assertAlmostEqual(float(reward[0]), 0.0, places=6)
        self.assertAlmostEqual(float(reward[1]), 1.0, places=6)
        self.assertLess(float(reward[2]), 0.0)

    def test_tracking_metrics_report_nonzero_response_slope(self):
        commands = torch.tensor([[[0.2, 0.0, -0.4], [0.4, 0.0, 0.2], [0.6, 0.0, 0.8]]])
        base_lin_vel = torch.tensor([[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]])
        base_ang_vel = torch.tensor([[[0.0, 0.0, -0.2], [0.0, 0.0, 0.1], [0.0, 0.0, 0.4]]])
        metrics = command_tracking_metrics(commands, base_lin_vel, base_ang_vel)
        self.assertAlmostEqual(metrics["tracking/nonzero_vx_response_slope"], 0.5, places=6)
        self.assertAlmostEqual(metrics["tracking/nonzero_vx_command_correlation"], 1.0, places=6)
        self.assertAlmostEqual(metrics["tracking/nonzero_vx_mae"], 0.2, places=6)
        self.assertAlmostEqual(metrics["tracking/nonzero_yaw_response_slope"], 0.5, places=6)
        self.assertAlmostEqual(metrics["tracking/nonzero_yaw_command_correlation"], 1.0, places=6)
        self.assertAlmostEqual(metrics["tracking/nonzero_yaw_rate_mae"], 0.2333333, places=6)
        self.assertAlmostEqual(metrics["tracking/stand_base_planar_speed_mean"], 0.0, places=6)
        self.assertAlmostEqual(metrics["tracking/stand_base_yaw_rate_abs_mean"], 0.0, places=6)

    def test_amp_discriminator_wgan_gp_is_finite(self):
        discriminator = AMPDiscriminator(feature_dim=12, hidden_dims=(32, 16))
        losses = discriminator.loss(torch.randn(7, 12), torch.randn(7, 12))
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in discriminator.parameters()))

    def test_gae_distinguishes_terminal_and_timeout(self):
        rewards = torch.tensor([[1.0], [2.0], [3.0]])
        values = torch.zeros_like(rewards)
        terminated = torch.tensor([[False], [True], [False]])
        truncated = torch.zeros_like(terminated)
        advantages, returns = compute_gae(
            rewards,
            values,
            terminated,
            truncated,
            torch.zeros_like(rewards),
            torch.tensor([4.0]),
            discount=0.9,
            gae_lambda=1.0,
        )
        expected = torch.tensor([[2.8], [2.0], [6.6]])
        self.assertTrue(torch.allclose(advantages, expected))
        self.assertTrue(torch.allclose(returns, expected))

    def test_transition_core_maps_target_isaac_tensors(self):
        simulator = SimpleNamespace(
            robot_root_states=torch.zeros(2, 13),
            _rigid_body_pos=torch.zeros(2, 4, 3),
            _rigid_body_rot=torch.zeros(2, 4, 4),
            _rigid_body_ang_vel=torch.zeros(2, 4, 3),
            contact_forces=torch.zeros(2, 4, 3),
            dof_pos=torch.zeros(2, 23),
            dof_vel=torch.zeros(2, 23),
        )
        live = SimpleNamespace(
            simulator=simulator,
            base_quat=torch.zeros(2, 4),
            base_lin_vel=torch.zeros(2, 3),
            base_ang_vel=torch.zeros(2, 3),
            torques=torch.zeros(2, 23),
            default_dof_pos=torch.zeros(23),
            default_dof_pos_offset=torch.zeros(2, 23),
            num_envs=2,
            device=torch.device("cpu"),
        )
        core = _transition_core(live, {})
        self.assertIs(core.dof_pos, simulator.dof_pos)
        self.assertIs(core.body_pos, simulator._rigid_body_pos)

    def test_transition_core_uses_pre_reset_snapshot(self):
        live = SimpleNamespace(num_envs=2, device="cpu", dof_pos=torch.full((2, 1), 9.0))
        snapshot = {"dof_pos": torch.tensor([[1.0], [2.0]])}
        transition = _transition_core(live, {"terminal_state": snapshot})
        self.assertTrue(torch.equal(transition.dof_pos, snapshot["dof_pos"]))

    def test_stage2_termination_matches_ufo(self):
        core = SimpleNamespace(
            num_envs=2,
            device=torch.device("cpu"),
            reset_buf=torch.zeros(2, dtype=torch.bool),
            extras={},
            termination_contact_indices=torch.tensor([1]),
            simulator=SimpleNamespace(
                contact_forces=torch.tensor(
                    [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
                )
            ),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0], [0.91, 0.0, -0.4]]),
        )
        _stage2_update_reset_buf(core)
        self.assertTrue(torch.equal(core.reset_buf, torch.tensor([True, True])))
        self.assertTrue(torch.equal(core.extras["termination_crash"], torch.tensor([True, False])))
        self.assertTrue(torch.equal(core.extras["termination_fall_over"], torch.tensor([False, True])))

    def test_stage2_captures_terminal_state_and_timeout_observation(self):
        simulator = SimpleNamespace(
            robot_root_states=torch.ones(2, 13),
            _rigid_body_pos=torch.ones(2, 3, 3),
            _rigid_body_rot=torch.ones(2, 3, 4),
            _rigid_body_ang_vel=torch.ones(2, 3, 3),
            contact_forces=torch.ones(2, 3, 3),
            dof_pos=torch.ones(2, 2),
            dof_vel=torch.ones(2, 2),
        )
        core = SimpleNamespace(
            simulator=simulator,
            base_quat=torch.ones(2, 4),
            base_lin_vel=torch.ones(2, 3),
            base_ang_vel=torch.ones(2, 3),
            projected_gravity=torch.ones(2, 3),
            torques=torch.ones(2, 2),
            default_dof_pos=torch.ones(2, 2),
            default_dof_pos_offset=torch.ones(2, 2),
            time_out_buf=torch.tensor([True, False]),
            extras={},
        )
        core._compute_observations = lambda: setattr(
            core,
            "obs_buf_dict_raw",
            {
                "actor_obs": {
                    "dof_pos": torch.ones(2, 2),
                    "dof_vel": torch.ones(2, 2),
                    "projected_gravity": torch.ones(2, 3),
                    "base_ang_vel": torch.ones(2, 3),
                    "actions": torch.ones(2, 2),
                    "max_local_self": torch.ones(2, 5),
                    "history_actor": torch.ones(2, 7),
                }
            },
        )
        _stage2_prepare_pre_reset_transition(core, torch.tensor([0]))
        self.assertIn("terminal_state", core.extras)
        self.assertIn("terminal_observation", core.extras)
        self.assertEqual(core.extras["terminal_observation"]["state"].shape, (2, 10))

    def test_survival_reward_is_available_for_stage2_override(self):
        core = SimpleNamespace(num_envs=3, device=torch.device("cpu"))
        survival = LeggedRobotBase._reward_survival(core)
        self.assertTrue(torch.equal(survival, torch.ones(3)))

    def test_mimiclite_reward_terms_are_stable(self):
        state = MimicLiteLocomotionRewardState(
            num_envs=1,
            num_dof=2,
            dt=0.02,
            feet_indices=torch.tensor([0, 1]),
            torso_index=0,
            joint_vel_indices=torch.tensor([0]),
            joint_deviation_indices=torch.tensor([0]),
            device=torch.device("cpu"),
        )
        core = SimpleNamespace(
            num_envs=1,
            device=torch.device("cpu"),
            base_lin_vel=torch.zeros(1, 3),
            base_ang_vel=torch.zeros(1, 3),
            body_ang_vel=torch.zeros(1, 1, 3),
            contact_forces=torch.zeros(1, 2, 3),
            body_pos=torch.tensor([[[0.0, 0.0, 0.35], [0.0, 0.0, 0.10]]]),
            body_rot=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
            torques=torch.zeros(1, 2),
            dof_vel=torch.zeros(1, 2),
            dof_pos=torch.zeros(1, 2),
            default_dof_pos=torch.zeros(1, 2),
            default_dof_pos_offset=torch.zeros(1, 2),
        )
        reward, components = state.compute(core, torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1, dtype=torch.bool))
        self.assertEqual(set(components), set(MIMICLITE_LOCOMOTION_WEIGHTS))
        self.assertTrue(torch.isfinite(reward).all())

    def test_stand_still_reward_prefers_default_pose_and_is_command_gated(self):
        state = MimicLiteLocomotionRewardState(
            num_envs=1,
            num_dof=2,
            dt=0.02,
            feet_indices=torch.tensor([0, 1]),
            torso_index=0,
            joint_vel_indices=torch.tensor([0]),
            joint_deviation_indices=torch.tensor([0]),
            device=torch.device("cpu"),
        )
        core = SimpleNamespace(
            num_envs=1,
            device=torch.device("cpu"),
            base_lin_vel=torch.zeros(1, 3),
            base_ang_vel=torch.zeros(1, 3),
            body_ang_vel=torch.zeros(1, 1, 3),
            contact_forces=torch.zeros(1, 2, 3),
            body_pos=torch.tensor([[[0.0, 0.0, 0.35], [0.0, 0.0, 0.10]]]),
            body_rot=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
            torques=torch.zeros(1, 2),
            dof_vel=torch.zeros(1, 2),
            dof_pos=torch.zeros(1, 2),
            default_dof_pos=torch.zeros(1, 2),
            default_dof_pos_offset=torch.zeros(1, 2),
        )
        done = torch.zeros(1, dtype=torch.bool)
        actions = torch.zeros(1, 2)
        _, default_components = state.compute(core, torch.zeros(1, 3), actions, done)
        core.dof_pos.fill_(0.4)
        _, crooked_components = state.compute(core, torch.zeros(1, 3), actions, done)
        _, moving_components = state.compute(core, torch.tensor([[0.4, 0.0, 0.0]]), actions, done)

        self.assertEqual(default_components["stand_still"].item(), 0.0)
        self.assertAlmostEqual(crooked_components["stand_still"].item(), -0.02 * 0.8 * 0.8)
        self.assertEqual(moving_components["stand_still"].item(), 0.0)

    def test_cross_axis_velocity_penalty_is_command_orthogonal_and_stand_gated(self):
        state = MimicLiteLocomotionRewardState(
            num_envs=1,
            num_dof=2,
            dt=0.02,
            feet_indices=torch.tensor([0, 1]),
            torso_index=0,
            joint_vel_indices=torch.tensor([0]),
            joint_deviation_indices=torch.tensor([0]),
            device=torch.device("cpu"),
            cross_axis_velocity_weight=5.0,
        )
        core = SimpleNamespace(
            num_envs=1,
            device=torch.device("cpu"),
            base_lin_vel=torch.tensor([[1.0, 0.2, 0.0]]),
            base_ang_vel=torch.zeros(1, 3),
            body_ang_vel=torch.zeros(1, 1, 3),
            contact_forces=torch.zeros(1, 2, 3),
            body_pos=torch.tensor([[[0.0, 0.0, 0.35], [0.0, 0.0, 0.10]]]),
            body_rot=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
            torques=torch.zeros(1, 2),
            dof_vel=torch.zeros(1, 2),
            dof_pos=torch.zeros(1, 2),
            default_dof_pos=torch.zeros(1, 2),
            default_dof_pos_offset=torch.zeros(1, 2),
        )
        _, moving = state.compute(core, torch.tensor([[1.0, 0.0, 0.0]]), torch.zeros(1, 2), torch.zeros(1, dtype=torch.bool))
        _, standing = state.compute(core, torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1, dtype=torch.bool))
        self.assertAlmostEqual(moving["cross_axis_velocity_l2"].item(), -0.02 * 5.0 * 0.2**2, places=7)
        self.assertEqual(standing["cross_axis_velocity_l2"].item(), 0.0)

    def test_command_sampler_supports_standing(self):
        low = torch.tensor([-0.5, -0.2, -0.8])
        high = torch.tensor([1.2, 0.2, 0.8])
        commands = _sample_commands(32, torch.device("cpu"), low, high, stand_prob=1.0)
        self.assertTrue(torch.equal(commands, torch.zeros_like(commands)))

    def test_command_sampler_supports_turning_in_place(self):
        low = torch.tensor([-0.8, -0.5, 0.5])
        high = torch.tensor([0.8, 0.5, 0.5])
        commands = _sample_commands(32, torch.device("cpu"), low, high, stand_prob=0.0, turn_prob=1.0)
        self.assertTrue(torch.equal(commands[:, :2], torch.zeros_like(commands[:, :2])))
        self.assertTrue(torch.equal(commands[:, 2], torch.full((32,), 0.5)))

    def test_done_command_targets_are_resampled_to_avoid_warmup_starvation(self):
        targets = torch.zeros(4, 3)
        done = torch.tensor([True, False, True, False])
        low = torch.tensor([-0.8, -0.5, -0.8])
        high = torch.tensor([0.8, 0.5, 0.8])
        torch.manual_seed(7)
        updated = _resample_done_command_targets(
            targets,
            done,
            low,
            high,
            stand_prob=0.0,
            turn_prob=0.0,
            lateral_prob=0.0,
        )
        self.assertTrue(torch.equal(updated[~done], torch.zeros_like(updated[~done])))
        self.assertTrue(torch.any(updated[done].abs() > 0.0))
        self.assertTrue(torch.equal(targets, torch.zeros_like(targets)))

    def test_torchrun_command_uses_standard_pytorch_launcher(self):
        command = _torchrun_command(4, ["--smoke", "--gpu-ids", "all"])
        self.assertEqual(command[1:4], ["-m", "torch.distributed.run", "--standalone"])
        self.assertIn("--nproc_per_node=4", command)
        self.assertEqual(command[-3:], ["--smoke", "--gpu-ids", "all"])


class Amp018FootSeparationPenaltyTest(unittest.TestCase):
    def _state(self):
        return Amp018MimicLiteLocomotionRewardState(
            num_envs=1,
            num_dof=2,
            dt=0.02,
            feet_indices=torch.tensor([0, 1]),
            torso_index=0,
            joint_vel_indices=torch.tensor([0]),
            joint_deviation_indices=torch.tensor([0]),
            device=torch.device("cpu"),
        )

    def _core(self, foot_y: float) -> SimpleNamespace:
        return SimpleNamespace(
            num_envs=1,
            device=torch.device("cpu"),
            base_lin_vel=torch.zeros(1, 3),
            base_ang_vel=torch.zeros(1, 3),
            body_ang_vel=torch.zeros(1, 1, 3),
            contact_forces=torch.zeros(1, 2, 3),
            body_pos=torch.tensor([[[0.0, foot_y, 0.05], [0.0, -foot_y, 0.05]]]),
            body_rot=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
            torques=torch.zeros(1, 2),
            dof_vel=torch.zeros(1, 2),
            dof_pos=torch.zeros(1, 2),
            default_dof_pos=torch.zeros(1, 2),
            default_dof_pos_offset=torch.zeros(1, 2),
        )

    def test_foot_separation_penalty_penalizes_below_threshold(self):
        state = self._state()
        core = self._core(0.05)  # lateral separation 0.10 < 0.14
        _, components = state.compute(core, torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1, dtype=torch.bool))
        violation = AMP018_FOOT_MIN_SEPARATION - 0.10
        expected = -state.dt * 20.0 * violation * violation
        self.assertAlmostEqual(components["foot_separation_penalty"].item(), expected, places=7)

    def test_foot_separation_penalty_is_zero_above_threshold(self):
        state = self._state()
        core = self._core(0.10)  # lateral separation 0.20 > 0.14
        _, components = state.compute(core, torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1, dtype=torch.bool))
        self.assertEqual(components["foot_separation_penalty"].item(), 0.0)

    def test_foot_separation_penalty_is_command_gated(self):
        state = self._state()
        core = self._core(0.05)
        commands_cases = {
            "slow forward": torch.tensor([[0.3, 0.0, 0.0]]),
            "fast forward": torch.tensor([[0.6, 0.0, 0.0]]),
            "lateral": torch.tensor([[0.0, 0.4, 0.0]]),
            "turning": torch.tensor([[0.0, 0.0, 0.6]]),
            "standing": torch.zeros(1, 3),
        }
        values = {}
        for name, commands in commands_cases.items():
            _, components = state.compute(core, commands, torch.zeros(1, 2), torch.zeros(1, dtype=torch.bool))
            values[name] = components["foot_separation_penalty"].item()
        self.assertEqual(values["standing"], values["slow forward"])
        self.assertLess(values["standing"], 0.0)
        self.assertEqual(values["fast forward"], 0.0)
        self.assertEqual(values["lateral"], 0.0)
        self.assertEqual(values["turning"], 0.0)

    def test_foot_separation_diagnostics_are_reported(self):
        state = self._state()
        core = self._core(0.05)
        state.compute(core, torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1, dtype=torch.bool))
        diagnostics = state.last_feet_diagnostics
        self.assertEqual(set(diagnostics), set(AMP018_FEET_DIAGNOSTIC_KEYS))
        self.assertAlmostEqual(diagnostics["lateral_foot_separation_mean"].item(), 0.10, places=6)
        self.assertAlmostEqual(diagnostics["foot_separation_violation_mean"].item(), AMP018_FOOT_MIN_SEPARATION - 0.10, places=6)
        self.assertEqual(diagnostics["foot_separation_active_fraction"].item(), 1.0)

    def test_foot_separation_requires_two_feet(self):
        with self.assertRaises(ValueError):
            Amp018MimicLiteLocomotionRewardState(
                num_envs=1,
                num_dof=2,
                dt=0.02,
                feet_indices=torch.tensor([0, 1, 2]),
                torso_index=0,
                joint_vel_indices=torch.tensor([0]),
                joint_deviation_indices=torch.tensor([0]),
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
