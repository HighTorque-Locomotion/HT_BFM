import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from humanoidverse.amp_stage2 import (
    AMPDiscriminator,
    CommandEncoderPolicy,
    _load_piplus_robot_contract,
    _motion_qpos,
    _policy_dof_from_motion,
    _sample_commands,
    _torchrun_command,
    _transition_core,
    compute_gae,
    project_latent,
)


class AmpStage2Test(unittest.TestCase):
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

    def test_command_sampler_supports_standing(self):
        low = torch.tensor([-0.5, -0.2, -0.8])
        high = torch.tensor([1.2, 0.2, 0.8])
        commands = _sample_commands(32, torch.device("cpu"), low, high, stand_prob=1.0)
        self.assertTrue(torch.equal(commands, torch.zeros_like(commands)))

    def test_torchrun_command_uses_standard_pytorch_launcher(self):
        command = _torchrun_command(4, ["--smoke", "--gpu-ids", "all"])
        self.assertEqual(command[1:4], ["-m", "torch.distributed.run", "--standalone"])
        self.assertIn("--nproc_per_node=4", command)
        self.assertEqual(command[-3:], ["--smoke", "--gpu-ids", "all"])


if __name__ == "__main__":
    unittest.main()
