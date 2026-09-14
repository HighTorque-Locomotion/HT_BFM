"""CPU coverage for the delivered ordinary AMP path and its data/assets."""

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import rsl_rl
import torch
from legged_lab.scripts.export_policy import Policy
from rsl_rl.algorithms import AMPWAQPPO
from rsl_rl.modules import ActorCriticWaq, Discriminator
from rsl_rl.utils import AMPLoaderPi, AMPLoaderPi22, Normalizer

from humanoidverse.flat_amp import VENDOR_ROOT

ROOT = VENDOR_ROOT


@pytest.mark.parametrize("joints", [22, 23])
def test_amp_updates_policy_vae_discriminator_and_exports(joints):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    dim = 9 + 3 * joints + 2
    policy = ActorCriticWaq(dim, dim + 5, dim * 10, joints, actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16])
    loader_cls = AMPLoaderPi22 if joints == 22 else AMPLoaderPi
    motion = "pi_22_amp/walk_forward_retargetted.npz" if joints == 22 else "new_npz/left.npz"
    loader = loader_cls(
        "cpu",
        time_between_frames=0.02,
        preload_transitions=True,
        num_preload_transitions=128,
        amp_exclude_ankle_roll=False,
        motion_files=[str(ROOT / "legged_lab/envs/pi/datasets/motion_amp_expert" / motion)],
    )
    assert loader.observation_dim == (65 if joints == 22 else 70)
    discriminator = Discriminator(loader.observation_dim * 2, 0.3, [32, 16], "cpu", 0.65)
    algo = AMPWAQPPO(
        policy,
        discriminator,
        loader,
        Normalizer(loader.observation_dim),
        amp_replay_buffer_size=64,
        num_learning_epochs=2,
        num_mini_batches=2,
        min_std=torch.zeros(joints),
        beta=1e-4,
    )
    algo.init_storage("rl", 4, 4, dim, [dim], [dim + 5], [dim * 10], [joints])
    before = copy.deepcopy(policy.state_dict())
    before_disc = copy.deepcopy(discriminator.state_dict())
    with torch.inference_mode():
        for _ in range(4):
            obs = torch.randn(4, dim)
            critic = torch.randn(4, dim + 5)
            history = torch.randn(4, dim * 10)
            actions = algo.act(obs, critic, history, torch.randn(4, loader.observation_dim))
            assert actions.shape == (4, joints)
            algo.process_env_step(
                torch.randn(4),
                torch.zeros(4, dtype=torch.bool),
                {"time_outs": torch.zeros(4)},
                torch.randn(4, loader.observation_dim),
                torch.randn(4, dim),
            )
        algo.compute_returns(critic)
    losses = algo.update()
    assert all(math.isfinite(float(v)) for v in losses.values())
    for key in ("actor.0.weight", "critic.0.weight", "encoder_module.0.weight", "cv_vel.weight"):
        assert not torch.equal(before[key], policy.state_dict()[key]), key
    assert any(not torch.equal(before_disc[k], v) for k, v in discriminator.state_dict().items())
    history = torch.randn(4, dim * 10)
    scripted = torch.jit.script(Policy(policy).eval())
    with torch.no_grad():
        torch.testing.assert_close(scripted(history), policy.act_inference(history[:, -dim:], history)[0])


def test_urdfs_have_expected_joints_and_all_meshes():
    assets = ROOT / "legged_lab/assets/pi"
    urdfs = list(assets.rglob("*.urdf"))
    assert len(urdfs) == 2
    for path in urdfs:
        root = ET.parse(path)
        joints = [j.attrib["name"] for j in root.findall("joint") if j.attrib["type"] in {"revolute", "continuous", "prismatic"}]
        assert len(joints) == (22 if "22dof" in path.name else 23)
        assert len(joints) == len(set(joints))
        for mesh in root.findall(".//mesh"):
            assert (path.parent / mesh.attrib["filename"]).is_file()
        assert any(link.attrib["name"] == "camera_link" for link in root.findall("link"))


def test_no_perception_or_student_training_packages():
    assert Path(rsl_rl.__file__).resolve().is_relative_to(ROOT), "Use the project's bundled RSL-RL."
    assert not (ROOT / "legged_lab/sensors").exists()
    assert not (ROOT / "legged_lab/terrains").exists()
    assert {p.name for p in (ROOT / "rsl_rl/algorithms").glob("*.py")} == {
        "__init__.py",
        "amp_ppo.py",
        "amp_waq_ppo.py",
    }
    assert {p.name for p in (ROOT / "legged_lab/envs/pi").glob("*cfg*.py")} == {"pi_cfg.py", "pi_22dof_cfg.py"}
