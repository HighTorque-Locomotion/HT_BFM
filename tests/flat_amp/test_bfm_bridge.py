"""Exercise actual ordering, control conversion, episode boundaries and PPO latent updates on CPU."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch
from legged_lab.utils.bfm_bridge import (
    BFMBridge,
    BFMContract,
    BFMLatentEnv,
    validate_frozen_state,
    with_terminal_amp_capture,
)
from legged_lab.utils.bfm_training import BFMOnPolicyRunner, make_bfm_train_cfg
from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.modules.bfm_command_actor_critic import BFMCommandActorCritic
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.utils import Normalizer
from torch import nn

BFM_JOINTS = (
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_upper_arm_joint",
    "r_elbow_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_upper_arm_joint",
    "l_elbow_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_thigh_joint",
    "r_calf_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_thigh_joint",
    "l_calf_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
)
ISAAC_JOINTS = (
    "l_hip_pitch_joint",
    "r_hip_pitch_joint",
    "l_hip_roll_joint",
    "r_hip_roll_joint",
    "head_yaw_joint",
    "l_shoulder_pitch_joint",
    "r_shoulder_pitch_joint",
    "l_thigh_joint",
    "r_thigh_joint",
    "head_pitch_joint",
    "l_shoulder_roll_joint",
    "r_shoulder_roll_joint",
    "l_calf_joint",
    "r_calf_joint",
    "l_upper_arm_joint",
    "r_upper_arm_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_elbow_joint",
    "r_elbow_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
)


def contract():
    return BFMContract(
        BFM_JOINTS,
        tuple(i / 100 for i in range(22)),
        tuple(0.1 + i / 100 for i in range(22)),
    )


def test_joint_mapping_and_absolute_target_round_trip():
    bridge = BFMBridge(contract(), list(ISAAC_JOINTS), 2, "cpu")
    source_q = torch.arange(22).float().expand(2, -1)
    target_q = source_q[:, bridge.to_target]
    obs = bridge.observe(target_q, target_q + 1, torch.zeros(2, 3), torch.ones(2, 3))
    torch.testing.assert_close(obs["state"][:, :22], source_q - bridge.default_pos)
    torch.testing.assert_close(obs["state"][:, 22:44], source_q + 1)
    torch.testing.assert_close(obs["state"][:, -3:], torch.full((2, 3), 0.25))
    output = torch.linspace(-2, 2, 22).expand(2, -1)
    target_default = torch.linspace(0.4, -0.2, 22)
    target_scale = torch.linspace(0.05, 0.3, 22)
    actions, sim_action = bridge.target_actions(output, target_default, target_scale)
    actual_targets = actions * target_scale + target_default
    expected_targets = (bridge.default_pos + (output * 32).clamp(-32, 32) * bridge.action_scale)[:, bridge.to_target]
    torch.testing.assert_close(actual_targets, expected_targets)
    torch.testing.assert_close(sim_action, (output * 32).clamp(-32, 32))


def test_reject_missing_duplicate_joints_and_invalid_latents():
    for names in (list(ISAAC_JOINTS[:-1]), [ISAAC_JOINTS[0]] * 22):
        with pytest.raises(ValueError):
            BFMBridge(contract(), names, 2, "cpu")
    bridge = BFMBridge(contract(), list(ISAAC_JOINTS), 2, "cpu")
    for z in (torch.zeros(2, 22), torch.full((2, 256), float("nan"))):
        with pytest.raises(ValueError):
            bridge.project_z(z)
    torch.testing.assert_close(bridge.project_z(torch.randn(2, 256)).norm(dim=-1), torch.full((2,), 16.0))


def test_grouped_history_order_scaled_last_action_and_partial_reset():
    bridge = BFMBridge(contract(), list(BFM_JOINTS), 2, "cpu")
    obs = bridge.observe(
        torch.ones(2, 22),
        torch.full((2, 22), 7.0),
        torch.full((2, 3), -1.0),
        torch.full((2, 3), 8.0),
    )
    bridge.advance(obs, torch.full((2, 22), 12.0), torch.tensor([], dtype=torch.long))
    second = bridge.observe(
        torch.full((2, 22), 2.0),
        torch.ones(2, 22),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
    )
    assert second["last_action"][0, 0] == 12
    history = second["history_actor"]
    assert history[0, 88] == 2  # base_ang_vel field follows 4*22 actions, already scaled by 0.25
    assert history[0, 4 * (22 + 3 + 22)] == 7
    features = bridge.encoder_observation(second, torch.ones(2, 3))
    assert features.shape == (2, 363)
    assert features[0, 3 + 50 + 22 + 4 * (22 + 3 + 22)] == pytest.approx(0.35)
    bridge.advance(second, torch.full((2, 22), 16.0), torch.tensor([0]))
    assert bridge.last_action[0].count_nonzero() == 0
    assert bridge.last_action[1, 0] == 16
    assert bridge.history["actions"][0].count_nonzero() == 0
    assert bridge.history["actions"][1, 0, 0] == 12
    assert bridge.history["dof_vel"][1, 1, 0] == 7


def base_cfg():
    return {
        "seed": 42,
        "device": "cpu",
        "num_steps_per_env": 4,
        "max_iterations": 1,
        "empirical_normalization": False,
        "save_interval": 1,
        "logger": "tensorboard",
        "experiment_name": "walk_pi_waq_simple_22",
        "policy": {"critic_hidden_dims": [32, 16], "activation": "elu"},
        "algorithm": {
            "class_name": "AMPWAQDEPTHPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "learning_rate": 1e-3,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "value_loss_coef": 1.0,
            "dagger_loss_coef": 0.0,
            "dagger_loss_min_coef": 0.1,
            "dagger_loss_decay": True,
            "dagger_only": False,
            "use_amp": True,
            "freeze_amp": False,
            "symmetry_cfg": {"use_mirror_loss": True, "mirror_loss_coeff": 10.0},
        },
        "amp_loader_class_name": "AMPLoaderPi22",
        "amp_reward_coef": 0.3,
        "amp_task_reward_lerp": 0.65,
        "amp_discr_hidden_dims": [32, 16],
        "amp_motion_files": ["expert.npz"],
        "amp_motion_weights": {"expert": 2.0},
        "amp_num_preload_transitions": 16,
        "amp_exclude_ankle_roll": True,
        "min_normalized_std": [0.05] * 22,
    }


def test_target_amp_and_ppo_coefficients_are_preserved():
    cfg = base_cfg()
    original = copy.deepcopy(cfg)
    migrated = make_bfm_train_cfg(cfg)
    assert cfg == original
    for key, value in cfg.items():
        if key.startswith("amp_") or key in ("num_steps_per_env", "max_iterations"):
            assert migrated[key] == value
    for key, value in cfg["algorithm"].items():
        if key not in (
            "class_name",
            "symmetry_cfg",
            "dagger_loss_coef",
            "dagger_loss_min_coef",
            "dagger_loss_decay",
            "dagger_only",
            "use_amp",
            "freeze_amp",
        ):
            assert migrated["algorithm"][key] == value


def test_frozen_checkpoint_requires_actor_and_normalization_buffers(tmp_path):
    from safetensors.torch import save_file

    model = nn.Module()
    model._actor = nn.Linear(2, 2)
    model._obs_normalizer = nn.BatchNorm1d(2)
    path = tmp_path / "model.safetensors"
    state = model.state_dict()
    save_file(state, str(path))
    validate_frozen_state(model, path)
    del state["_obs_normalizer.running_mean"]
    save_file(state, str(path))
    with pytest.raises(ValueError, match="missing frozen tensors"):
        validate_frozen_state(model, path)
    state = model.state_dict()
    state["_actor.weight"] = torch.full((2, 2), float("nan"))
    save_file(state, str(path))
    with pytest.raises(ValueError, match="Invalid BFM checkpoint tensor"):
        validate_frozen_state(model, path)


class FakePiEnv:
    """Minimal physical environment with real reset semantics, not a learned dynamics substitute."""

    def __init__(self):
        self.num_envs, self.num_actions, self.device, self.step_dt = 4, 22, "cpu", 0.02
        self.max_episode_length = 5
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.action_scale = torch.full((22,), 0.25)
        self.obs_scales = SimpleNamespace(joint_pos=1.0, joint_vel=0.05, ang_vel=0.25, projected_gravity=1.0)
        self.robot = SimpleNamespace(
            joint_names=list(ISAAC_JOINTS),
            data=SimpleNamespace(
                default_joint_pos=torch.zeros(4, 22),
                default_joint_vel=torch.zeros(4, 22),
            ),
        )
        self.extras = {}
        self.reset_env_ids = torch.tensor([], dtype=torch.long)
        self.action = torch.zeros(4, 22)
        self.cfg = SimpleNamespace(gait=SimpleNamespace(gait_cycle=0.8, lin_command_threshold=0.1, yaw_command_threshold=0.1))
        self.phase = torch.tensor([0.0, 0.5, 0.25, 0.75])

    def get_observations(self):
        frame = torch.zeros(4, 77)
        frame[:, 5] = -1
        frame[:, 6] = 0.5
        frame[:, 9:31] = self.action * self.action_scale
        frame[:, 53:75] = self.action
        frame[:, 75] = torch.sin(2 * torch.pi * self.phase)
        frame[:, 76] = torch.cos(2 * torch.pi * self.phase)
        self.extras["observations"] = {
            "current_actor_obs_add_nosie": frame,
            "critic": torch.cat((frame, torch.zeros(4, 5)), -1),
        }
        return frame.repeat(1, 10), self.extras

    def get_amp_obs_for_expert_trans(self):
        return self.action[:, :6].clone()

    def reset(self, env_ids):
        self.action[env_ids] = 0
        self.phase[env_ids] = 0.5

    def step(self, actions):
        assert actions.shape == (4, 22)
        self.action = actions.clone()
        self.episode_length_buf += 1
        self.phase = (self.phase + self.step_dt / self.cfg.gait.gait_cycle) % 1.0
        done = self.episode_length_buf % 3 == 0
        self.reset_env_ids = done.nonzero().flatten()
        self.extras["time_outs"] = torch.zeros_like(done)
        reward = 1 - self.action.square().mean(-1)
        self.reset(self.reset_env_ids)
        _, extras = self.get_observations()
        return (
            extras["observations"]["current_actor_obs_add_nosie"],
            reward,
            done,
            extras,
        )


class FakeFrozenActor:
    def __init__(self):
        self.model = nn.Linear(256, 22).eval().requires_grad_(False)
        self.digest = "test-only-not-a-robot-checkpoint"

    def __call__(self, obs, z):
        return 0.01 * self.model(z)


class ExpertTransitions:
    observation_dim = 6

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            yield torch.randn(mini_batch_size, 6), torch.randn(mini_batch_size, 6)


@pytest.mark.parametrize("use_phase", [False, True])
def test_real_ppo_updates_latent_encoder_and_discriminator_but_not_frozen_actor(
    use_phase,
):
    torch.manual_seed(8)
    physical = with_terminal_amp_capture(FakePiEnv)()
    frozen = FakeFrozenActor()
    env = BFMLatentEnv(physical, frozen, contract(), use_phase=use_phase)
    obs, extras = env.get_observations()
    critic = extras["observations"]["critic"]
    policy = BFMCommandActorCritic(obs.shape[-1], critic.shape[-1], 256, critic_hidden_dims=[32], hidden_dim=32)
    disc = Discriminator(12, 0.3, [32, 16], "cpu", 0.65)
    alg = AMPPPO(
        policy,
        disc,
        ExpertTransitions(),
        Normalizer(6),
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
        learning_rate=1e-3,
        amp_replay_buffer_size=64,
    )
    alg.init_storage("rl", 4, 4, [obs.shape[-1]], [critic.shape[-1]], [256])
    frozen_before = {k: v.clone() for k, v in frozen.model.state_dict().items()}
    encoder_before = policy.actor[-1].weight.clone()
    input_before = policy.actor[0].weight.clone()
    disc_before = disc.amp_linear.weight.clone()
    amp_obs = physical.get_amp_obs_for_expert_trans()
    with torch.inference_mode():
        for _ in range(4):
            z = alg.act(obs, critic, amp_obs)
            obs, reward, done, extras = env.step(z)
            critic = extras["observations"]["critic"]
            next_amp = physical.get_amp_obs_for_expert_trans()
            terminal_amp = next_amp.clone()
            terminal_amp[physical.reset_env_ids] = extras["terminal_amp_states"]
            if done.any():
                assert next_amp[done].count_nonzero() == 0
                assert terminal_amp[done].count_nonzero() > 0
            mixed, _, _ = disc.predict_amp_reward(amp_obs, terminal_amp, reward, alg.amp_normalizer)
            alg.process_env_step(mixed, done, extras, terminal_amp)
            amp_obs = next_amp
        alg.compute_returns(critic)
    loss = alg.update()
    assert loss is not None
    assert not torch.equal(encoder_before, policy.actor[-1].weight)
    if use_phase:
        assert obs.shape[-1] == 365 and critic.shape[-1] == 447
        assert not torch.equal(input_before[:, -2:], policy.actor[0].weight[:, -2:])
    assert not torch.equal(disc_before, disc.amp_linear.weight)
    for key, value in frozen.model.state_dict().items():
        torch.testing.assert_close(value, frozen_before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in frozen.model.parameters())
    assert alg.storage.actions.shape[-1] == 256
    assert physical.action.shape[-1] == 22


@pytest.mark.parametrize("use_phase", [False, True])
def test_runner_save_resume_contract_and_joint_vs_latent_dimensions(monkeypatch, tmp_path, use_phase):
    import rsl_rl.runners.amp_on_policy_runner as runner_module

    monkeypatch.setattr(runner_module, "AMPLoaderPi22", lambda *args, **kwargs: ExpertTransitions())
    monkeypatch.setattr(runner_module, "store_code_state", lambda *args, **kwargs: [])
    physical = with_terminal_amp_capture(FakePiEnv)()
    env = BFMLatentEnv(physical, FakeFrozenActor(), contract(), use_phase=use_phase)
    runner = BFMOnPolicyRunner(env, make_bfm_train_cfg(base_cfg(), use_phase=use_phase), device="cpu")
    runner.learn(1, init_at_random_ep_len=True)
    runner.current_learning_iteration = 9
    path = tmp_path / "model_9.pt"
    runner.save(path)
    runner.current_learning_iteration = 0
    runner.load(path)
    assert runner.current_learning_iteration == 9
    assert runner.bfm_metadata["schema"] == (2 if use_phase else 1)
    incompatible_env = BFMLatentEnv(
        with_terminal_amp_capture(FakePiEnv)(),
        FakeFrozenActor(),
        contract(),
        use_phase=not use_phase,
    )
    incompatible = BFMOnPolicyRunner(
        incompatible_env,
        make_bfm_train_cfg(base_cfg(), use_phase=not use_phase),
        device="cpu",
    )
    with pytest.raises(ValueError, match="identity/control contract"):
        incompatible.load(path)
    runner.bfm_metadata["actor_sha256"] = "different-bfm"
    with pytest.raises(ValueError, match="identity/control contract"):
        runner.load(path)


def test_initial_latent_std_is_in_latent_coordinates():
    policy = BFMCommandActorCritic(363, 445, 256, critic_hidden_dims=[16])
    policy.update_distribution(torch.zeros(2, 363))
    torch.testing.assert_close(policy.action_std, torch.full((2, 256), math.exp(-1.5)))


def test_explicit_phase_changes_encoder_but_not_frozen_bfm_observations():
    physical = FakePiEnv()
    env = BFMLatentEnv(physical, FakeFrozenActor(), contract())
    observations, extras = env.get_observations()
    frame = extras["observations"]["current_actor_obs_add_nosie"]
    assert observations.shape == (4, 365)
    torch.testing.assert_close(observations[:, -2:], frame[:, 75:77], rtol=0, atol=0)
    torch.testing.assert_close(observations[0, :363], observations[1, :363], rtol=0, atol=0)
    assert not torch.equal(observations[0, -2:], observations[1, -2:])
    for value in env._observations.values():
        torch.testing.assert_close(value[0], value[1], rtol=0, atol=0)
    assert set(env._observations) == {"state", "last_action", "history_actor"}
    assert env._observations["history_actor"].shape[-1] == 288


def test_phase_uses_post_reset_frame_and_does_not_reset_other_memory():
    physical = FakePiEnv()
    env = BFMLatentEnv(physical, FakeFrozenActor(), contract())
    env.get_observations()
    physical.episode_length_buf = torch.tensor([2, 0, 0, 0])
    observations, _, done, extras = env.step(torch.ones(4, 256))
    assert done.tolist() == [True, False, False, False]
    torch.testing.assert_close(
        observations[:, -2:],
        extras["observations"]["current_actor_obs_add_nosie"][:, 75:77],
    )
    assert observations[0, -1] == -1
    assert not env.bridge.last_action[0].any()
    assert env.bridge.last_action[1].any()
    assert env.bridge.history["projected_gravity"][0].count_nonzero() == 0
    assert env.bridge.history["projected_gravity"][1].count_nonzero() > 0


def test_phase_shape_is_not_silently_ignored():
    env = BFMLatentEnv(FakePiEnv(), FakeFrozenActor(), contract())
    env.get_observations()
    with pytest.raises(ValueError, match="phase sin/cos"):
        env.bridge.encoder_observation(env._observations, torch.zeros(4, 3), torch.zeros(4, 1))
