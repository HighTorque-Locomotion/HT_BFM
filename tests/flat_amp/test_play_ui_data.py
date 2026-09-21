from types import SimpleNamespace

import pytest
import torch

from humanoidverse.flat_amp.play_ui_data import PlaySignals


def make_env(joints=12):
    actor = torch.arange(2 * (11 + 3 * joints), dtype=torch.float32).reshape(2, -1)
    critic = torch.cat((actor, torch.ones(2, 5)), dim=1)
    env = SimpleNamespace(
        num_envs=2,
        action=torch.zeros(2, joints),
        robot=SimpleNamespace(
            joint_names=[f"joint_{i}" for i in range(joints)],
            data=SimpleNamespace(default_joint_pos=torch.zeros(2, joints), applied_torque=torch.ones(2, joints)),
        ),
        feet_cfg=SimpleNamespace(body_names=[".*foot"], body_ids=[1, 2]),
        contact_sensor=SimpleNamespace(body_names=["torso", "left_foot", "right_foot"]),
        extras={"observations": {"current_actor_obs_add_nosie": actor, "current_critic_obs": critic}},
    )
    return env


@pytest.mark.parametrize("joints", [12, 22, 23])
def test_plots_read_exact_policy_frame_without_mutating_history(joints):
    env = make_env(joints)
    signals = PlaySignals(env)
    before = env.extras["observations"]["current_actor_obs_add_nosie"].clone()
    for env_idx in (1, 0):
        terms = signals.terms("observations", env_idx)
        actor_values = [v for name, values, _ in terms if name.startswith("policy/") for v in values]
        assert actor_values == before[env_idx].tolist()
        assert all(len(values) == len(legends) for _, values, legends in terms)
    torch.testing.assert_close(env.extras["observations"]["current_actor_obs_add_nosie"], before)
    with pytest.raises(ValueError, match="out of range"):
        signals.terms("observations", 2)


def test_actions_distinguish_policy_applied_and_position_targets():
    env = make_env()
    signals = PlaySignals(env)
    signals.raw_actions[:] = 10
    env.action[:] = 1
    signals.joint_targets[:] = 0.5
    terms = dict((name, values) for name, values, _ in signals.terms("actions", 0))
    assert terms["policy_actions"] == [10] * 12
    assert terms["applied_actions"] == [1] * 12
    assert terms["joint_position_target (rad)"] == [0.5] * 12


def test_ground_force_adds_friction_and_preserves_valid_collision_pairs():
    env = make_env()
    normal = torch.tensor([[[[1., 2., 3.], [float("nan")] * 3]]]).repeat(2, 1, 1, 1)
    friction = torch.tensor([[[[4., 5., 6.], [float("nan")] * 3]]]).repeat(2, 1, 1, 1)
    sensor = SimpleNamespace(body_names=["ankle"], data=SimpleNamespace(force_matrix_w=normal, friction_forces_w=friction))
    env.scene = SimpleNamespace(sensors={"left_foot_force": sensor, "right_foot_force": sensor})
    for _, values, legends in PlaySignals(env).terms("contact_forces", 0):
        assert values == [5, 7, 9]
        assert legends == ["Fx", "Fy", "Fz"]


def test_reward_and_termination_use_cached_results():
    env = make_env()
    env.reward_manager = SimpleNamespace(get_active_iterable_terms=lambda i: [("tracking", [float(i)])])
    env.reset_reason_buf = {"contact_excess": torch.tensor([False, True]), "time_out": torch.tensor([True, False])}
    signals = PlaySignals(env)
    assert signals.terms("rewards", 1) == [("tracking", [1.0], ["tracking"])]
    assert signals.terms("termination", 0) == [
        ("contact_excess", [False], ["contact_excess"]), ("time_out", [True], ["time_out"]),
    ]
