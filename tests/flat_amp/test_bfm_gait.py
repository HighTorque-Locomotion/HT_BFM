"""Historical 0.5/0.8-second command clock and strict variable-phase checkpoint contract."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from legged_lab.utils.bfm_bridge import BFMLatentEnv, with_terminal_amp_capture
from legged_lab.utils.bfm_gait import BackwardFastGait, CommandGaitClock, ContactCycleTracker, with_gait_environment
from legged_lab.utils.bfm_training import BFMOnPolicyRunner, configure_backward_fast_gait, make_bfm_train_cfg
from test_bfm_bridge import ExpertTransitions, FakeFrozenActor, FakePiEnv, base_cfg, contract


def test_backward_mixed_commands_use_half_second_and_other_motion_uses_point_eight():
    command = torch.tensor([[-0.3, 0, 0], [-0.3, 0.2, 0.6], [0.3, 0, 0], [0, 0.25, 0], [0, 0, 0.6], [0, 0, 0]])
    clock = CommandGaitClock(6, "cpu", BackwardFastGait())
    phase = clock.advance(command, 0.02, torch.zeros(6))
    torch.testing.assert_close(clock.period, torch.tensor([0.5, 0.5, 0.8, 0.8, 0.8, 0.8]))
    torch.testing.assert_close(phase, torch.tensor([0.04, 0.04, 0.025, 0.025, 0.025, 0]))


def test_period_switch_integrates_progress_and_standing_discards_offset():
    clock = CommandGaitClock(1, "cpu", BackwardFastGait())
    for _ in range(11):
        clock.advance(torch.tensor([[0.3, 0, 0]]), 0.02, torch.zeros(1))
    phase = clock.advance(torch.tensor([[-0.3, 0, 0]]), 0.02, torch.zeros(1))
    assert phase.item() == pytest.approx(0.315)
    assert clock.advance(torch.zeros(1, 3), 0.02, torch.ones(1) * 0.5).item() == 0
    assert clock.progress.item() == 0


def test_complete_cycle_and_reset_do_not_modify_other_environments():
    clock = CommandGaitClock(2, "cpu", BackwardFastGait())
    command = torch.tensor([[-0.3, 0, 0], [-0.3, 0, 0]])
    offset = torch.tensor([0, 0.5])
    for _ in range(26):
        phase = clock.advance(command, 0.02, offset)
    torch.testing.assert_close(phase, torch.tensor([0.04, 0.54]))
    clock.reset(torch.tensor([0]))
    assert clock.progress[0] == 0
    assert clock.progress[1].item() == pytest.approx(0.04)
    assert clock.period.tolist() == pytest.approx([0.8, 0.5])


@pytest.mark.parametrize("value", [0, -0.5, float("inf"), float("nan")])
def test_invalid_clock_period_rejected(value):
    with pytest.raises(ValueError):
        BackwardFastGait(backward_period=value)


@pytest.mark.parametrize("steps", [25, 40])
def test_contact_diagnostics_measure_full_cycle_and_reject_chatter(steps):
    tracker = ContactCycleTracker(1, "cpu")
    measured = []
    for index in range(steps * 4):
        force = torch.full((1, 2), 0.0 if index % steps < steps // 2 else 10.0)
        event, period = tracker.update(force, 0.02, torch.tensor([True]))
        measured.extend(period[event].tolist())
    assert len(measured) == 6
    assert measured == pytest.approx([steps * 0.02] * 6, abs=1e-5)
    tracker.reset(torch.tensor([0]))
    for index in range(20):
        event, _ = tracker.update(torch.full((1, 2), 10.0 if index % 2 else 0.0), 0.02, torch.tensor([True]))
        assert not event.any()


def test_variable_clock_configuration_preserves_original_objective_and_sampling():
    env = SimpleNamespace(
        gait=SimpleNamespace(gait_cycle=0.8, air_ratio=0.5), rewards={"gait": 3.0}, commands={"ranges": [-1, 1]}
    )
    original_env = copy.deepcopy(env)
    cfg = make_bfm_train_cfg(base_cfg(), use_phase=True)
    original_cfg = copy.deepcopy(cfg)
    profile = BackwardFastGait()
    configure_backward_fast_gait(env, cfg, profile)
    assert env == original_env
    assert cfg["bfm_gait_profile"] == profile.metadata()
    assert cfg["policy"]["project_log_std"] is True
    for key, value in original_cfg.items():
        if key not in ("policy", "run_name"):
            assert cfg[key] == value
    policy = cfg["policy"].copy()
    policy.pop("project_log_std")
    assert policy == original_cfg["policy"]
    with pytest.raises(ValueError, match="phase"):
        configure_backward_fast_gait(env, make_bfm_train_cfg(base_cfg(), use_phase=False), profile)


def test_schema_four_resume_requires_identical_clock_contract(monkeypatch, tmp_path):
    import rsl_rl.runners.amp_on_policy_runner as runner_module

    monkeypatch.setattr(runner_module, "AMPLoaderPi22", lambda *args, **kwargs: ExpertTransitions())
    monkeypatch.setattr(runner_module, "store_code_state", lambda *args, **kwargs: [])
    frozen = FakeFrozenActor()
    env = BFMLatentEnv(with_terminal_amp_capture(FakePiEnv)(), frozen, contract(), use_phase=True)
    cfg = make_bfm_train_cfg(base_cfg(), use_phase=True)
    configure_backward_fast_gait(env.cfg, cfg, BackwardFastGait())
    runner = BFMOnPolicyRunner(env, cfg, device="cpu")
    assert runner.bfm_metadata["schema"] == 4
    assert runner.alg.policy.actor[0].in_features == 365
    path = tmp_path / "variable.pt"
    runner.save(path)
    runner.load(path)
    different = make_bfm_train_cfg(base_cfg(), use_phase=True)
    configure_backward_fast_gait(env.cfg, different, BackwardFastGait(backward_period=0.6))
    with pytest.raises(ValueError, match="identity/control contract"):
        BFMOnPolicyRunner(env, different, device="cpu").load(path)
    with pytest.raises(ValueError, match="identity/control contract"):
        BFMOnPolicyRunner(env, make_bfm_train_cfg(base_cfg()), device="cpu").load(path)


class GaitProbeEnv:
    """Expose the original environment hook surface without importing Isaac Sim."""

    def __init__(self):
        self.num_envs, self.device, self.step_dt = 3, "cpu", 0.02
        self.gait_start = torch.tensor([0, 0.5, 0])
        self.phase = self.gait_start.clone()
        self.gait_phase = torch.zeros(3, 2)
        self.gait_cycle = torch.full((3,), 0.8)
        self.command_generator = SimpleNamespace(command=torch.tensor([[-0.3, 0, 0], [0, 0, 0], [0.3, 0, 0]]))
        self.robot = SimpleNamespace(
            joint_names=["l_shoulder_pitch_joint", "r_shoulder_pitch_joint"],
            data=SimpleNamespace(
                root_quat_w=torch.tensor([[1.0, 0, 0, 0]]).expand(3, -1),
                root_lin_vel_w=torch.zeros(3, 3),
                root_ang_vel_w=torch.zeros(3, 3),
                joint_vel=torch.zeros(3, 2),
            ),
        )
        self.contact_sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(3, 2, 3)))
        self.feet_cfg = SimpleNamespace(body_ids=[0, 1])

    def _calculate_gait_para(self):
        self.phase = (self.phase + self.step_dt / self.gait_cycle) % 1

    def reset(self, ids):
        self.phase[ids] = self.gait_start[ids]

    def step(self, actions):
        self._calculate_gait_para()
        return actions, torch.ones(3), torch.zeros(3, dtype=torch.bool), {}


def test_environment_hook_updates_both_leg_phases_and_preserves_commands_and_reward():
    env = with_gait_environment(GaitProbeEnv, BackwardFastGait())()
    commands = env.command_generator.command.clone()
    _, reward, done, _ = env.step(torch.zeros(3, 22))
    torch.testing.assert_close(env.phase, torch.tensor([0.04, 0, 0.025]))
    torch.testing.assert_close(env.gait_phase, torch.tensor([[0.04, 0.54], [0, 0], [0.025, 0.525]]))
    torch.testing.assert_close(env.gait_cycle, torch.tensor([0.5, 0.8, 0.8]))
    torch.testing.assert_close(env.command_generator.command, commands, rtol=0, atol=0)
    torch.testing.assert_close(reward, torch.ones(3), rtol=0, atol=0)
    assert not done.any()
    phase_before = env.phase.clone()
    env.reset(torch.tensor([0]))
    assert env.phase[0] == 0
    torch.testing.assert_close(env.phase[1:], phase_before[1:], rtol=0, atol=0)
    metrics = env.pop_gait_metrics()
    assert metrics["Gait/backward/scheduled_period"] == 0.5
    assert metrics["Gait/forward/scheduled_period"] == pytest.approx(0.8)
