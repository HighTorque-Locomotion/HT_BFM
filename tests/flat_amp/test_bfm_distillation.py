"""Historical command-routed dual teachers, warm start and real PPO updates on CPU."""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from legged_lab.utils.bfm_bridge import BFMLatentEnv, with_terminal_amp_capture
from legged_lab.utils.bfm_distillation import DistillationSchedule
from legged_lab.utils.bfm_gait import BackwardFastGait
from legged_lab.utils.bfm_training import BFMOnPolicyRunner, configure_backward_fast_gait, make_bfm_train_cfg
from test_bfm_bridge import ExpertTransitions, FakeFrozenActor, FakePiEnv, base_cfg, contract


def physical_env(use_phase=True):
    return BFMLatentEnv(with_terminal_amp_capture(FakePiEnv)(), FakeFrozenActor(), contract(), use_phase=use_phase)


@pytest.fixture
def teachers(monkeypatch, tmp_path):
    import rsl_rl.runners.amp_on_policy_runner as runner_module

    monkeypatch.setattr(runner_module, "AMPLoaderPi22", lambda *args, **kwargs: ExpertTransitions())
    monkeypatch.setattr(runner_module, "store_code_state", lambda *args, **kwargs: [])
    result = {}
    for use_phase in (False, True):
        torch.manual_seed(8 if use_phase else 9)
        runner = BFMOnPolicyRunner(
            physical_env(use_phase), make_bfm_train_cfg(base_cfg(), use_phase=use_phase), device="cpu"
        )
        runner.current_learning_iteration = 49999
        path = tmp_path / ("phase.pt" if use_phase else "no_phase.pt")
        runner.save(path)
        result["phase_teacher" if use_phase else "no_phase_teacher"] = str(path)
    return result


def student(teachers, **schedule):
    env = physical_env()
    cfg = make_bfm_train_cfg(base_cfg())
    configure_backward_fast_gait(env.cfg, cfg, BackwardFastGait())
    if teachers is not None:
        cfg["bfm_distillation"] = {**teachers, "schedule": schedule}
    return BFMOnPolicyRunner(env, cfg, device="cpu")


def test_historical_coefficient_schedule_and_invalid_settings():
    schedule = DistillationSchedule()
    assert schedule.weight(0) == schedule.weight(1000) == 5
    assert schedule.weight(6000) == 2.75
    assert schedule.weight(11000) == schedule.weight(99999) == 0.5
    for settings in (
        {"coefficient": 0},
        {"initial_raw_norm": 0},
        {"decay_iterations": 0},
        {"coefficient": float("nan")},
        {"final_coefficient": 6},
    ):
        with pytest.raises(ValueError):
            DistillationSchedule(**settings)


def test_routes_each_command_to_one_normalized_teacher(teachers):
    runner = student(teachers)
    obs = torch.randn(7, 365)
    obs[:, :3] = torch.tensor(
        [[-0.3, 0, 0], [-0.3, 0.2, 0.6], [0, 0, 0], [0.3, 0, 0], [0, 0.25, 0], [0, -0.25, 0], [0, 0, 0.6]]
    )
    target, groups = runner.distiller.targets(obs)
    with torch.no_grad():
        no_phase = F.normalize(runner.distiller.no_phase_actor(obs[:, :363]), dim=-1)
        phase = F.normalize(runner.distiller.phase_actor(obs), dim=-1)
    torch.testing.assert_close(target[:3], no_phase[:3], rtol=0, atol=0)
    torch.testing.assert_close(target[3:], phase[3:], rtol=0, atol=0)
    torch.testing.assert_close(target.norm(dim=-1), torch.ones(7))
    assert groups["backward"].sum() == 2
    assert groups["standing"].sum() == 1
    assert groups["phase_motion"].sum() == 4
    mean = (target * 37).clone().requires_grad_(True)
    loss, stats = runner.distiller(obs, mean, 0)
    assert loss.item() < 1e-6
    loss.backward()
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in runner.distiller.phase_actor.parameters())
    assert stats["coefficient"] == 5
    with pytest.raises(ValueError, match="365"):
        runner.distiller.targets(obs[:, :363])


def test_thresholds_use_scaled_commands_not_the_current_physical_state(teachers):
    runner = student(teachers)
    runner.distiller.command_scale = 2.5
    commands = torch.tensor(
        [
            [-0.1, 0.3, 0.8],  # Inclusive backward threshold, even with side/yaw commands.
            [-0.099, 0, 0],  # Below moving threshold: standing.
            [0, 0, 0],
            [0.1, 0, 0],  # Inclusive moving threshold.
            [0, 0, 0.1],  # Inclusive yaw threshold.
            [0.08, 0.08, 0],  # Euclidean planar norm exceeds the threshold.
        ]
    )
    obs = torch.full((len(commands), 365), 100.0)
    obs[:, :3] = commands * runner.distiller.command_scale
    groups = runner.distiller.groups(obs)
    assert groups["backward"].tolist() == [True, False, False, False, False, False]
    assert groups["standing"].tolist() == [False, True, True, False, False, False]
    assert groups["phase_motion"].tolist() == [False, False, False, True, True, True]
    obs[:, 3:] = -100
    for name, mask in runner.distiller.groups(obs).items():
        torch.testing.assert_close(mask, groups[name], rtol=0, atol=0)


def test_group_means_are_equally_weighted_and_empty_groups_are_ignored(teachers):
    distiller = student(teachers).distiller
    obs = torch.randn(12, 365)
    obs[:, :3] = torch.tensor([[0.3, 0, 0]])
    obs[0, :3] = torch.tensor([-0.3, 0, 0])
    obs[1:4, :3] = 0
    target, _ = distiller.targets(obs)
    mean = target.clone()
    mean[0] *= -1  # backward loss 2
    # Eight phase-motion samples with cosine zero; three standing samples remain correct.
    direction = torch.zeros_like(target[4:])
    direction[:, 0] = 1
    mean[4:] = F.normalize(direction - (direction * target[4:]).sum(-1, keepdim=True) * target[4:], dim=-1)
    mean.requires_grad_(True)
    loss, stats = distiller(obs, mean, 0)
    assert stats["cosine_backward"].item() == pytest.approx(2)
    assert stats["cosine_standing"].item() == pytest.approx(0, abs=1e-6)
    assert stats["cosine_phase_motion"].item() == pytest.approx(1, abs=1e-6)
    assert stats["cosine"].item() == pytest.approx(1, abs=1e-6)
    assert loss.item() == pytest.approx(5, abs=1e-5)
    assert stats["cosine"].item() != pytest.approx(10 / 12)
    loss.backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    # An absent motion group contributes no extra zero term to the denominator.
    loss, stats = distiller(obs[:4], mean[:4], 11000)
    assert stats["fraction_phase_motion"] == 0
    assert stats["cosine"].item() == pytest.approx(1, abs=1e-6)
    assert loss.item() == pytest.approx(0.5, abs=1e-6)


def test_no_phase_warm_start_zeros_added_columns_and_starts_fresh(teachers):
    runner = student(teachers)
    checkpoint = torch.load(teachers["no_phase_teacher"], weights_only=False)
    info = runner.initialize_from_no_phase_teacher()
    obs = torch.randn(8, 365)
    with torch.no_grad():
        teacher = runner.distiller.no_phase_actor(obs[:, :363])
        actual = runner.alg.policy.act_inference(obs)
        torch.testing.assert_close(F.normalize(actual, dim=-1), F.normalize(teacher, dim=-1), atol=1e-6, rtol=1e-5)
        runner.alg.policy.update_distribution(obs)
        torch.testing.assert_close(runner.alg.policy.action_std, torch.full((8, 256), math.exp(-1.5)))
    assert not runner.alg.policy.actor[0].weight[:, -2:].count_nonzero()
    assert not runner.alg.policy.critic[0].weight[:, -2:].count_nonzero()
    torch.testing.assert_close(runner.alg.policy.critic[0].weight[:, :-2], checkpoint["model_state_dict"]["critic.0.weight"])
    for key, value in runner.alg.policy.critic.state_dict().items():
        if key != "0.weight":
            torch.testing.assert_close(value, checkpoint["model_state_dict"][f"critic.{key}"], rtol=0, atol=0)
    for key, value in runner.alg.discriminator.state_dict().items():
        torch.testing.assert_close(value, checkpoint["discriminator_state_dict"][key], rtol=0, atol=0)
    for name in ("mean", "var", "count"):
        torch.testing.assert_close(
            torch.as_tensor(getattr(runner.alg.amp_normalizer, name)),
            torch.as_tensor(getattr(checkpoint["amp_normalizer"], name)),
            rtol=0,
            atol=0,
        )
    assert runner.current_learning_iteration == 0 and not runner.alg.optimizer.state
    assert info["projected_direction_max_abs_error"] < 1e-5
    assert info["mode"] == "teacher_warm_start_not_resume"
    assert info["initial_raw_norm_target"] == 16


def test_real_ppo_updates_student_phase_columns_and_keeps_all_teachers_frozen(teachers, tmp_path):
    runner = student(teachers)
    runner.initialize_from_no_phase_teacher()
    old_weights = runner.alg.policy.actor[0].weight.detach().clone()
    teacher_actors = (runner.distiller.no_phase_actor, runner.distiller.phase_actor)
    teacher_states = [copy.deepcopy(teacher.state_dict()) for teacher in teacher_actors]
    frozen = copy.deepcopy(runner.env.actor.model.state_dict())
    observed_losses = []
    real_update = runner.alg.update

    def update():
        result = real_update()
        observed_losses.append(result)
        return result

    runner.alg.update = update
    runner.learn(2, init_at_random_ep_len=True)
    assert all(row["distill/weighted"] > 0 and row["distill/coefficient"] == 5 for row in observed_losses)
    assert not torch.equal(old_weights[:, -2:], runner.alg.policy.actor[0].weight[:, -2:])
    for teacher, state in zip(teacher_actors, teacher_states):
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters())
    for key, value in runner.env.actor.model.state_dict().items():
        torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in runner.env.actor.model.parameters())
    runner.distiller.assert_frozen(runner.alg.optimizer)
    with pytest.raises(ValueError, match="fresh optimizer"):
        runner.distiller.initialize_student(runner)
    path = tmp_path / "distilled.pt"
    runner.save(path)
    restored = student(teachers)
    restored.load(path)
    assert restored.current_learning_iteration == runner.current_learning_iteration
    assert restored.distillation_initialization == runner.distillation_initialization
    assert restored.alg.optimizer.state
    with pytest.raises(ValueError, match="identical teacher identities and schedule"):
        student(teachers, coefficient=3).load(path)
    plain = student(None)
    with pytest.raises(ValueError, match="identical teacher identities and schedule"):
        plain.load(path)
    plain.load(path, load_optimizer=False)  # A deployed student never needs teacher files.
    with torch.inference_mode():
        obs = torch.randn(3, 365)
        torch.testing.assert_close(plain.alg.policy.act_inference(obs), runner.alg.policy.act_inference(obs))


def test_resume_rejects_changed_teacher_identity_even_with_matching_architecture(teachers, tmp_path):
    runner = student(teachers)
    path = tmp_path / "student.pt"
    runner.save(path)
    teacher = torch.load(teachers["phase_teacher"], weights_only=False)
    teacher["iter"] -= 1
    replacement = tmp_path / "different_phase.pt"
    torch.save(teacher, replacement)
    with pytest.raises(ValueError, match="identical teacher identities and schedule"):
        student({**teachers, "phase_teacher": str(replacement)}).load(path)


def test_teacher_control_contract_mismatch_rejected(teachers, tmp_path):
    checkpoint = torch.load(teachers["no_phase_teacher"], weights_only=False)
    checkpoint["infos"]["bfm"]["actor_sha256"] = "different-frozen-bfm"
    path = tmp_path / "wrong_teacher.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="actor_sha256 differs"):
        student({**teachers, "no_phase_teacher": str(path)})
