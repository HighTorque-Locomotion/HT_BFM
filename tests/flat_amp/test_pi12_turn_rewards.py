"""Exercise reward and termination math without starting Isaac Sim."""

import ast
import math
from types import SimpleNamespace

import pytest
import torch

from humanoidverse.flat_amp import VENDOR_ROOT


def quat_from_rpy(roll, pitch, yaw):
    cr, cp, cy = (math.cos(a / 2) for a in (roll, pitch, yaw))
    sr, sp, sy = (math.sin(a / 2) for a in (roll, pitch, yaw))
    return torch.tensor(
        [[cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]],
        dtype=torch.float64,
    )


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_apply(q, v):
    t = 2 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


@pytest.fixture(scope="module")
def functions():
    # Follow the existing Pi12 tests: load real tensor functions, stub only Isaac entities.
    path = VENDOR_ROOT / "legged_lab/mdp/rewards.py"
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)]
    env_path = VENDOR_ROOT / "legged_lab/envs/pi/pi_12dof_env.py"
    cls = next(n for n in ast.parse(env_path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "PiDof12Env")
    nodes.append(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "check_reset"))
    command_path = VENDOR_ROOT / "legged_lab/mdp/commands.py"
    command_cls = next(n for n in ast.parse(command_path.read_text()).body if isinstance(n, ast.ClassDef))
    nodes.extend(n for n in command_cls.body if isinstance(n, ast.FunctionDef) and n.name in ("_resample_command", "_update_command"))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {
        "torch": torch,
        "math": math,
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name, joint_ids=slice(None), body_ids=[0]),
        "math_utils": SimpleNamespace(
            quat_apply=quat_apply,
            quat_conjugate=quat_conjugate,
            quat_apply_inverse=lambda q, v: quat_apply(quat_conjugate(q), v),
            wrap_to_pi=lambda angle: (angle + math.pi) % (2 * math.pi) - math.pi,
        ),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("rpy", [(0, 0, 0), (0.2, 0, 0), (0, -0.3, 0), (0, 0, 2.5), (-0.4, 0.3, -2.8)])
def test_wxyz_signed_euler_and_quaternion_sign(functions, rpy):
    q = quat_from_rpy(*rpy)
    expected = torch.tensor([rpy], dtype=q.dtype)
    torch.testing.assert_close(functions.get_euler_xyz_tensor(q), expected)
    torch.testing.assert_close(functions.get_euler_xyz_tensor(-q), expected)


@pytest.mark.parametrize("roll,pitch", [(0, 0), (0.2, 0), (0, -0.2), (0.2, -0.3)])
def test_orientation_reward_is_heading_invariant(functions, roll, pitch):
    scores = []
    for yaw in (0, 1.5, -2.8):
        data = SimpleNamespace(
            body_quat_w=quat_from_rpy(roll, pitch, yaw).unsqueeze(1), GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
        )
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
        scores.append(functions.body_orientation_euler(env))
    for score in scores[1:]:
        torch.testing.assert_close(score, scores[0])
    assert scores[0].item() == pytest.approx(1.0) if roll == pitch == 0 else scores[0].item() < 1.0


def tilt_env(command, roll=0.0, pitch=0.0, yaw=0.0):
    return SimpleNamespace(
        command_generator=SimpleNamespace(command=torch.tensor([command], dtype=torch.float64)),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(body_quat_w=quat_from_rpy(roll, pitch, yaw).unsqueeze(1)))},
    )


@pytest.mark.parametrize(
    "command,roll,pitch",
    [
        ([0, 0, 0], 0.0, 0.0),
        ([1, 0, 0], 0.0, 0.0),
        ([-0.6, 0, 0], 0.0, 0.0),
        ([0, 0.5, 0], 0.0, 0.0),
        ([0, 0, 1], 0.0, 0.0),
        ([1, 0, 1], -math.atan(1 / 9.81), 0.0),
        ([1, 0, -1], math.atan(1 / 9.81), 0.0),
        ([-0.6, 0, 1], math.atan(0.6 / 9.81), 0.0),
        ([0, 0.5, 1], 0.0, -math.atan(0.5 / 9.81)),
        # Both horizontal axes accelerate: independently precomputed reference angles in radians.
        ([1, 0.5, 1], -0.10145408307397486, -0.05092436798528353),
    ],
)
def test_command_tilt_reference_and_heading_invariance(functions, command, roll, pitch):
    for yaw in (0.0, 1.4, -2.8):
        env = tilt_env(command, roll, pitch, yaw)
        reward = functions.track_command_tilt_exp(env, std=math.radians(5), tolerance=0.0)
        assert reward.item() == pytest.approx(1.0, abs=1e-8)
    if command == [1, 0, 1]:
        # Turning left should prefer leaning left, not the equal-and-opposite roll.
        wrong_direction = functions.track_command_tilt_exp(tilt_env(command, -roll), std=math.radians(5))
        assert wrong_direction.item() < 0.3


@pytest.mark.parametrize(
    "roll_error,pitch_error,expected",
    [
        (0, 0, 1.0),
        (3, 0, 1.0),
        (-3, 3, 1.0),
        (5, 0, math.exp(-0.08)),
        (8, 0, math.exp(-0.5)),
        (0, -8, math.exp(-0.5)),
        (13, 0, math.exp(-2)),
        (8, -8, math.exp(-1)),
    ],
)
def test_command_tilt_deadband_and_gaussian_std(functions, roll_error, pitch_error, expected):
    env = tilt_env([1, 0, 0], math.radians(roll_error), math.radians(pitch_error))
    reward = functions.track_command_tilt_exp(env, std=math.radians(5))
    assert reward.item() == pytest.approx(expected)


def test_command_tilt_is_batched_and_has_no_command_derivative(functions):
    env = tilt_env([0, 0, 0])
    env.scene["robot"].data.body_quat_w = env.scene["robot"].data.body_quat_w.repeat(3, 1, 1)
    commands = torch.tensor([[1.0, 0.5, 1.57], [-0.6, -0.5, -1.57], [0.0, 0.0, 0.0]], dtype=torch.float64)
    env.command_generator.command = commands
    initial = functions.track_command_tilt_exp(env, std=math.radians(5))
    for command in (torch.zeros_like(commands), -commands, commands):
        env.command_generator.command = command
        reward = functions.track_command_tilt_exp(env, std=math.radians(5))
        assert reward.shape == (3,)
        assert torch.isfinite(reward).all() and ((reward >= 0) & (reward <= 1)).all()
    torch.testing.assert_close(reward, initial)


@pytest.mark.parametrize("parameter,value", [("std", 0), ("std", float("nan")), ("tolerance", -1), ("gravity", 0)])
def test_command_tilt_rejects_invalid_parameters(functions, parameter, value):
    params = {"std": math.radians(5), parameter: value}
    with pytest.raises(ValueError, match=parameter):
        functions.track_command_tilt_exp(tilt_env([0, 0, 0]), **params)


@pytest.mark.parametrize(
    "command,hip_active,feet_active",
    [
        ([0, 0, 0], False, True),
        ([0.05, 0.05, 0.05], False, True),
        ([0.09, 0.09, 0], False, False),  # Norm exceeds the standing deadband.
        ([0.1, 0, 0], True, True),
        ([0.5, 0.099, 0.099], True, True),
        ([0.5, 0, 0.1], False, False),
        ([0.5, 0, -0.1], False, False),
        ([0.5, 0.1, 0], False, False),
        ([0.5, -0.1, 0], False, False),
        ([-0.5, 0, 0], True, True),
        ([-0.1, 0, 0], True, True),
        ([-0.099, 0, 0], False, True),
        ([-0.5, 0, 0.1], False, False),
        ([-0.5, -0.1, 0], False, False),
        ([0, 0.3, 0], False, False),
        ([0, 0, 0.5], False, False),
    ],
)
def test_joint_and_foot_penalties_use_command_modes(functions, command, hip_active, feet_active):
    data = SimpleNamespace(
        joint_pos=torch.tensor([[0.2, -0.3]]),
        default_joint_pos=torch.tensor([[0.1, -0.1]]),
        body_pos_w=torch.tensor([[[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]]),
        root_link_pos_w=torch.zeros(1, 3),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    robot = SimpleNamespace(data=data)
    env = SimpleNamespace(
        scene={"robot": robot},
        robot=robot,
        feet_body_ids=[0, 1],
        command_generator=SimpleNamespace(command=torch.tensor([command], dtype=torch.float32)),
    )
    hip = functions.joint_deviation_l2(env, lin_command_threshold=0.1, yaw_command_threshold=0.1)
    feet = functions.feet_y_distance(env, lin_command_threshold=0.1, yaw_command_threshold=0.1)
    assert hip.item() == pytest.approx(0.05 if hip_active else 0.0)
    assert feet.item() == pytest.approx(0.04 if feet_active else 0.0)


@pytest.mark.parametrize("heading_command", [True, False])
def test_command_groups_and_straight_override(functions, monkeypatch, heading_command):
    generator = SimpleNamespace(
        device="cpu",
        cfg=SimpleNamespace(
            heading_command=heading_command,
            rel_heading_envs=1.0,
            heading_control_stiffness=0.5,
            rel_standing_envs=0.1,
            rel_onestep_envs=0.15,
            rel_turn_only_envs=0.15,
            rel_straight_envs=0.15,
            ranges=SimpleNamespace(lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)),
        ),
        vel_command_b=torch.zeros(100, 3),
        heading_target=torch.zeros(100),
        robot=SimpleNamespace(data=SimpleNamespace(heading_w=torch.zeros(100))),
    )
    group_names = ("is_standing_env", "is_onestep_env", "is_turn_only_env", "is_straight_env")
    for flag in (*group_names, "is_heading_env"):
        setattr(generator, flag, torch.zeros(100, dtype=torch.bool))
    # Stratify only the group draws so exact proportions can be checked without flaky statistical assertions.
    monkeypatch.setattr(torch, "rand", lambda count, device: (torch.arange(count, device=device) + 0.5) / count)
    functions._resample_command(generator, torch.arange(100))
    groups = torch.stack([getattr(generator, flag) for flag in group_names])
    assert groups.sum(dim=1).tolist() == [10, 15, 15, 15]
    assert (groups.sum(dim=0) <= 1).all()
    assert (~groups.any(dim=0)).sum().item() == 45
    original = generator.vel_command_b.clone()
    assert ((original[:, 0] >= -0.6) & (original[:, 0] <= 1.0)).all()
    assert (original[:, 1].abs() <= 0.5).all()
    assert (original[:, 2].abs() <= 1.57).all()
    for heading in (0.0, 2.0):
        generator.robot.data.heading_w.fill_(heading)
        functions._update_command(generator)
        command = generator.vel_command_b
        assert (command[generator.is_standing_env] == 0).all()
        assert (command[generator.is_onestep_env][:, [0, 2]] == 0).all()
        assert (command[generator.is_turn_only_env, :2] == 0).all()
        assert (command[generator.is_straight_env, 1:] == 0).all()
        torch.testing.assert_close(command[generator.is_straight_env, 0], original[generator.is_straight_env, 0])
        torch.testing.assert_close(command[generator.is_onestep_env, 1], original[generator.is_onestep_env, 1])
        torch.testing.assert_close(command[~groups.any(dim=0), :2], original[~groups.any(dim=0), :2])
        assert (command[:, 2].abs() <= 1.57).all()
    # Partial reset changes only the requested environments, including stale straight-group flags.
    before = generator.vel_command_b.clone()
    generator.is_straight_env[:10] = True
    functions._resample_command(generator, torch.arange(10))
    torch.testing.assert_close(generator.vel_command_b[10:], before[10:])
    assert not generator.is_straight_env[0]


def termination_env(q):
    n = len(q)
    return SimpleNamespace(
        contact_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w_history=torch.zeros(n, 1, 1, 3))),
        termination_contact_cfg=SimpleNamespace(body_ids=[0]),
        cfg=SimpleNamespace(robot=SimpleNamespace(max_tilt_angle_rad=None, max_tilt_angle_deg=45.0)),
        robot=SimpleNamespace(data=SimpleNamespace(
            projected_gravity_b=quat_apply(quat_conjugate(q), q.new_tensor([0.0, 0.0, -1.0]).expand(n, 3)),
            root_pos_w=torch.tensor([[0.0, 0.0, 0.4]]).repeat(n, 1),
        )),
        scene=SimpleNamespace(env_origins=torch.zeros(n, 3)),
        is_on_forbidden=torch.zeros(n, dtype=torch.bool),
        episode_length_buf=torch.zeros(n),
        max_episode_length=1000,
        reset_reason_buf={
            name: torch.zeros(n, dtype=torch.bool)
            for name in ("contact_excess", "bad_orientation", "bad_height", "time_out")
        },
    )


@pytest.mark.parametrize(
    "rpy,bad_tilt",
    [
        ((0, 0, 2.8), False),
        ((0, 0, -2.8), False),
        ((0.3, -0.2, 2.8), False),
        ((0.81, 0, 0), True),
        ((-0.81, 0, 0), True),
        ((0, 1.01, 0), True),
        ((0, -1.01, 0), True),
        ((0.79, 0, 0), True),
        ((math.pi / 4 - 1e-6, 0, 0), False),
        ((math.pi / 4 + 1e-6, 0, 0), True),
        ((0.6, 0.6, 0), True),
    ],
)
def test_termination_checks_total_tilt_not_heading(functions, rpy, bad_tilt):
    env = termination_env(quat_from_rpy(*rpy))
    reset, timeout = functions.check_reset(env)
    assert env.reset_reason_buf["bad_orientation"].item() == bad_tilt
    assert reset.item() == bad_tilt
    assert not timeout.item()
    # Equality does not terminate; raising the limit no longer applies hidden Euler limits.
    env.cfg.robot.max_tilt_angle_rad = torch.acos(-env.robot.data.projected_gravity_b[0, 2]).abs().item()
    assert not functions.check_reset(env)[0].item()
    env.cfg.robot.max_tilt_angle_rad = math.pi
    assert not functions.check_reset(env)[0].item()


def test_total_tilt_preserves_previous_default_decisions(functions):
    q = torch.randn(10000, 4, generator=torch.Generator().manual_seed(20260928), dtype=torch.float64)
    q /= q.norm(dim=-1, keepdim=True)
    env = termination_env(q)
    angles = functions.get_euler_xyz_tensor(q)
    tilt = torch.acos(-env.robot.data.projected_gravity_b[:, 2]).abs()
    previous = (tilt > math.pi / 4) | (angles[:, 0].abs() > 0.8) | (angles[:, 1].abs() > 1.0)
    actual, timeout = functions.check_reset(env)
    torch.testing.assert_close(actual, previous)
    assert not timeout.any()
