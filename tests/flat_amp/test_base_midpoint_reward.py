"""CPU checks for the actual reward function without starting Isaac Sim."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def reward():
    path = Path(__file__).resolve().parents[2] / "humanoidverse/flat_amp/_vendor/legged_lab/mdp/rewards.py"
    tree = ast.parse(path.read_text())
    names = {"base_com_feet_midpoint_exp", "_yaw_from_quat_wxyz"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    scope = {"torch": torch, "math": math}
    exec(compile(module, str(path), "exec"), scope)
    return scope["base_com_feet_midpoint_exp"]


def make_env(offsets):
    # Deliberately nonconsecutive IDs and unequal foot heights (one swinging).
    link = torch.zeros(len(offsets), 5, 3, dtype=torch.float64)
    link[:, 3] = torch.tensor([0.15, 0.10, 0.12])
    link[:, 1] = torch.tensor([-0.15, -0.10, 0.03])
    link[:, 4] = torch.tensor([0.4, 0.0, 0.6])  # Wrong answer if base link origin is used.
    com = link.clone()
    com[:, 4, :2] = torch.tensor(offsets, dtype=torch.float64)
    com[:, [1, 3], 0] += 0.2  # Foot COMs must not replace foot link origins.
    quats = torch.zeros(len(offsets), 5, 4, dtype=torch.float64)
    quats[:, :, 0] = 1.0
    data = SimpleNamespace(body_com_pos_w=com, body_link_pos_w=link, body_link_quat_w=quats, root_quat_w=quats[:, 0].clone())
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
    return env, SimpleNamespace(name="robot", body_ids=[4]), SimpleNamespace(name="robot", body_ids=[3, 1])


def test_projection_uses_com_and_two_feet_with_requested_falloff(reward):
    env, base, feet = make_env([[-0.03, 0], [-0.01, 0], [0, 0], [-0.06, 0], [0.01, 0], [-0.03, 0.2]])
    actual = reward(env, 0.03, base, feet, tolerance=0.0)
    expected = torch.tensor([1.0, 0.6411803884, 0.3678794412, 0.3678794412, 0.1690133154, 1.0], dtype=torch.float64)
    torch.testing.assert_close(actual, expected)


def test_five_mm_tolerance_is_symmetric_and_decay_uses_excess(reward):
    errors = [0, 0.004, -0.004, 0.005, -0.005, 0.006, -0.006, 0.03, 0.035, -0.035]
    env, base, feet = make_env([[error - 0.03, 0] for error in errors])
    actual = reward(env, 0.03, base, feet)
    expected = torch.tensor(
        [1, 1, 1, 1, 1, 0.9988895069, 0.9988895069, 0.4993517886, 0.3678794412, 0.3678794412], dtype=torch.float64
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("tolerance", [-0.005, math.nan, math.inf])
def test_invalid_tolerance(reward, tolerance):
    env, base, feet = make_env([[0, 0]])
    with pytest.raises(ValueError, match="tolerance"):
        reward(env, 0.03, base, feet, tolerance=tolerance)


def test_horizontal_rotation_translation_foot_order_and_height_invariance(reward):
    env, base, feet = make_env([[0.01, -0.02], [-0.02, 0.01]])
    expected = reward(env, 0.02, base, feet)
    data = env.scene["robot"].data
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    for positions in (data.body_com_pos_w, data.body_link_pos_w):
        positions[:, :, :2] = positions[:, :, :2] @ rotation.T + torch.tensor([12.0, -7.0])
        positions[:, :, 2] += torch.arange(5) * 10.0
    yaw_quat = torch.tensor([math.sqrt(0.5), 0, 0, math.sqrt(0.5)], dtype=torch.float64)
    data.body_link_quat_w[:] = yaw_quat
    data.root_quat_w[:] = yaw_quat
    feet.body_ids.reverse()
    torch.testing.assert_close(reward(env, 0.02, base, feet), expected)


def test_offset_follows_each_foot_orientation(reward):
    env, base, feet = make_env([[-0.015, 0.4]])
    # One foot pitched 90 degrees: its local x offset becomes vertical.
    # The flat foot contributes -3 cm in world x, so the midpoint is -1.5 cm.
    env.scene["robot"].data.body_link_quat_w[:, 3] = torch.tensor(
        [math.sqrt(0.5), 0, math.sqrt(0.5), 0], dtype=torch.float64
    )
    torch.testing.assert_close(reward(env, 0.03, base, feet), torch.ones(1, dtype=torch.float64))


def test_zero_offset_uses_original_link_midpoint(reward):
    env, base, feet = make_env([[0, 0.3]])
    torch.testing.assert_close(reward(env, 0.03, base, feet, foot_x_offset=0), torch.ones(1, dtype=torch.float64))


@pytest.mark.parametrize("std", [0, -0.02, math.nan, math.inf])
def test_invalid_width(reward, std):
    env, base, feet = make_env([[0, 0]])
    with pytest.raises(ValueError, match="std"):
        reward(env, std, base, feet)


@pytest.mark.parametrize("base_ids,feet_ids", [([4, 0], [3, 1]), ([4], [3])])
def test_invalid_body_selection(reward, base_ids, feet_ids):
    env, base, feet = make_env([[0, 0]])
    base.body_ids, feet.body_ids = base_ids, feet_ids
    with pytest.raises(ValueError, match="one base body and two foot bodies"):
        reward(env, 0.02, base, feet)
