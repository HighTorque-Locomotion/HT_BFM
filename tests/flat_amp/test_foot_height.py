"""CPU regression checks for the HT_lab swing-height shortfall port."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from humanoidverse.flat_amp.foot_height import FootHeightShortfallEvents, SoleClearance


def update(events, phase, height, eligible=True):
    phase = torch.as_tensor(phase, dtype=torch.float32).expand(events.seen.shape)
    height = torch.as_tensor(height, dtype=torch.float32).expand_as(phase)
    eligible = torch.as_tensor(eligible, dtype=torch.bool).expand(phase.shape[0])
    return events.update(height, phase, None, eligible)


def test_shortfall_settles_once_and_rearms_next_cycle():
    events = FootHeightShortfallEvents(1, "cpu")
    assert not update(events, 0.1, 0.0).any()
    assert not update(events, 0.45, 0.0169).any()
    assert update(events, 0.46, 0.0).all()
    assert not update(events, 0.8, 0.0).any()
    assert not update(events, 0.1, 0.0).any()
    assert update(events, 0.46, 0.0).all()


def test_peak_inside_window_qualifies_independently_for_each_foot():
    events = FootHeightShortfallEvents(1, "cpu")
    update(events, [0.1, 0.6], [0.017, 0.0])
    assert not update(events, [0.46, 0.96], 0.0).any()
    update(events, [0.6, 0.1], 0.0)
    assert update(events, [0.96, 0.46], 0.0).tolist() == [[False, True]]


@pytest.mark.parametrize("outside", [0.09, 0.46])
def test_height_outside_window_does_not_qualify(outside):
    events = FootHeightShortfallEvents(1, "cpu")
    if outside < 0.1:
        update(events, outside, 0.1)
    update(events, 0.3, 0.0)
    assert update(events, 0.46, 0.1 if outside > 0.45 else 0.0).all()


def test_no_window_no_penalty_and_standing_discards_pending_window():
    events = FootHeightShortfallEvents(1, "cpu")
    assert not update(events, 0.7, 0.0).any()
    update(events, 0.1, 0.0)
    update(events, 0.2, 0.0, eligible=False)
    assert not update(events, 0.46, 0.0).any()


def test_partial_reset_preserves_other_environment():
    events = FootHeightShortfallEvents(2, "cpu")
    update(events, 0.3, 0.0)
    events.reset(torch.tensor([0]))
    assert update(events, 0.46, 0.0).tolist() == [[False, False], [True, True]]


@pytest.mark.parametrize("kwargs", [{"minimum_height": 0}, {"minimum_height": math.nan}, {"phase_window": (0.1, 0.5)}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        FootHeightShortfallEvents(1, "cpu", **kwargs)


def test_clearance_uses_lowest_collision_surface_and_foot_rotation():
    transform = torch.eye(4).tolist()
    cube = {"type": "Cube", "transform": transform, "size": 0.04}
    capsule = {"type": "Capsule", "transform": transform, "height": 0.06, "radius": 0.01, "axis": "X"}
    geometry = SoleClearance([[cube, capsule], [cube, capsule]], "cpu")
    positions = torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]])
    # Flat foot: cube extends 2 cm below the origin. Pitched foot: capsule 4 cm.
    quaternions = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]]])
    torch.testing.assert_close(geometry(positions, quaternions, torch.tensor([0.01])), torch.tensor([[0.07, 0.05]]))


def load_adapter():
    """Load the real adapter without starting Isaac Sim for manager imports."""
    path = Path(__file__).resolve().parents[2] / "humanoidverse/flat_amp/_vendor/legged_lab/mdp/rewards.py"
    tree = ast.parse(path.read_text())
    names = {"SwingHeightShortfallPenalty", "command_activity_masks"}
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]

    class ManagerTermBase:
        def __init__(self, cfg, env):
            pass

    scope = dict(torch=torch, ManagerTermBase=ManagerTermBase, FootHeightShortfallEvents=FootHeightShortfallEvents)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["SwingHeightShortfallPenalty"], scope["command_activity_masks"]


def test_all_moving_command_directions_are_eligible():
    _, masks = load_adapter()
    commands = torch.tensor([[0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1], [0.08, 0.08, 0.2], [0.05, 0.05, 0.09]])
    assert masks(commands)[2].tolist() == [True, True, True, True, True, False]


def test_adapter_collects_substep_peak_cancels_dt_and_handles_mode_reset():
    adapter_type, _ = load_adapter()
    adapter = object.__new__(adapter_type)
    adapter.events = FootHeightShortfallEvents(2, "cpu")
    adapter.lin_threshold = adapter.yaw_threshold = 0.1
    adapter.step_events = torch.zeros(2, 2, dtype=torch.bool)
    adapter.last_events = torch.zeros_like(adapter.step_events)
    adapter.last_clearance = torch.zeros(2, 2)
    adapter.previous_moving = torch.full((2,), -1, dtype=torch.long)
    adapter.offsets = torch.tensor([0.0, 0.5])
    adapter.feet_ids = [1, 0]
    height = torch.zeros(2, 2)
    adapter.geometry = lambda positions, quaternions, ground_z: height
    env = SimpleNamespace(
        cfg=SimpleNamespace(gait=SimpleNamespace(gait_cycle=0.8)),
        command_generator=SimpleNamespace(command=torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.0, 0.2]])),
        phase_length_buf=torch.tensor([17, 17]), gait_start=torch.zeros(2), step_dt=0.02, physics_dt=0.005,
        robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=torch.zeros(2, 2, 3), body_quat_w=torch.zeros(2, 2, 4))),
        scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
    )
    adapter.begin_step(env)
    # A single physical substep exceeds the threshold, then returns to ground.
    height[0, 0] = 0.018
    adapter.record_substep(env, 0)
    height.zero_()
    for substep in [1, 2, 3]:
        adapter.record_substep(env, substep)
    env.phase_length_buf += 1
    adapter.begin_step(env)
    adapter.record_substep(env, 0)
    value = adapter(env, None, None, None, None)
    torch.testing.assert_close(-value * env.step_dt, torch.tensor([0.0, -1.0]))
    assert not adapter(env, None, None, None, None).any()
    env.phase_length_buf[:] = 4
    adapter.begin_step(env)
    adapter.record_substep(env, 0)
    env.command_generator.command.zero_()
    adapter.begin_step(env)
    assert not adapter.events.seen.any()
    adapter.reset(torch.tensor([0]))
    assert adapter.previous_moving.tolist() == [-1, 0]
