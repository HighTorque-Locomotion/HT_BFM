import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from humanoidverse.flat_amp.__main__ import build_invocation, build_parser
from humanoidverse.flat_amp.keyboard_control import AXES, KeyboardVelocity, refresh_play_observations


def ranges():
    return SimpleNamespace(lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57))


def test_ht_lab_bindings_limits_and_zero():
    control = KeyboardVelocity(ranges(), lin_step=0.2, ang_step=0.3)
    for key in ("W", "A", "Q"):
        assert control.handle_key(key)
    assert control.command == (0.2, 0.2, 0.3)
    for key in ("S", "D", "E"):
        control.handle_key(key)
    assert control.command == (0.0, 0.0, 0.0)
    assert not control.handle_key("Z")
    for _ in range(50):
        for key in ("S", "D", "E"):
            control.handle_key(key)
    assert control.command == (-0.6, -0.5, -1.57)
    for _ in range(50):
        for key in ("W", "A", "Q"):
            control.handle_key(key)
    assert control.command == (1.0, 0.5, 1.57)
    control.handle_key("X")
    assert control.command == (0.0, 0.0, 0.0)
    assert KeyboardVelocity(ranges(), (20, -20, 20)).command == (1.0, -0.5, 1.57)


def test_command_commit_and_resampling():
    original_ranges = ranges()
    control = KeyboardVelocity(original_ranges)
    generator = SimpleNamespace(cfg=SimpleNamespace(ranges=original_ranges), command=torch.ones(3, 3))
    for flag in ("is_standing_env", "is_heading_env", "is_onestep_env", "is_turn_only_env", "is_straight_env"):
        setattr(generator, flag, torch.ones(3, dtype=torch.bool))
    env = SimpleNamespace(command_generator=generator)
    assert control.apply(env)
    assert not control.apply(env)
    control.handle_key("W")
    control.handle_key("A")
    control.handle_key("Q")
    assert generator.command.count_nonzero() == 0  # Events do not mutate tensors inside render().
    assert control.apply(env)
    torch.testing.assert_close(generator.command, torch.full((3, 3), 0.1))
    for flag in ("is_standing_env", "is_heading_env", "is_onestep_env", "is_turn_only_env", "is_straight_env"):
        assert not getattr(generator, flag).any()
    for i, axis in enumerate(AXES):
        # Timer and reset use these same ranges, so both retain the manual target.
        generator.command[:, i].uniform_(*getattr(generator.cfg.ranges, axis))
    torch.testing.assert_close(generator.command, torch.full((3, 3), 0.1))
    assert control.limits == ((-0.6, 1.0), (-0.5, 0.5), (-1.57, 1.57))


@pytest.mark.parametrize("frame_dim", [47])
def test_refresh_latest_frame_only(frame_dim):
    actor = torch.randn(2, frame_dim)
    critic = torch.cat([actor, torch.randn(2, 5)], dim=-1)
    phase_calls = []
    env = SimpleNamespace(
        clip_obs=1.0, _calculate_gait_para=lambda: phase_calls.append(True),
        compute_current_observations=lambda: (actor, critic),
    )
    history = torch.randn(2, 10 * frame_dim)
    before = history.clone()
    critic_history = torch.randn(2, 3 * (frame_dim + 5))
    critic_before = critic_history.clone()
    obs = {"critic": critic_history}
    refresh_play_observations(env, history, obs)
    assert phase_calls == [True]
    torch.testing.assert_close(history[:, :-frame_dim], before[:, :-frame_dim])
    torch.testing.assert_close(history[:, -frame_dim:], actor.clamp(-1, 1))
    torch.testing.assert_close(critic_history[:, :-critic.shape[1]], critic_before[:, :-critic.shape[1]])
    torch.testing.assert_close(critic_history[:, -critic.shape[1]:], critic.clamp(-1, 1))
    torch.testing.assert_close(obs["current_actor_obs_add_nosie"], actor)
    assert obs["obs_history"] is history


def test_keyboard_cli_forwarding_and_fixed_mode():
    parser = build_parser()
    base = ["play", "--task", "walk_pi_waq_12", "--checkpoint", "model.pt"]
    _, options = build_invocation(parser.parse_args(base))
    assert "--keyboard-control" not in options
    for suffix in ("-", "_"):
        args = parser.parse_args(base + [
            "--keyboard" + suffix + "control", "--keyboard" + suffix + "linvel" + suffix + "step", "0.2",
            "--keyboard" + suffix + "angvel" + suffix + "step", "0.3", "--velocity", "0.5", "0", "0",
        ])
        module, options = build_invocation(args)
        assert module == "legged_lab.scripts.play"
        assert "--keyboard-control" in options
        assert options[options.index("--keyboard-linvel-step") + 1] == "0.2"
        assert options[options.index("--keyboard-angvel-step") + 1] == "0.3"
        assert options[options.index("--velocity") + 1:options.index("--velocity") + 4] == ["0.5", "0.0", "0.0"]
    with pytest.raises(ValueError, match="GUI"):
        build_invocation(parser.parse_args(base + ["--keyboard-control", "--headless"]))
    with pytest.raises(SystemExit):
        build_invocation(parser.parse_args(base + ["--keyboard-control", "--controller", "bfm"]))


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf"])
def test_reject_invalid_steps(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["play", "--checkpoint", "model.pt", "--keyboard-linvel-step", value])


def test_input_subscription_press_repeat_release_and_cleanup(monkeypatch):
    carb = ModuleType("carb")
    carb.input = ModuleType("carb.input")
    carb.input.KeyboardInput = SimpleNamespace(**{key: key for key in "WSADQEX"})
    carb.input.KeyboardEventType = SimpleNamespace(KEY_PRESS=1, KEY_REPEAT=2, KEY_RELEASE=3)
    events = {}
    cleaned = []
    keyboard = object()

    def subscribe(device, callback):
        assert device is keyboard
        events["callback"] = callback
        return 0  # Zero is a valid subscription handle.

    carb.input.acquire_input_interface = lambda: SimpleNamespace(
        subscribe_to_keyboard_events=subscribe,
        unsubscribe_to_keyboard_events=lambda device, handle: cleaned.append((device, handle)),
    )
    omni = ModuleType("omni")
    omni.appwindow = ModuleType("omni.appwindow")
    omni.appwindow.get_default_app_window = lambda: SimpleNamespace(get_keyboard=lambda: keyboard)
    for name, module in (("carb", carb), ("carb.input", carb.input), ("omni", omni), ("omni.appwindow", omni.appwindow)):
        monkeypatch.setitem(sys.modules, name, module)
    control = KeyboardVelocity(ranges())
    control.connect()
    callback = events["callback"]
    assert callback(SimpleNamespace(type=1, input="W"))
    assert callback(SimpleNamespace(type=2, input="W"))
    assert not callback(SimpleNamespace(type=3, input="W"))
    assert control.command == (0.2, 0.0, 0.0)
    assert not callback(SimpleNamespace(type=1, input="ESCAPE"))
    control.close()
    control.close()
    assert cleaned == [(keyboard, 0)]
