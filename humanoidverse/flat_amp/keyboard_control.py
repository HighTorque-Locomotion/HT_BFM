"""HT_lab-style keys for Pi12 play; simulator APIs are imported only on connect."""

import argparse
import math

AXES = ("lin_vel_x", "lin_vel_y", "ang_vel_z")


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def add_keyboard_arguments(parser):
    parser.add_argument("--keyboard-control", "--keyboard_control", action="store_true", help="Interactive Pi12 playback in the GUI")
    parser.add_argument("--keyboard-linvel-step", "--keyboard_linvel_step", type=positive_float, default=0.1, help="m/s per key event")
    parser.add_argument("--keyboard-angvel-step", "--keyboard_angvel_step", type=positive_float, default=0.1, help="rad/s per key event")


class KeyboardVelocity:
    def __init__(self, ranges, initial=(0.0, 0.0, 0.0), lin_step=0.1, ang_step=0.1):
        self.limits = tuple(tuple(getattr(ranges, axis)) for axis in AXES)
        if any(not all(math.isfinite(v) for v in pair) or not pair[0] <= 0 <= pair[1] for pair in self.limits):
            raise ValueError("Keyboard velocity limits must be finite and contain zero")
        self.lin_step = positive_float(lin_step)
        self.ang_step = positive_float(ang_step)
        if len(initial) != 3 or not all(math.isfinite(v) for v in initial):
            raise ValueError("Initial velocity must contain three finite values")
        self.command = self._clamp(initial)
        self._applied = None
        self._subscription = None

    def _clamp(self, values):
        return tuple(max(low, min(high, round(value, 10))) for value, (low, high) in zip(values, self.limits))

    def handle_key(self, key):
        bindings = {"W": (0, 1), "S": (0, -1), "A": (1, 1), "D": (1, -1), "Q": (2, 1), "E": (2, -1)}
        if key == "X":
            self.command = (0.0, 0.0, 0.0)
        elif key in bindings:
            axis, sign = bindings[key]
            values = list(self.command)
            values[axis] += sign * (self.ang_step if axis == 2 else self.lin_step)
            self.command = self._clamp(values)
        else:
            return False
        self.print_command()
        return True

    def print_command(self):
        vx, vy, yaw = self.command
        print(f"[Keyboard] vx={vx:.2f} m/s, vy={vy:.2f} m/s, yaw={yaw:.2f} rad/s", flush=True)

    def connect(self):
        import carb.input
        import omni.appwindow

        self._window = omni.appwindow.get_default_app_window()
        if self._window is None or self._window.get_keyboard() is None:
            raise RuntimeError("Keyboard control requires an Isaac Sim GUI window; omit --headless")
        self._keyboard = self._window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        key_names = {getattr(carb.input.KeyboardInput, key): key for key in ("W", "S", "A", "D", "Q", "E", "X")}

        def on_event(event, *args):
            if event.type not in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
                return False
            return self.handle_key(key_names.get(event.input))

        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, on_event)
        print("[Keyboard] Focus the Isaac Sim window. W/S: vx, A/D: vy, Q/E: yaw, X: zero command. Release keeps velocity.", flush=True)
        print(f"[Keyboard] Limits (vx, vy, yaw): {self.limits}; all environments share the command.", flush=True)
        self.print_command()

    def close(self):
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None

    def apply(self, env):
        """Commit queued input between simulation steps, including future resets."""
        command = self.command
        if command == self._applied:
            return False
        generator = env.command_generator
        for axis, value in zip(AXES, command):
            # Timer resampling and episode resets must retain the manual command.
            setattr(generator.cfg.ranges, axis, (value, value))
        generator.command[:] = generator.command.new_tensor(command)
        for flag in ("is_standing_env", "is_heading_env", "is_onestep_env", "is_turn_only_env", "is_straight_env"):
            getattr(generator, flag)[:] = False
        self._applied = command
        return True


def refresh_play_observations(env, history, obs):
    """Refresh only the latest frame, without appending an extra history sample.

    Play disables observation noise. Older history retains previous commands.
    """
    env._calculate_gait_para()
    actor, critic = env.compute_current_observations()
    obs["current_actor_obs"] = actor
    obs["current_actor_obs_add_nosie"] = actor.clone()
    obs["current_critic_obs"] = critic
    history[:, -actor.shape[1]:] = actor.clamp(-env.clip_obs, env.clip_obs)
    obs["obs_history"] = history
    obs["critic"][:, -critic.shape[1]:] = critic.clamp(-env.clip_obs, env.clip_obs)
