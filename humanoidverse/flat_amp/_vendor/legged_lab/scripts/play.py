"""Replay a flat AMP checkpoint with deterministic actions."""

import argparse
import copy
import time
import traceback
from pathlib import Path

from humanoidverse.flat_amp.keyboard_control import KeyboardVelocity, add_keyboard_arguments, refresh_play_observations
from legged_lab.scripts.train import TASKS


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default=TASKS[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--velocity", nargs=3, type=float, metavar=("VX", "VY", "YAW"))
    add_keyboard_arguments(parser)
    parser.add_argument("--actuator-profile", choices=("ht",), default="ht")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.checkpoint.expanduser().is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.keyboard_control and args.headless:
        parser.error("--keyboard-control requires a GUI; omit --headless")
    if args.velocity is None:
        args.velocity = (0.0, 0.0, 0.0) if args.keyboard_control else (0.4, 0.0, 0.0)
    import torch

    if args.device and str(args.device).startswith("cuda"):
        torch.cuda.set_device(args.device)
    launcher = AppLauncher(args, multi_gpu=False)
    app = launcher.app
    keyboard = None
    play_window = None
    env = None
    try:
        from rsl_rl.modules import ActorCriticWaq

        import legged_lab.envs  # noqa: F401
        from legged_lab.utils import task_registry

        env_cfg, agent_cfg = map(copy.deepcopy, task_registry.get_cfgs(args.task))
        env_cfg.device = args.device or agent_cfg.device
        env_cfg.scene.seed = agent_cfg.seed
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.scene.play_ui = not args.headless
        env_cfg.scene.max_episode_length_s = 600.0
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.events.push_robot = None
        env_cfg.commands.heading_command = False
        env_cfg.commands.rel_standing_envs = 0.0
        env_cfg.commands.rel_onestep_envs = 0.0
        env_cfg.commands.rel_turn_only_envs = 0.0
        env_cfg.commands.rel_straight_envs = 0.0
        if args.keyboard_control:
            keyboard = KeyboardVelocity(
                env_cfg.commands.ranges, args.velocity, args.keyboard_linvel_step, args.keyboard_angvel_step
            )
            args.velocity = keyboard.command
        for axis, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), args.velocity):
            setattr(env_cfg.commands.ranges, axis, (value, value))
        env = task_registry.get_task_class(args.task)(env_cfg, args.headless)
        if keyboard is not None:
            keyboard.apply(env)
        history, extras = env.get_observations()
        obs = extras["observations"]
        cfg = agent_cfg.policy.to_dict()
        cfg.pop("class_name")
        policy = ActorCriticWaq(
            obs["current_actor_obs"].shape[1], obs["critic"].shape[1], history.shape[1], env.num_actions, **cfg
        ).to(env.device)
        saved = torch.load(args.checkpoint.expanduser(), map_location=env.device, weights_only=False)
        if (saved.get("infos") or {}).get("bfm"):
            raise ValueError("Expected a walk_pi_waq_12 AMP checkpoint.")
        if "obs_norm_state_dict" in saved:
            raise ValueError("This task uses empirical_normalization=False; checkpoint is incompatible.")
        policy.load_state_dict(saved["model_state_dict"], strict=True)
        policy.eval()
        if not args.headless:
            from humanoidverse.flat_amp.play_ui import FlatAmpPlayWindow

            play_window = FlatAmpPlayWindow(env)
        if keyboard is not None:
            keyboard.connect()
        with torch.inference_mode():
            for _ in range(args.steps):
                start = time.perf_counter()
                if not app.is_running():
                    break
                if keyboard is not None and keyboard.apply(env):
                    refresh_play_observations(env, history, obs)
                actions, _ = policy.act_inference(obs["current_actor_obs_add_nosie"], history)
                history, _, _, extras = env.step(actions)
                obs = extras["observations"]
                if keyboard is not None:
                    # Interactive playback should not run faster than wall time.
                    time.sleep(max(0.0, env.step_dt - (time.perf_counter() - start)))
        print(f"[INFO] Deterministic replay completed: {args.steps} requested steps.", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        try:
            if play_window is not None:
                play_window.close()
            if keyboard is not None:
                keyboard.close()
        finally:
            if env is not None and not args.headless:
                # Match Isaac Lab environment cleanup. Its standalone STOP
                # callback otherwise keeps rendering during app.close().
                env.sim.clear_all_callbacks()
                env.sim.clear_instance()
            app.close(wait_for_replicator=False, skip_cleanup=args.headless)


if __name__ == "__main__":
    main()
