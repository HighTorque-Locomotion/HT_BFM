"""Train Pi12 command-to-z with frozen BFM + online AMP/PPO, from origin/bmf-amp-lyp."""

from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from humanoidverse.flat_amp import PROJECT_ROOT, TASKS
from humanoidverse.flat_amp.__main__ import positive_int
from humanoidverse.flat_amp.keyboard_control import KeyboardVelocity, add_keyboard_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default=TASKS[0])
    parser.add_argument("--bfm-weights", type=Path, required=True)
    parser.add_argument("--bfm-robot-config", type=Path)
    parser.add_argument("--num_envs", type=positive_int)
    parser.add_argument("--max_iterations", type=positive_int)
    parser.add_argument("--amp-preload", type=positive_int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-phase", action="store_true")
    parser.add_argument("--resume", type=Path, help="Pi12 command-to-z checkpoint")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--play-steps", type=positive_int, default=1000)
    parser.add_argument("--record-policy", type=Path)
    parser.add_argument("--velocity", nargs=3, type=float, help="Fixed velocity, or initial keyboard command")
    add_keyboard_arguments(parser)
    preview, _ = parser.parse_known_args()
    if preview.dry_run:
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--headless", action="store_true")
    else:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.record_policy is not None and not args.play:
        parser.error("--record-policy requires --play")
    if args.record_policy is not None and args.record_policy.expanduser().exists():
        parser.error("--record-policy directory must not already exist")
    if args.play and (args.resume is None or args.dry_run):
        parser.error("--play requires --resume and cannot be combined with --dry-run")
    if args.keyboard_control and not args.play:
        parser.error("--keyboard-control requires --play")
    if args.keyboard_control and args.headless:
        parser.error("--keyboard-control requires a GUI; omit --headless")
    if args.velocity is None:
        args.velocity = (0.0, 0.0, 0.0) if args.keyboard_control else (0.4, 0.0, 0.0)

    import torch

    from legged_lab.utils.bfm_actor import FrozenBFM, resolve_robot_config
    from legged_lab.utils.bfm_bridge import BFMBridge, BFMContract, BFMLatentEnv

    device = args.device or "cuda:0"
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
    contract = BFMContract.from_robot_yaml(resolve_robot_config(args.bfm_weights, args.bfm_robot_config))
    actor = FrozenBFM(args.bfm_weights, contract, device)
    if args.dry_run:
        batch = args.num_envs or 2
        bridge = BFMBridge(contract, list(contract.joint_names), batch, device)
        q = bridge.default_pos.expand(batch, -1)
        obs = bridge.observe(q, torch.zeros_like(q), q.new_tensor([0, 0, -1]).expand(batch, -1), q.new_zeros(batch, 3))
        phase = q.new_tensor([0, 1]).expand(batch, -1) if not args.no_phase else None
        encoder_obs = bridge.encoder_observation(obs, q.new_zeros(batch, 3), phase)
        z = bridge.project_z(torch.randn(batch, contract.z_dim, device=device))
        result = actor(obs, z)
        if result.shape != (batch, 12) or not torch.isfinite(result).all():
            raise RuntimeError("BFM validation produced invalid actions")
        print(json.dumps({
            "encoder_observation_shape": list(encoder_obs.shape), "action_shape": list(result.shape),
            "z_norm": z.norm(dim=-1).mean().item(), "trainable_bfm_parameters": 0,
            "actor_sha256": actor.digest, "contract": contract.metadata(),
        }, indent=2))
        return

    # BFM weights and dimensions are validated before creating the physics stage.
    launcher = AppLauncher(args, multi_gpu=False)
    app = launcher.app
    runner = None
    physical_env = None
    play_window = None
    keyboard = None
    recorder = None
    try:
        from isaaclab.utils.io import dump_yaml

        import legged_lab.envs  # noqa: F401
        from legged_lab.utils import task_registry
        from legged_lab.utils.bfm_training import BFMOnPolicyRunner, make_bfm_train_cfg

        env_cfg, agent_cfg = map(copy.deepcopy, task_registry.get_cfgs(args.task))
        env_cfg.device = device
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        elif args.play:
            env_cfg.scene.num_envs = 1
        if args.seed is not None:
            agent_cfg.seed = args.seed
        env_cfg.scene.seed = agent_cfg.seed
        torch.manual_seed(agent_cfg.seed)
        cfg = make_bfm_train_cfg(agent_cfg.to_dict(), contract.z_dim, use_phase=not args.no_phase)
        cfg["device"] = device
        cfg["bfm_actuator_profile"] = "ht"
        cfg["policy"]["hidden_dim"] = actor.encoder_hidden_dim
        cfg["policy"]["hidden_layers"] = actor.encoder_hidden_layers
        if args.max_iterations is not None:
            cfg["max_iterations"] = args.max_iterations
        if args.amp_preload is not None:
            cfg["amp_num_preload_transitions"] = args.amp_preload
        if args.play:
            env_cfg.scene.play_ui = not args.headless
            env_cfg.scene.max_episode_length_s = 600.0
            env_cfg.foot_contact_randomization.enabled = False
            env_cfg.foot_contact_stiffness = 0.0
            env_cfg.foot_contact_damping = 0.0
            env_cfg.velocity_curriculum.enabled = False
            env_cfg.noise.add_noise = False
            env_cfg.domain_rand.events.push_robot = None
            env_cfg.commands.heading_command = False
            for name in ("rel_standing_envs", "rel_onestep_envs", "rel_turn_only_envs", "rel_straight_envs"):
                setattr(env_cfg.commands, name, 0.0)
            if args.keyboard_control:
                keyboard = KeyboardVelocity(
                    env_cfg.commands.ranges, args.velocity, args.keyboard_linvel_step, args.keyboard_angvel_step
                )
                args.velocity = keyboard.command
            for name, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), args.velocity):
                setattr(env_cfg.commands.ranges, name, (value, value))
            cfg["amp_num_preload_transitions"] = 1
        physical_env = task_registry.get_task_class(args.task)(env_cfg, args.headless)
        if keyboard is not None:
            keyboard.apply(physical_env)
        env = BFMLatentEnv(physical_env, actor, contract, use_phase=not args.no_phase)
        run_dir = args.log_dir or PROJECT_ROOT / "logs/flat_amp" / args.task / (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_bfm"
        )
        if not args.play:
            run_dir = run_dir.expanduser().resolve()
            run_dir.mkdir(parents=True, exist_ok=False)
            dump_yaml(str(run_dir / "params/env.yaml"), env_cfg)
            dump_yaml(str(run_dir / "params/agent.yaml"), cfg)
            (run_dir / "bfm_contract.json").write_text(json.dumps(contract.metadata(), indent=2) + "\n")
        runner = BFMOnPolicyRunner(env, copy.deepcopy(cfg), log_dir=None if args.play else str(run_dir), device=device)
        if args.resume:
            runner.load(str(args.resume.expanduser().resolve()), load_optimizer=not args.play)
        if args.play:
            policy = runner.get_inference_policy()
            if args.record_policy is not None:
                from humanoidverse.flat_amp.bfm_policy_trace import PolicyTrace

                recorder = PolicyTrace(args.record_policy, env, args.resume, args.bfm_weights, vars(args))
            obs, _ = env.get_observations()
            if not args.headless:
                from humanoidverse.flat_amp.play_ui import FlatAmpPlayWindow

                play_window = FlatAmpPlayWindow(physical_env)
            if keyboard is not None:
                keyboard.connect()
            with torch.inference_mode():
                for _ in range(args.play_steps):
                    start = time.perf_counter()
                    if not app.is_running():
                        break
                    if keyboard is not None and keyboard.apply(physical_env):
                        obs, _ = env.refresh_play_observations()
                    raw_z = policy(obs)
                    if recorder is not None:
                        recorder.begin(obs, raw_z)
                    obs, _, done, _ = env.step(raw_z)
                    if recorder is not None:
                        recorder.finish(done)
                    if keyboard is not None:
                        time.sleep(max(0.0, env.step_dt - (time.perf_counter() - start)))
            print("[INFO] Pi12 BFM playback completed.", flush=True)
        else:
            print(f"[INFO] Pi12 frozen BFM + command-to-z AMP/PPO run: {run_dir}", flush=True)
            runner.learn(cfg["max_iterations"], init_at_random_ep_len=True)
            print("[INFO] BFM training and checkpoint save completed.", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        try:
            if recorder is not None:
                recorder.close()
            if play_window is not None:
                play_window.close()
            if keyboard is not None:
                keyboard.close()
            if runner is not None and runner.writer is not None:
                runner.writer.flush()
                runner.writer.close()
        finally:
            if args.play and not args.headless and physical_env is not None:
                physical_env.sim.clear_all_callbacks()
                physical_env.sim.clear_instance()
            app.close(wait_for_replicator=False, skip_cleanup=args.headless)


if __name__ == "__main__":
    main()
