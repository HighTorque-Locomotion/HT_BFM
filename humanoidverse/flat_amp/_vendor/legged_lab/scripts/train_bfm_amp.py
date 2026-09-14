"""Train a command encoder over frozen Pi BFM using walk_pi_waq_22's rewards and AMP data."""

from __future__ import annotations

import argparse
import copy
import json
import os
import traceback
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="walk_pi_waq_22", choices=["walk_pi_waq_22"])
    parser.add_argument(
        "--bfm-weights",
        required=True,
        help="Native run/checkpoint/model directory, or frozen actor ONNX",
    )
    parser.add_argument("--bfm-source", required=True, help="Path to HT_BFM source checkout")
    parser.add_argument(
        "--bfm-robot-config",
        help="Robot YAML used by the BFM checkpoint; defaults to source H0W config",
    )
    parser.add_argument("--actuator-profile", choices=["ht", "legacy"], default="ht")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--no-phase",
        action="store_true",
        help="Legacy 363-D encoder, for replay/resume of old no-phase checkpoints; default is explicit sin/cos (365-D)",
    )
    parser.add_argument(
        "--resume",
        help="Checkpoint from this BFM migration, not an old locomotion model",
    )
    parser.add_argument("--log-dir", help="New run directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and check actual BFM weights without starting Isaac",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Run --resume using deterministic latent means",
    )
    parser.add_argument("--play-steps", type=int, default=1000)
    # Parse --help and CPU validation without importing Isaac Sim.
    preview, _ = parser.parse_known_args()
    if preview.dry_run:
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--headless", action="store_true")
        args = parser.parse_args()
        app = None
    else:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        app = None

    os.environ["PI_22DOF_ACTUATOR_PROFILE"] = args.actuator_profile
    import torch

    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        metadata = (saved.get("infos") or {}).get("bfm") or {}
        has_phase = metadata.get("schema") == 2 and "phase_observation" in metadata
        if has_phase == args.no_phase:
            raise ValueError(
                "Checkpoint/CLI phase mismatch: use --no-phase for legacy 363-D checkpoints; "
                "365-D phase training requires a phase checkpoint or a fresh run (omit --resume)."
            )
        del saved

    from legged_lab.utils.bfm_bridge import (
        BFMBridge,
        BFMContract,
        BFMLatentEnv,
        FrozenBFM,
        with_terminal_amp_capture,
    )

    source = Path(args.bfm_source).expanduser().resolve()
    robot_config = args.bfm_robot_config or source / "humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H0W.yaml"
    contract = BFMContract.from_robot_yaml(robot_config)
    device = getattr(args, "device", None) or "cuda:0"
    if torch.device(device).type == "cuda":
        # The source loader accepts only "cuda", so select the requested card before
        # it allocates weights; otherwise cuda:4 briefly loads the full BFM on GPU 0.
        torch.cuda.set_device(device)
    # Check weight availability before spending time creating the simulation.
    actor = FrozenBFM(args.bfm_weights, contract, device, source)
    if args.dry_run:
        bridge = BFMBridge(contract, list(contract.joint_names), args.num_envs, device)
        q = bridge.default_pos.expand(args.num_envs, -1)
        obs = bridge.observe(
            q,
            torch.zeros_like(q),
            q.new_tensor([0, 0, -1]).expand(args.num_envs, -1),
            q.new_zeros(args.num_envs, 3),
        )
        z = bridge.project_z(torch.randn(args.num_envs, contract.z_dim, device=device))
        result = actor(obs, z)
        if result.shape != (args.num_envs, 22) or not torch.isfinite(result).all():
            raise RuntimeError("BFM validation produced invalid actions")
        print(
            json.dumps(
                {
                    "action_shape": list(result.shape),
                    "z_norm": z.norm(dim=-1).mean().item(),
                    "trainable_bfm_parameters": 0,
                    "actor_sha256": actor.digest,
                    "contract": contract.metadata(),
                },
                indent=2,
            )
        )
        return

    # This entry point is single-GPU, including rendering, on shared training hosts.
    launcher = AppLauncher(args, multi_gpu=False)
    app = launcher.app
    runner = None
    try:
        from isaaclab.utils.io import dump_yaml

        import legged_lab.envs  # noqa: F401
        from legged_lab.utils import task_registry
        from legged_lab.utils.bfm_training import BFMOnPolicyRunner, make_bfm_train_cfg

        env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
        env_cfg, agent_cfg = copy.deepcopy(env_cfg), copy.deepcopy(agent_cfg)
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.device = device
        env_cfg.sim.device = device
        if args.seed is not None:
            agent_cfg.seed = args.seed
        env_cfg.scene.seed = agent_cfg.seed
        torch.manual_seed(agent_cfg.seed)
        cfg = make_bfm_train_cfg(agent_cfg.to_dict(), contract.z_dim, use_phase=not args.no_phase)
        cfg["device"] = device
        cfg["bfm_actuator_profile"] = args.actuator_profile
        if actor.model is not None:
            cfg["policy"]["hidden_dim"] = int(actor.model.cfg.archi.b.hidden_dim)
            cfg["policy"]["hidden_layers"] = int(actor.model.cfg.archi.b.hidden_layers)
        if args.max_iterations is not None:
            cfg["max_iterations"] = args.max_iterations
        env_class = with_terminal_amp_capture(task_registry.get_task_class(args.task))
        physical_env = env_class(env_cfg, args.headless)
        env = BFMLatentEnv(physical_env, actor, contract, use_phase=not args.no_phase)
        run_dir = (
            Path(args.log_dir)
            if args.log_dir
            else Path("logs") / cfg["experiment_name"] / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        if not args.play:
            run_dir.mkdir(parents=True, exist_ok=False)
            dump_yaml(str(run_dir / "params/env.yaml"), env_cfg)
            dump_yaml(str(run_dir / "params/agent.yaml"), cfg)
            (run_dir / "bfm_contract.json").write_text(json.dumps(contract.metadata(), indent=2) + "\n")
        runner = BFMOnPolicyRunner(env, cfg, log_dir=None if args.play else str(run_dir), device=device)
        if args.resume:
            runner.load(args.resume, load_optimizer=not args.play)
        if args.play:
            if not args.resume:
                raise ValueError("--play requires a trained command encoder checkpoint via --resume")
            policy = runner.get_inference_policy()
            obs, _ = env.get_observations()
            with torch.inference_mode():
                for _ in range(args.play_steps):
                    if not app.is_running():
                        break
                    obs, _, _, _ = env.step(policy(obs))
            print("BFM playback completed", flush=True)
        else:
            print("Frozen BFM + target AMP/PPO; reward weights/data unchanged; joint-action mirror loss disabled.")
            runner.learn(cfg["max_iterations"], init_at_random_ep_len=True)
            print(f"BFM AMP training completed: {run_dir}", flush=True)
    except BaseException:
        # Isaac shutdown can delay exception propagation; expose the original error first.
        traceback.print_exc()
        raise
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        # bxkj/Isaac 5.1 can hang closing a headless physics stage. Checkpoints and
        # logging are finished above; use SimulationApp's supported immediate shutdown.
        app.close(wait_for_replicator=False, skip_cleanup=args.headless)


if __name__ == "__main__":
    main()
