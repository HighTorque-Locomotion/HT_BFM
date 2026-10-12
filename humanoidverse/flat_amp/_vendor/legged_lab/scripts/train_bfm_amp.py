"""Train a command encoder over frozen Pi BFM using walk_pi_waq_22's rewards and AMP data."""

from __future__ import annotations

import argparse
import copy
import json
import os
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def _dry_run_distillation(actor, contract, settings, profile, device):
    """Validate actual teachers and warm start without Isaac or expert preloading."""
    from types import SimpleNamespace

    import torch
    from rsl_rl.modules import Discriminator
    from rsl_rl.modules.bfm_command_actor_critic import BFMCommandActorCritic
    from rsl_rl.utils import Normalizer

    from legged_lab.utils.bfm_bridge import BFMBridge
    from legged_lab.utils.bfm_distillation import DualBFMTeachers

    checkpoint = torch.load(settings["no_phase_teacher"], map_location="cpu", weights_only=False)
    metadata = copy.deepcopy((checkpoint.get("infos") or {}).get("bfm") or {})
    for key, value in {
        "contract": contract.metadata(),
        "actor_sha256": actor.digest,
        "actuator_profile": "ht",
        "latent_projection": "sqrt(z_dim)*normalize(raw_z)",
    }.items():
        if metadata.get(key) != value:
            raise ValueError(f"Teacher {key} differs from the requested frozen BFM")
    metadata.update(
        schema=4,
        latent_std_mode="trainable_projected_log_std_-5_1",
        actor_observation="command,state,last_action,history_actor,phase_sin_cos",
        gait_profile=profile.metadata(),
        phase_observation={
            "encoding": "sin(2*pi*phase),cos(2*pi*phase)",
            "source": "current_actor_obs_add_nosie[:,75:77]",
            "encoder_indices": [363, 364],
            "gait_cycle": 0.8,
            "lin_command_threshold": 0.1,
            "yaw_command_threshold": 0.1,
            "standing_phase": 0.0,
            "reset_phase_offsets": [0.0, 0.5],
            "right_leg_phase_offset": 0.5,
            "scheduler": profile.metadata(),
        },
    )
    state = checkpoint["model_state_dict"]
    critic_weights = [v for k, v in state.items() if k.startswith("critic.") and k.endswith("weight") and v.ndim == 2]
    actor_weights = [v for k, v in state.items() if k.startswith("actor.") and k.endswith("weight") and v.ndim == 2]
    hidden_dim = actor_weights[0].shape[0]
    hidden_layers = len(actor_weights) - 1
    if actor.model is not None and (
        hidden_dim != int(actor.model.cfg.archi.b.hidden_dim) or hidden_layers != int(actor.model.cfg.archi.b.hidden_layers)
    ):
        raise ValueError("Teacher architecture differs from the frozen BFM command encoder")
    student = BFMCommandActorCritic(
        365,
        critic_weights[0].shape[1] + 2,
        contract.z_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        critic_hidden_dims=[w.shape[0] for w in critic_weights[:-1]],
        activation="elu",
        project_log_std=True,
    ).to(device)
    teachers = DualBFMTeachers(student, metadata, settings)
    discriminator_state = checkpoint["discriminator_state_dict"]
    trunk_weights = [v for k, v in discriminator_state.items() if k.startswith("trunk.") and k.endswith("weight")]
    discriminator = Discriminator(trunk_weights[0].shape[1], 0.3, [v.shape[0] for v in trunk_weights], device, 0.65)
    optimizer = torch.optim.Adam(student.parameters())
    teachers.assert_frozen(optimizer)
    bridge = BFMBridge(contract, metadata["target_joint_names"], 6, device)
    source_q = bridge.default_pos.expand(6, -1)
    observations = bridge.observe(
        source_q[:, bridge.to_target], torch.zeros_like(source_q), source_q.new_tensor([0, 0, -1]).expand(6, -1), source_q.new_zeros(6, 3)
    )
    commands = source_q.new_tensor([[-0.2, 0, 0], [0, 0, 0], [0.3, 0, 0], [-0.2, 0.1, 0.2], [0, 0.3, 0], [0, 0, 0.3]])
    obs = bridge.encoder_observation(observations, commands, source_q.new_tensor([0, 1]).expand(6, -1))
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            policy=student, optimizer=optimizer, discriminator=discriminator, amp_normalizer=Normalizer(trunk_weights[0].shape[1] // 2)
        ),
        env=SimpleNamespace(get_observations=lambda: (obs, {})),
    )
    initialization = teachers.initialize_student(runner)
    loss, stats = teachers(obs, student.act_inference(obs), 0)
    loss.backward()
    if not torch.isfinite(loss) or any(
        p.grad is not None for teacher in (teachers.no_phase_actor, teachers.phase_actor) for p in teacher.parameters()
    ):
        raise RuntimeError("Distillation CPU validation produced invalid loss or teacher gradients")
    return {
        "teachers": teachers.metadata,
        "initialization": initialization,
        "encoder_input": 365,
        "trainable_teacher_parameters": 0,
        "loss": float(loss.detach()),
        "groups": {k: int(v.sum()) for k, v in teachers.groups(obs).items()},
        "coefficient": float(stats["coefficient"]),
    }


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
    parser.add_argument(
        "--gait-profile",
        choices=["fixed", "backward_fast"],
        default="fixed",
        help="Optional historical clock: backward 0.5s, other moving commands 0.8s",
    )
    parser.add_argument("--backward-period", type=float, default=0.5)
    parser.add_argument("--distill-no-phase-teacher", help="Schema-1 teacher for backward/standing and student initialization")
    parser.add_argument("--distill-phase-teacher", help="Original 0.8s schema-2 teacher for other moving commands")
    parser.add_argument("--distill-coef", type=float, default=5.0)
    parser.add_argument("--distill-final-coef", type=float, default=0.5)
    parser.add_argument("--distill-hold", type=int, default=1000)
    parser.add_argument("--distill-decay", type=int, default=10000)
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

    variable_gait = args.gait_profile == "backward_fast"
    distillation = bool(args.distill_no_phase_teacher or args.distill_phase_teacher)
    if variable_gait and (args.no_phase or args.actuator_profile != "ht"):
        parser.error("Backward-fast gait requires explicit phase observations and HTMotor")
    if distillation and (not args.distill_no_phase_teacher or not args.distill_phase_teacher or not variable_gait):
        parser.error("Distillation requires both teachers and --gait-profile backward_fast")
    if distillation and args.play:
        parser.error("Play the trained student without teacher options")
    profile = None
    settings = None
    if variable_gait:
        from legged_lab.utils.bfm_gait import BackwardFastGait

        profile = BackwardFastGait(backward_period=args.backward_period)
    if distillation:
        from legged_lab.utils.bfm_distillation import DistillationSchedule

        settings = {
            "no_phase_teacher": str(Path(args.distill_no_phase_teacher).expanduser().resolve()),
            "phase_teacher": str(Path(args.distill_phase_teacher).expanduser().resolve()),
            "schedule": asdict(
                DistillationSchedule(
                    coefficient=args.distill_coef,
                    final_coefficient=args.distill_final_coef,
                    hold_iterations=args.distill_hold,
                    decay_iterations=args.distill_decay,
                )
            ),
        }
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        metadata = (saved.get("infos") or {}).get("bfm") or {}
        if metadata.get("schema") not in (1, 2, 4):
            parser.error("Expected the original frozen-BFM schemas 1/2 or dual-teacher variable-gait schema 4")
        if variable_gait != (metadata.get("schema") == 4):
            parser.error("Schema-4 students require --gait-profile backward_fast; original baselines use fixed gait")
        if not args.play and bool((saved.get("infos") or {}).get("distillation")) != distillation:
            parser.error("Distillation resume requires both original teachers and the original schedule")
        has_phase = metadata.get("schema") in (2, 4) and "phase_observation" in metadata
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
        validation = {
            "action_shape": list(result.shape),
            "z_norm": z.norm(dim=-1).mean().item(),
            "trainable_bfm_parameters": 0,
            "actor_sha256": actor.digest,
            "contract": contract.metadata(),
        }
        if settings is not None:
            validation["distillation"] = _dry_run_distillation(actor, contract, settings, profile, device)
        print(json.dumps(validation, indent=2))
        return

    # This entry point is single-GPU, including rendering, on shared training hosts.
    launcher = AppLauncher(args, multi_gpu=False)
    app = launcher.app
    runner = None
    try:
        from isaaclab.utils.io import dump_yaml

        import legged_lab.envs  # noqa: F401
        from legged_lab.utils import task_registry
        from legged_lab.utils.bfm_training import BFMOnPolicyRunner, configure_backward_fast_gait, make_bfm_train_cfg, snapshot_bfm_run

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
        if profile is not None:
            from legged_lab.utils.bfm_gait import with_gait_environment

            configure_backward_fast_gait(env_cfg, cfg, profile)
            env_class = with_gait_environment(env_class, profile)
            env_cfg.scene.robot.spawn.force_usd_conversion = False
        if settings is not None:
            cfg["bfm_distillation"] = settings
            cfg["run_name"] = "dual_teacher_distillation_ppo_warm_start"
        physical_env = env_class(env_cfg, args.headless)
        env = BFMLatentEnv(physical_env, actor, contract, use_phase=not args.no_phase)
        run_dir = (
            Path(args.log_dir) if args.log_dir else Path("logs") / cfg["experiment_name"] / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        if not args.play:
            run_dir.mkdir(parents=True, exist_ok=False)
            dump_yaml(str(run_dir / "params/env.yaml"), env_cfg)
            dump_yaml(str(run_dir / "params/agent.yaml"), cfg)
            (run_dir / "bfm_contract.json").write_text(json.dumps(contract.metadata(), indent=2) + "\n")
            if profile is not None:
                snapshot_bfm_run(run_dir, cfg)
        runner = BFMOnPolicyRunner(env, cfg, log_dir=None if args.play else str(run_dir), device=device)
        if args.resume:
            runner.load(args.resume, load_optimizer=not args.play)
        elif settings is not None:
            initialization = runner.initialize_from_no_phase_teacher()
            with (run_dir / "launch/distillation_initialization.json").open("x") as output:
                json.dump({"teachers": runner.distiller.metadata, "initialization": initialization}, output, indent=2)
            print(f"[BFM distillation] teacher warm start, not resume: {initialization}", flush=True)
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
