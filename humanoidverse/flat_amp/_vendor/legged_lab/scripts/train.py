"""Train one of the three flat Pi AMP tasks."""

import argparse
import copy
import os
import traceback
from datetime import datetime
from pathlib import Path

TASKS = ("walk_pi_waq_22", "walk_pi_waq_run_22", "walk_pi_waq_23")


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default=TASKS[0])
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path, help="Full AMP checkpoint to resume")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--logger", choices=("tensorboard", "wandb", "neptune"), default="tensorboard")
    parser.add_argument("--actuator-profile", choices=("ht", "legacy"), default="ht")
    parser.add_argument("--amp-preload", type=int, help="Override expert preload count for smoke tests")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    os.environ["PI_22DOF_ACTUATOR_PROFILE"] = args.actuator_profile
    import torch

    if args.device and str(args.device).startswith("cuda"):
        torch.cuda.set_device(args.device)
    launcher = AppLauncher(args, multi_gpu=False)
    app = launcher.app
    runner = None
    try:
        from isaaclab.utils.io import dump_yaml
        import legged_lab.envs  # noqa: F401
        from legged_lab import LEGGED_LAB_ROOT_DIR
        from legged_lab.utils import task_registry
        from rsl_rl.runners import AmpWaqOnPolicyRunner

        env_cfg, agent_cfg = map(copy.deepcopy, task_registry.get_cfgs(args.task))
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        if args.seed is not None:
            agent_cfg.seed = args.seed
        if args.max_iterations is not None:
            agent_cfg.max_iterations = args.max_iterations
        if args.amp_preload is not None:
            agent_cfg.amp_num_preload_transitions = args.amp_preload
        env_cfg.device = args.device or agent_cfg.device
        agent_cfg.device = env_cfg.device
        agent_cfg.logger = args.logger
        env_cfg.scene.seed = agent_cfg.seed
        torch.manual_seed(agent_cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        if args.run_name:
            stamp += "_" + args.run_name
        log_dir = args.log_dir or Path(LEGGED_LAB_ROOT_DIR) / "logs" / args.task / stamp
        log_dir = log_dir.expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=False)
        env = task_registry.get_task_class(args.task)(env_cfg, args.headless)
        train_cfg = agent_cfg.to_dict()
        # Save pristine configuration before the runner resolves its classes.
        dump_yaml(str(log_dir / "params/env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent_cfg)
        runner = AmpWaqOnPolicyRunner(env, copy.deepcopy(train_cfg), str(log_dir), env_cfg.device)
        if args.resume:
            runner.load(str(args.resume.expanduser().resolve()))
        print(f"[INFO] Flat AMP run: {log_dir}")
        print(f"[INFO] Runtime sources: {Path(__file__).resolve()}")
        runner.learn(agent_cfg.max_iterations, init_at_random_ep_len=True)
        print("[INFO] Training and checkpoint save completed.")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        app.close(skip_cleanup=args.headless)


if __name__ == "__main__":
    main()
