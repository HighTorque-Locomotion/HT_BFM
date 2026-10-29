"""Export AMP or paired BFM+AMP deployment models, without Isaac Sim."""

import argparse
import json
from pathlib import Path

import torch
from rsl_rl.modules import ActorCriticWaq
from torch import nn


class Policy(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.encoder = actor.encoder_module
        self.velocity = actor.cv_vel
        self.latent = actor.cv_mu
        self.actor = actor.actor
        self.num_obs = actor.num_actor_obs

    def forward(self, observation_history):
        h = self.encoder(observation_history)
        current = observation_history[:, -self.num_obs :]
        return self.actor(torch.cat((self.velocity(h), self.latent(h), current), dim=-1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New export directory; required unless --check-only")
    parser.add_argument("--controller", choices=("amp", "bfm"), help="Default: detect from checkpoint metadata")
    parser.add_argument("--bfm-weights", type=Path, help="BFM: original native weights used for training")
    parser.add_argument("--bfm-robot-config", type=Path, help="BFM: override the original config.yaml")
    parser.add_argument("--env-config", type=Path, help="BFM: default <checkpoint parent>/params/env.yaml")
    parser.add_argument("--urdf", type=Path, help="BFM: override the target URDF location saved in env.yaml")
    parser.add_argument("--check-only", action="store_true", help="BFM: validate native models/config without exporting")
    args = parser.parse_args(argv)
    if args.output is None and not args.check_only:
        parser.error("--output is required unless --check-only is selected")
    if not args.check_only and args.output.expanduser().exists():
        raise FileExistsError(f"Output must be new: {args.output}")
    # Explicit local checkpoints include RSL-RL normalizer objects.
    saved = torch.load(args.checkpoint.expanduser(), map_location="cpu", weights_only=False)
    controller = "bfm" if (saved.get("infos") or {}).get("bfm") is not None else "amp"
    if args.controller is not None and args.controller != controller:
        raise ValueError(f"--controller {args.controller} does not match the {controller} checkpoint")
    if "obs_norm_state_dict" in saved:
        raise ValueError("Expected a flat_amp checkpoint without empirical observation normalization.")
    if controller == "bfm":
        if args.bfm_weights is None:
            parser.error("BFM+AMP export requires --bfm-weights with the original native BFM")
        from legged_lab.scripts.export_bfm_policy import export_checkpoint

        return export_checkpoint(args, saved)
    if args.bfm_weights or args.bfm_robot_config or args.env_config or args.urdf or args.check_only:
        parser.error("BFM export options require a BFM+AMP checkpoint")
    export_amp_checkpoint(args, saved)


def export_amp_checkpoint(args, saved):
    """Preserve the ordinary AMP TorchScript/ONNX interface."""
    state = saved["model_state_dict"]

    def sizes(prefix):
        return [
            v.shape[0]
            for k, v in sorted(
                state.items(), key=lambda kv: int(kv[0].split(".")[1]) if kv[0].startswith(prefix + ".") else -1
            )
            if k.startswith(prefix + ".") and k.endswith(".weight")
        ]

    actions = sizes("actor")[-1]
    num_obs = 9 + 3 * actions + 2
    history = state["encoder_module.0.weight"].shape[1]
    if actions != 12 or history != 10 * num_obs:
        raise ValueError("Expected a Pi12, ten-frame proprioceptive checkpoint.")
    actor = ActorCriticWaq(
        num_obs,
        state["critic.0.weight"].shape[1],
        history,
        actions,
        actor_hidden_dims=sizes("actor")[:-1],
        critic_hidden_dims=sizes("critic")[:-1],
        encoder_hidden_dims=sizes("encoder_module"),
        decoder_hidden_dims=sizes("decoder_module")[:-1],
        latent_dim=state["cv_mu.weight"].shape[0] + 3,
    )
    actor.load_state_dict(state, strict=True)
    policy = Policy(actor).eval()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    example = torch.zeros(1, history)
    scripted = torch.jit.script(policy)
    scripted.save(str(output / "policy.pt"))
    torch.onnx.export(
        policy,
        example,
        str(output / "policy.onnx"),
        input_names=["observation_history"],
        output_names=["actions"],
        dynamic_axes={"observation_history": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    torch.manual_seed(42)
    samples = torch.randn(32, history)
    with torch.no_grad():
        reference = actor.act_inference(samples[:, -num_obs:], samples)[0]
        torch.testing.assert_close(scripted(samples), reference, atol=1e-6, rtol=1e-5)
    (output / "interface.json").write_text(
        json.dumps(
            {
                "dof": actions,
                "frame_dim": num_obs,
                "history_dim": history,
                "history_order": "oldest_to_newest",
                "input": "scaled proprioceptive observation history",
                "output": "normalized joint actions; apply task action scale/default pose in controller",
                "checkpoint": args.checkpoint.name,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[INFO] Exported and checked TorchScript: {output}")


if __name__ == "__main__":
    main()
