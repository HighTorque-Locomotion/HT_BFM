"""Export ordinary flat AMP inference as TorchScript and ONNX, without Isaac Sim."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from rsl_rl.modules import ActorCriticWaq


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New export directory")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint.expanduser(), map_location="cpu", weights_only=False)
    if (saved.get("infos") or {}).get("bfm"):
        raise ValueError("BFM command checkpoints require the frozen actor and bridge at inference.")
    if "obs_norm_state_dict" in saved:
        raise ValueError("Expected a flat AMP checkpoint without empirical observation normalization.")
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
    if actions not in (12, 22, 23) or history != 10 * num_obs:
        raise ValueError("Expected a Pi 12/22/23-DoF, ten-frame proprioceptive checkpoint.")
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
                "output": "normalized joint actions; apply task action scale/default pose and head mask in controller",
                "checkpoint": args.checkpoint.name,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[INFO] Exported and checked TorchScript: {output}")


if __name__ == "__main__":
    main()
