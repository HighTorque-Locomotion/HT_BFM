"""Record the tensors actually used by BFM playback, without extra policy inference."""

import json
from pathlib import Path

import numpy as np
import torch


class PolicyTrace:
    def __init__(self, path, env, checkpoint, weights, arguments, chunk_steps=512):
        from legged_lab.utils.bfm_deployment import file_sha256

        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        self.env = env
        self.chunk_steps = chunk_steps
        self.rows = []
        self.pending = None
        self.chunks = 0
        self.steps = 0
        self.metadata = {
            "schema": 1,
            "checkpoint": str(checkpoint.expanduser().resolve()),
            "checkpoint_sha256": file_sha256(checkpoint.expanduser()),
            "bfm_weights": str(weights.expanduser().resolve()),
            "actor_sha256": env.actor.digest,
            "contract": env.bridge.contract.metadata(),
            "target_joint_names": list(env.bridge.target_joint_names),
            "target_default_pos": self.array(env.env.robot.data.default_joint_pos).tolist(),
            "target_action_scale": self.array(torch.as_tensor(env.env.action_scale)).tolist(),
            "clip_actions": float(env.env.clip_actions),
            "step_dt": env.step_dt,
            "num_envs": env.num_envs,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in arguments.items()},
            "joint_target_note": "Target before actuator delay and encoder bias, in target_joint_names order (radians)",
        }
        self._write_metadata()
        self.hook = env.actor.model.register_forward_hook(self._capture_bfm)

    @staticmethod
    def array(tensor):
        return tensor.detach().cpu().numpy().copy()

    def begin(self, obs, raw_z):
        if self.pending is not None:
            raise RuntimeError("Unfinished trace step")
        self.pending = {"encoder_obs": self.array(obs), "raw_z": self.array(raw_z)}

    def _capture_bfm(self, module, inputs, output):
        if self.pending is None:
            return
        observations, z = inputs
        self.pending["bfm_input"] = self.array(torch.cat(
            [observations[key] for key in ("state", "last_action", "history_actor")] + [z], dim=-1
        ))
        self.pending["bfm_action"] = self.array(output)
        physical = self.env.env
        action, _ = self.env.bridge.target_actions(output, physical.robot.data.default_joint_pos, physical.action_scale)
        self.pending["target_action"] = self.array(action)
        self.pending["joint_target"] = self.array(
            action.clamp(-physical.clip_actions, physical.clip_actions) * physical.action_scale + physical.robot.data.default_joint_pos
        )

    def finish(self, done):
        if self.pending is None or "bfm_input" not in self.pending:
            raise RuntimeError("BFM inference was not captured")
        self.pending["done"] = self.array(done)
        self.rows.append(self.pending)
        self.pending = None
        self.steps += 1
        if len(self.rows) >= self.chunk_steps:
            self.flush()

    def _write_metadata(self):
        self.metadata.update(steps=self.steps, chunks=self.chunks)
        temporary = self.path / "metadata.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2) + "\n")
        temporary.replace(self.path / "metadata.json")

    def flush(self):
        if self.rows:
            data = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
            path = self.path / f"chunk_{self.chunks:06d}.npz"
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **data)
            temporary.replace(path)
            self.chunks += 1
            self.rows.clear()
            self._write_metadata()

    def close(self):
        self.hook.remove()
        self.flush()
        print(f"[INFO] Saved {self.steps} policy steps to {self.path}", flush=True)
