"""Foot collision clearance and once-per-swing height shortfall events.

Adapted from HT_lab/tasks/pi_soccer_headfree/mdp/foot_height.py:
SoleClearance and FootHeightShortfallEvents. No runtime dependency on HT_lab.
"""

import math

import torch


class SoleClearance:
    """Exact minimum of the Pi foot's loaded Cube/Capsule collision surfaces."""

    def __init__(self, shapes, device):
        if len(shapes) != 2 or len(shapes[0]) != len(shapes[1]) or not shapes[0]:
            raise ValueError("Expected two feet with matching nonempty collision shape counts")
        if any(s["type"] not in ("Cube", "Capsule") for foot in shapes for s in foot):
            raise ValueError("Pi sole clearance supports Cube and Capsule collisions only")

        def tensor(values):
            return torch.tensor(values, device=device, dtype=torch.float32)

        transform = tensor([[s["transform"] for s in foot] for foot in shapes])
        self.linear, self.center = transform[..., :3, :3], transform[..., 3, :3]
        self.cube = tensor([[s["type"] == "Cube" for s in foot] for foot in shapes]).bool()
        self.half_size = tensor([[s.get("size", 0) * 0.5 for s in foot] for foot in shapes])
        self.half_length = tensor([[s.get("height", 0) * 0.5 for s in foot] for foot in shapes])
        self.radius = tensor([[s.get("radius", 0) for s in foot] for foot in shapes])
        self.axis = torch.tensor([["XYZ".index(s.get("axis", "Z").upper()) for s in foot] for foot in shapes], device=device)

    @classmethod
    def from_stage(cls, stage, device):
        from pxr import Usd, UsdGeom, UsdPhysics

        shapes = []
        for side in ("l", "r"):
            link = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{side}_ankle_roll_link")
            if not link.IsValid():
                raise ValueError(f"Missing {side} foot prim")
            inverse = UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
            foot = []
            for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                    continue
                transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * inverse
                kind = prim.GetTypeName()
                item = dict(type=kind, transform=[list(row) for row in transform])
                if kind == "Cube":
                    item["size"] = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
                elif kind == "Capsule":
                    geom = UsdGeom.Capsule(prim)
                    item.update(
                        height=float(geom.GetHeightAttr().Get()),
                        radius=float(geom.GetRadiusAttr().Get()),
                        axis=str(geom.GetAxisAttr().Get()),
                    )
                foot.append(item)
            shapes.append(foot)
        return cls(shapes, device)

    def __call__(self, positions, quaternions, ground_z):
        q = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        w, x, y, z = q.unbind(-1)
        up = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)
        normal = torch.einsum("nfi,fsji->nfsj", up, self.linear)
        center_z = torch.einsum("nfi,fsi->nfs", up, self.center)
        along = normal.gather(-1, self.axis[None, ..., None].expand(len(up), -1, -1, -1)).squeeze(-1)
        extent = torch.where(
            self.cube, self.half_size * normal.abs().sum(-1), self.half_length * along.abs() + self.radius * normal.norm(dim=-1)
        )
        return positions[..., 2] - ground_z[:, None] + (center_z - extent).amin(-1)


class FootHeightShortfallEvents:
    """Settle a missed clearance target once, after each foot's window closes.

    Heights outside the window cannot qualify. A foot must have been observed
    inside an eligible window; spawning in stance cannot cause an immediate
    penalty. Mode changes/reset discard unfinished windows in the adapter.
    """

    def __init__(self, num_envs, device, phase_window=(0.10, 0.45), minimum_height=0.017):
        if (
            not all(math.isfinite(v) for v in (*phase_window, minimum_height))
            or not 0 <= phase_window[0] < phase_window[1] < 0.5
            or minimum_height <= 0
        ):
            raise ValueError("Expected a swing window within [0, .5) and positive finite minimum height")
        self.phase_window = phase_window
        self.minimum_height = minimum_height
        self.seen = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.reached = torch.zeros_like(self.seen)
        self.settled = torch.zeros_like(self.seen)
        self.previous_phase = torch.full((num_envs, 2), -1.0, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.seen[ids] = False
        self.reached[ids] = False
        self.settled[ids] = False
        self.previous_phase[ids] = -1

    def update(self, clearance, phase, speed, eligible):
        # Same update interface as positive height events; this target is fixed.
        wrapped = phase < self.previous_phase - 0.5
        self.previous_phase.copy_(phase)
        self.settled &= ~wrapped
        retain = ~wrapped & eligible[:, None]
        self.seen &= retain
        self.reached &= retain
        a, b = self.phase_window
        inside = eligible[:, None] & (phase >= a - 1e-6) & (phase <= b + 1e-6)
        self.seen |= inside
        self.reached |= inside & (clearance >= self.minimum_height)
        closed = eligible[:, None] & self.seen & (phase > b + 1e-6) & ~self.settled
        missed = closed & ~self.reached
        self.settled |= closed
        return missed
