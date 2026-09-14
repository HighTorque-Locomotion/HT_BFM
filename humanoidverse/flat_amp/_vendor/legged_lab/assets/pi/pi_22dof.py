"""Pi 22-DoF actuator profiles.

The default profile is the HT_lab identified motor parameter set imported from
PiPlus_S_12L8A0G2H0W.py. Set PI_22DOF_ACTUATOR_PROFILE=legacy to reproduce the
previous implicit-PD profile for old checkpoints or A/B comparisons.
"""

import os

_PROFILE = os.environ.get("PI_22DOF_ACTUATOR_PROFILE", "ht").strip().lower()
if _PROFILE in {"ht", "colleague", "ht_lab"}:
    from legged_lab.assets.pi.PiPlus_S_12L8A0G2H0W import (
        PiPlus_S_12L8A0G2H0W_ACTION_SCALE as PI_PLUS_22DOF_ACTION_SCALE,
        PiPlus_S_12L8A0G2H0W_CFG as PI_PLUS_22DOF_CFG,
    )
elif _PROFILE in {"legacy", "old", "implicit"}:
    from legged_lab.assets.pi.pi_22dof_legacy import (  # noqa: F401
        PI_PLUS_22DOF_ACTION_SCALE,
        PI_PLUS_22DOF_CFG,
    )
else:
    raise ValueError("Unsupported PI_22DOF_ACTUATOR_PROFILE={!r}; use ht or legacy".format(_PROFILE))

PI_22DOF_ACTUATOR_PROFILE = _PROFILE
