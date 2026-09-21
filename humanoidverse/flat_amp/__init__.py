"""Pi flat-ground AMP with an optional frozen BFM controller."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = Path(__file__).resolve().parent / "_vendor"
TASKS = ("walk_pi_waq_22", "walk_pi_waq_run_22", "walk_pi_waq_23", "walk_pi_waq_12")
CONTROLLERS = ("amp", "bfm", "bfm-phase")


def activate():
    """Use the bundled training core while preserving historical checkpoint module names.

    Activation is explicit so importing HumanoidVerse or its existing Stage2 does
    not change Python's package search path. A process cannot mix training cores.
    """
    for package in ("legged_lab", "rsl_rl"):
        module = sys.modules.get(package)
        if module is not None:
            location = getattr(module, "__file__", None)
            expected = VENDOR_ROOT / package
            if location is None or not Path(location).resolve().is_relative_to(expected):
                raise RuntimeError(f"{package} is already imported from {location}; start flat AMP in a fresh process.")
    for path in (PROJECT_ROOT, VENDOR_ROOT):
        entry = str(path)
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
