from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any


PACKAGE_SCHEME = "package://"


def _as_str(path: Any) -> str:
    return "" if path is None else str(path)


def _package_path(path: str) -> Path | None:
    if not path.startswith(PACKAGE_SCHEME):
        return None

    package_and_rel = path[len(PACKAGE_SCHEME) :]
    package_name, _, rel_path = package_and_rel.partition("/")
    if not package_name:
        raise ValueError(f"Invalid package asset path: {path!r}")

    module = importlib.import_module(package_name)
    if module.__file__ is None:
        raise ValueError(f"Package {package_name!r} does not expose a filesystem path.")

    package_root = Path(module.__file__).resolve().parent
    return package_root / rel_path if rel_path else package_root


def resolve_asset_path(asset_root: Any, asset_file: Any = "") -> Path:
    """Resolve local or package-scoped robot asset paths.

    Package paths use the form ``package://ht_urdf/path/in/package`` and are
    resolved from ``os.path.dirname(ht_urdf.__file__)``.
    """
    root = _as_str(asset_root)
    file = _as_str(asset_file)

    file_package_path = _package_path(file)
    if file_package_path is not None:
        return file_package_path

    if file and os.path.isabs(file):
        return Path(file)

    root_package_path = _package_path(root)
    if root_package_path is not None:
        return root_package_path / file

    return Path(os.path.join(root, file))
