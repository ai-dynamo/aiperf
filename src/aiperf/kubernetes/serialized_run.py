# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read controller-serialized benchmark runs from trusted paths."""

from pathlib import Path

from aiperf.common.path_safety import safe_read_template_path
from aiperf.kubernetes.environment import K8sEnvironment


def read_serialized_run_json(path: Path) -> str | None:
    """Read a serialized run, including Kubernetes ConfigMap volume symlinks.

    Ordinary files use the repository-wide safe reader. Kubernetes projects
    ConfigMaps through an atomic ``..data`` symlink, so the narrow fallback
    accepts only a direct-child symlink of the configured mount whose resolved
    regular-file target remains inside the current ``..data`` directory.
    """
    run_json = safe_read_template_path(str(path))
    if run_json is not None:
        return run_json

    mount_root = Path(K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH)
    if (
        not path.is_absolute()
        or not mount_root.is_absolute()
        or path.parent != mount_root
    ):
        return None

    try:
        if mount_root.is_symlink() or not path.is_symlink():
            return None
        data_root = (mount_root / "..data").resolve(strict=True)
        resolved = path.resolve(strict=True)
        if (
            not data_root.is_dir()
            or not resolved.is_relative_to(data_root)
            or not resolved.is_file()
        ):
            return None
        return safe_read_template_path(str(resolved))
    except (OSError, RuntimeError, UnicodeError, ValueError):
        return None
