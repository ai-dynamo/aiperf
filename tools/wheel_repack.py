# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inject the native `aiperf` binary into a built wheel's scripts directory.

maturin (``bindings = "pyo3"``) cannot install a native executable as the
``aiperf`` console script directly (``bindings = "bin"`` is illegal alongside a
pyo3 module + ``[project.scripts]``). So we build the wheel normally, then repack
it here: the compiled binary is added at ``<distribution>-<version>.data/scripts/
aiperf`` (mode 0755) and the ``RECORD`` is rewritten with its hash + size. pip
installs files under ``*.data/scripts/`` straight into the environment's bin
directory (PEP 427), so the ``aiperf`` command becomes the ELF binary itself with
no Python launcher shim.

Usage:
    python tools/wheel_repack.py --wheel-dir dist --binary target/release/aiperf

The wheel is edited in place (rewritten). Re-running on an already-repacked wheel
replaces the injected binary, so the step is idempotent.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import stat
import sys
import zipfile
from pathlib import Path

_SCRIPT_NAME = "aiperf"
# Regular file (S_IFREG), rwxr-xr-x, packed into the HIGH 16 bits of external_attr
# as the unix st_mode. pip's `zip_item_is_executable` computes `mode =
# external_attr >> 16` and requires `stat.S_ISREG(mode)` AND `mode & 0o111`, so the
# S_IFREG type bits MUST be part of the shifted mode (not OR'd into the low word) or
# pip installs the file non-executable and `aiperf` fails with Permission denied.
_EXEC_EXTERNAL_ATTR = (stat.S_IFREG | 0o755) << 16


def _record_hash(data: bytes) -> str:
    """Return the RECORD digest string ``sha256=<urlsafe-b64, no padding>``."""
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _find_wheel(wheel_dir: Path) -> Path:
    """Return the single aiperf wheel in ``wheel_dir`` (newest if several)."""
    wheels = sorted(
        wheel_dir.glob("aiperf-*.whl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not wheels:
        raise FileNotFoundError(f"no aiperf-*.whl found in {wheel_dir}")
    return wheels[0]


def _dist_and_version(wheel: Path) -> tuple[str, str]:
    """Parse ``{distribution}-{version}-...whl`` -> (distribution, version)."""
    parts = wheel.stem.split("-")
    if len(parts) < 2:
        raise ValueError(f"unexpected wheel filename: {wheel.name}")
    return parts[0], parts[1]


def repack(wheel: Path, binary: Path) -> None:
    """Inject ``binary`` into ``wheel`` as the data-scripts ``aiperf`` command."""
    distribution, version = _dist_and_version(wheel)
    script_arcname = f"{distribution}-{version}.data/scripts/{_SCRIPT_NAME}"
    record_arcname = f"{distribution}-{version}.dist-info/RECORD"

    binary_bytes = binary.read_bytes()
    binary_record_line = (
        f"{script_arcname},{_record_hash(binary_bytes)},{len(binary_bytes)}"
    )

    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        if record_arcname not in names:
            raise FileNotFoundError(f"{record_arcname} missing from {wheel.name}")
        # Preserve every entry except the one we replace/insert and RECORD, which
        # we regenerate from the surviving lines + the new binary line.
        preserved = [
            (info, zf.read(info.filename))
            for info in zf.infolist()
            if info.filename not in (script_arcname, record_arcname)
        ]
        record_text = zf.read(record_arcname).decode("utf-8")

    kept_lines = [
        line
        for line in record_text.splitlines()
        if line.strip()
        and not line.startswith(f"{script_arcname},")
        and not line.startswith(f"{record_arcname},")
    ]
    kept_lines.append(binary_record_line)
    # RECORD lists itself last with empty hash/size (PEP 376).
    kept_lines.append(f"{record_arcname},,")
    new_record = ("\n".join(kept_lines) + "\n").encode("utf-8")

    tmp = wheel.with_suffix(".whl.repack")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for info, data in preserved:
            out.writestr(info, data)
        script_info = zipfile.ZipInfo(script_arcname)
        script_info.external_attr = _EXEC_EXTERNAL_ATTR
        script_info.compress_type = zipfile.ZIP_DEFLATED
        out.writestr(script_info, binary_bytes)
        out.writestr(record_arcname, new_record)
    tmp.replace(wheel)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        default=Path("dist"),
        help="directory containing the built aiperf wheel (default: dist)",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help="explicit wheel path (overrides --wheel-dir discovery)",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("target/release/aiperf"),
        help="path to the compiled aiperf binary to inject",
    )
    args = parser.parse_args(argv)

    binary = args.binary
    if not binary.is_file():
        print(f"error: binary not found: {binary}", file=sys.stderr)
        return 1
    wheel = args.wheel or _find_wheel(args.wheel_dir)
    repack(wheel, binary)
    print(f"repacked {wheel.name}: injected {binary} as scripts/{_SCRIPT_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
