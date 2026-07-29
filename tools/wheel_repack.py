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
    python tools/wheel_repack.py --wheel-dir dist --binary rust/target/release/aiperf

The wheel is edited in place (rewritten). Re-running on an already-repacked wheel
replaces the injected binary, so the step is idempotent.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import platform
import stat
import struct
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

# ELF constants. Only the fields this scan reads are named.
_ELF_MAGIC = b"\x7fELF"
_ELFCLASS64 = 2
_ELFDATA2LSB = 1
_SHT_GNU_verneed = 0x6FFFFFFE


def glibc_versions(binary: Path) -> set[tuple[int, int]]:
    """Every ``GLIBC_x.y`` version need declared by a 64-bit LE ELF.

    Reads ``.gnu.version_r`` (``SHT_GNU_verneed``) directly — the same table
    ``objdump -T`` prints and auditwheel's ``versioned_symbols`` check consults.
    Implemented in-tree so the packaging path needs no auditwheel install.
    """
    data = binary.read_bytes()
    if data[:4] != _ELF_MAGIC:
        raise ValueError(f"{binary} is not an ELF file")
    if data[4] != _ELFCLASS64 or data[5] != _ELFDATA2LSB:
        raise ValueError(f"{binary} is not a 64-bit little-endian ELF")

    # Elf64_Ehdr: e_shoff at 0x28, e_shentsize at 0x3A, e_shnum at 0x3C.
    (e_shoff,) = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)

    versions: set[tuple[int, int]] = set()
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        # Elf64_Shdr: sh_type at +0x04, sh_offset at +0x18, sh_link at +0x28.
        (sh_type,) = struct.unpack_from("<I", data, off + 0x04)
        if sh_type != _SHT_GNU_verneed:
            continue
        (sh_offset,) = struct.unpack_from("<Q", data, off + 0x18)
        (sh_link,) = struct.unpack_from("<I", data, off + 0x28)
        # sh_link names the associated string table (.dynstr).
        (strtab_off,) = struct.unpack_from(
            "<Q", data, e_shoff + sh_link * e_shentsize + 0x18
        )
        versions |= _read_verneed(data, sh_offset, strtab_off)
    return versions


def _read_verneed(
    data: bytes, verneed_off: int, strtab_off: int
) -> set[tuple[int, int]]:
    """Walk the Elf64_Verneed / Elf64_Vernaux chains, collecting GLIBC_x.y names."""
    versions: set[tuple[int, int]] = set()
    need_off = verneed_off
    while True:
        # Elf64_Verneed: vn_cnt at +0x02, vn_aux at +0x08, vn_next at +0x0C.
        (vn_cnt,) = struct.unpack_from("<H", data, need_off + 0x02)
        vn_aux, vn_next = struct.unpack_from("<II", data, need_off + 0x08)
        aux_off = need_off + vn_aux
        for _ in range(vn_cnt):
            # Elf64_Vernaux: vna_name at +0x08, vna_next at +0x0C.
            vna_name, vna_next = struct.unpack_from("<II", data, aux_off + 0x08)
            name = _cstr(data, strtab_off + vna_name)
            if name.startswith("GLIBC_"):
                parsed = _parse_glibc_version(name)
                if parsed is not None:
                    versions.add(parsed)
            if vna_next == 0:
                break
            aux_off += vna_next
        if vn_next == 0:
            break
        need_off += vn_next
    return versions


def _cstr(data: bytes, offset: int) -> str:
    end = data.index(b"\0", offset)
    return data[offset:end].decode("utf-8", "replace")


def _parse_glibc_version(name: str) -> tuple[int, int] | None:
    """``"GLIBC_2.39"`` -> ``(2, 39)``. Returns None for private/odd names."""
    parts = name.removeprefix("GLIBC_").split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        # e.g. GLIBC_PRIVATE — not a public version floor.
        return None


def manylinux_tag(floor: tuple[int, int], machine: str) -> str:
    """PEP 600 tag for a glibc floor: ``((2, 39), "x86_64")`` -> ``manylinux_2_39_x86_64``."""
    major, minor = floor
    return f"manylinux_{major}_{minor}_{machine}"


def platform_tag_for(binary: Path) -> str:
    """The PEP 600 platform tag the injected ``binary`` actually requires."""
    versions = glibc_versions(binary)
    if not versions:
        raise ValueError(f"{binary} declares no GLIBC_ version needs")
    return manylinux_tag(max(versions), platform.machine())


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
        default=Path("rust/target/release/aiperf"),
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
