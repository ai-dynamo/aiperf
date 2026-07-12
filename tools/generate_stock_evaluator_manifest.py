# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and verify the relocatable stock evaluator launch manifest.

The generator runs only against the exact audited development closure.  It
freezes every copied regular file by its worker-root-relative destination and
raw SHA-256.  Distribution ``RECORD`` files are independently hashed and then
expanded; the runner repeats all checks before copying bytes into a fresh
single-link rootfs.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from aiperf.accuracy.evaluation.canonical import canonical_dumps, canonical_sha256
from aiperf.accuracy.evaluation.distributions import (
    ISOLATION_PROFILE,
    NEMO_EVALUATOR_DISTRIBUTION,
    OPENBENCH_DISTRIBUTION,
    STOCK_DISTRIBUTIONS,
    SourceComponentLock,
    StockDistributionDescriptor,
    task_manifest,
)
from aiperf.accuracy.evaluation.operation_schemas import (
    OPERATION_DIRECTION_SCHEMA_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "src/aiperf/accuracy/evaluation/manifests"
OUTPUT = MANIFEST_DIR / "stock_distributions.json"
PYTHON_VERSION = "3.12.10"
PYTHON_ABI = "cpython-312-x86_64-linux-gnu"
PLATFORM = "linux-x86_64"
RUNTIME_CLOSURE_ID = "cpython_3_12_10_linux_x86_64"
SITE_PACKAGES_DESTINATION = "runtime/lib/python3.12/site-packages"
PROGRAM_DESTINATION = "runtime/bin/python3.12"
CURRENT_DIR = "work"
STOCK_MANIFEST_RESOURCE = (
    "aiperf.accuracy.evaluation.manifests/stock_distributions.json"
)
GSM8K_CANARY_SOURCE = (
    ROOT / "src/aiperf/accuracy/evaluation/manifests/assets/gsm8k_canary.jsonl"
)
GSM8K_CANARY_DESTINATION = "assets/gsm8k_canary.jsonl"

# These are the exact distribution records observed during the complete pinned
# one-case provider proofs. Optional provider SDKs, registries, CLIs, and
# unselected benchmark packages are intentionally absent.
NEMO_RECORD_DISTRIBUTIONS = """
PyYAML aiohappyeyeballs aiohttp aiosignal annotated-types attrs brotli
frozenlist idna multidict nemo-evaluator numpy orjson propcache pydantic
pydantic_core setuptools typing-inspection typing_extensions yarl
""".split()  # noqa: SIM905 - compact audited inventory.

OPENBENCH_RECORD_DISTRIBUTIONS = """
PyYAML Pygments aiobotocore aiohappyeyeballs aiohttp aioitertools aiosignal
annotated-types anthropic anyio attrs boto3 botocore brotli certifi click
distro docstring_parser frozenlist fsspec h11 httpcore httpx idna ijson
inspect_ai jiter jmespath jsonlines jsonpatch jsonpointer jsonschema
jsonschema-specifications linkify-it-py markdown-it-py mdurl mmh3 multidict
nest-asyncio numpy openai openbench orjson platformdirs propcache psutil
pydantic pydantic_core python-dateutil python-dotenv referencing rich rpds-py
s3fs semver setuptools shortuuid six sniffio tenacity textual tiktoken tornado
typing-inspection typing_extensions uc-micro-py urllib3 wrapt yarl zstandard
""".split()  # noqa: SIM905 - compact audited inventory.


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


_COMPONENT_BY_DISTRIBUTION = {
    _canonical_distribution_name(component.distribution): component
    for descriptor in STOCK_DISTRIBUTIONS.values()
    for component in descriptor.components
}
_ELF_INTERPRETER = re.compile(r"Requesting program interpreter:\s*([^\]]+)\]")
_ALLOWED_WORKER_SUFFIXES = {".py", ".json", ".jsonl", ".toml", ".yaml", ".yml"}
_INSPECT_SOURCE_SUPPLEMENTS_BASE64 = {
    "tool/_tools/_computer/_resources/image_home_dir/.config/Code/User/settings.json": "ewogICJzZWN1cml0eS53b3Jrc3BhY2UudHJ1c3QuZW5hYmxlZCI6IGZhbHNlLAogICJ1cGRhdGUuc2hvd1JlbGVhc2VOb3RlcyI6IGZhbHNlLAogICJlZGl0b3IuY3Vyc29yQmxpbmtpbmciOiAic29saWQiLAogICJlZGl0b3IuY3Vyc29yV2lkdGgiOiAzLAogICJ3b3JrYmVuY2guY29sb3JDdXN0b21pemF0aW9ucyI6IHsKICAgICJlZGl0b3JDdXJzb3IuZm9yZWdyb3VuZCI6ICIjRkYwMDAwIgogIH0KfQo=",
    "tool/_tools/_computer/_resources/tool/.vscode/settings.json": "ewogICJjU3BlbGwud29yZHMiOiBbCiAgICAiRldYR0EiLAogICAgImdldG1vdXNlbG9jYXRpb24iLAogICAgImtleXVwIiwKICAgICJtb3VzZWRvd24iLAogICAgIm1vdXNlbW92ZSIsCiAgICAibW91c2V1cCIsCiAgICAic2Nyb3QiLAogICAgIldYR0EiCiAgXQp9Cg==",
    "tool/_tools/_computer/_resources/tool/pyproject.toml": "W2J1aWxkLXN5c3RlbV0KcmVxdWlyZXMgPSBbInNldHVwdG9vbHM+PTY0IiwgInNldHVwdG9vbHNfc2NtW3RvbWxdPj04Il0KYnVpbGQtYmFja2VuZCA9ICJzZXR1cHRvb2xzLmJ1aWxkX21ldGEiCgpbdG9vbC5zZXR1cHRvb2xzX3NjbV0KClt0b29sLnNldHVwdG9vbHMucGFja2FnZXMuZmluZF0Kd2hlcmUgPSBbIi4iXQppbmNsdWRlID0gWyJpbnNwZWN0X2FpKiJdCgpbdG9vbC5ydWZmXQpzcmMgPSBbIi4iXQoKW3Rvb2wucnVmZi5saW50XQpzZWxlY3QgPSBbCiAgICAiRSIsICAgICAgIyBweWNvZGVzdHlsZSBlcnJvcnMKICAgICJXIiwgICAgICAjIHB5Y29kZXN0eWxlIHdhcm5pbmdzCiAgICAiRiIsICAgICAgIyBmbGFrZTgKICAgICJEIiwgICAgICAjIHB5ZG9jc3R5bGUKICAgICJJIiwgICAgICAjIGlzb3J0CiAgICAiU0lNMTAxIiwgIyBkdXBsaWNhdGUgaXNpbnN0YW5jZQogICAgIyAiUkVUIiwgIyBmbGFrZTgtcmV0dXJuCiAgICAjICJSVUYiLCAjIHJ1ZmYgcnVsZXMKXQppZ25vcmUgPSBbIkUyMDMiLCAiRTUwMSIsICJEMTAiLCAiRDIxMiIsICJENDE1Il0KClt0b29sLnJ1ZmYubGludC5weWRvY3N0eWxlXQpjb252ZW50aW9uID0gImdvb2dsZSIKClt0b29sLnB5dGVzdC5pbmlfb3B0aW9uc10KbWludmVyc2lvbiA9ICI3LjAiCmFkZG9wdHMgPSAiLXJBIC0tZG9jdGVzdC1tb2R1bGVzIC0tY29sb3I9eWVzIgpkb2N0ZXN0X29wdGlvbmZsYWdzID0gWyJOT1JNQUxJWkVfV0hJVEVTUEFDRSIsICJJR05PUkVfRVhDRVBUSU9OX0RFVEFJTCJdCmFzeW5jaW9fbW9kZSA9ICJhdXRvIgphc3luY2lvX2RlZmF1bHRfZml4dHVyZV9sb29wX3Njb3BlID0gImZ1bmN0aW9uIgpsb2dfbGV2ZWwgPSAid2FybmluZyIKClt0b29sLm15cHldCndhcm5fdW51c2VkX2lnbm9yZXMgPSB0cnVlCm5vX2ltcGxpY2l0X3JlZXhwb3J0ID0gdHJ1ZQpzdHJpY3RfZXF1YWxpdHkgPSB0cnVlCndhcm5fcmVkdW5kYW50X2Nhc3RzID0gdHJ1ZQp3YXJuX3VudXNlZF9jb25maWdzID0gdHJ1ZQpkaXNhbGxvd19hbnlfZXhwbGljaXQgPSB0cnVlCmRpc2FsbG93X2FueV9nZW5lcmljcyA9IHRydWUKZGlzYWxsb3dfc3ViY2xhc3NpbmdfYW55ID0gdHJ1ZQpwbHVnaW5zPVsicHlkYW50aWMubXlweSJdCgoKW3Rvb2wucHlkYW50aWMtbXlweV0KaW5pdF9mb3JiaWRfZXh0cmEgPSB0cnVlCmluaXRfdHlwZWQgPSB0cnVlCgpbdG9vbC5jaGVjay13aGVlbC1jb250ZW50c10KaWdub3JlID0gWyJXMDAyIiwgIlcwMDkiXQoKW3Byb2plY3RdCm5hbWUgPSAid2ViX2Jyb3dzZXJfdG9vbF9jb250YWluZXIiCnJlcXVpcmVzLXB5dGhvbiA9ICI+PTMuMTAiCmR5bmFtaWMgPSBbInZlcnNpb24iLCAiZGVwZW5kZW5jaWVzIl0KCgpbcHJvamVjdC5vcHRpb25hbC1kZXBlbmRlbmNpZXNdCmRldiA9IFsicHl0ZXN0Il0=",
}


class ManifestGenerationError(RuntimeError):
    """The installed closure did not match the frozen generator contract."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _normalize_relative(value: str) -> str:
    if not value or value.startswith("/") or "\\" in value or "\x00" in value:
        raise ManifestGenerationError(f"invalid logical relative path {value!r}")
    parts: list[str] = []
    for part in value.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise ManifestGenerationError(f"logical path escaped root: {value!r}")
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        raise ManifestGenerationError(f"logical path normalized empty: {value!r}")
    return "/".join(parts)


def _destination(prefix: str, relative: str) -> str:
    return _normalize_relative(f"{prefix}/{relative}")


def _is_executable(path: Path) -> bool:
    return bool(path.stat().st_mode & 0o111)


def _logical_file(
    source_relative_path: str,
    destination: str,
    content: bytes,
    executable: bool,
) -> dict[str, Any]:
    return {
        "source_relative_path": _normalize_relative(source_relative_path),
        "destination": _normalize_relative(destination),
        "artifact_content_sha256": _sha256(content),
        "executable": executable,
    }


def _runtime_root() -> Path:
    root = Path(sys.base_prefix).resolve(strict=True)
    executable = root / "bin/python3.12"
    if (
        sys.implementation.name != "cpython"
        or ".".join(map(str, sys.version_info[:3])) != PYTHON_VERSION
        or sysconfig.get_config_var("SOABI") != PYTHON_ABI
        or not executable.is_file()
        or executable.is_symlink()
    ):
        raise ManifestGenerationError("generator is not running on pinned CPython")
    return root


def _runtime_files(root: Path) -> tuple[dict[str, Any], ...]:
    candidates = [root / "bin/python3.12", root / "lib/libpython3.12.so.1.0"]
    stdlib = root / "lib/python3.12"
    candidates.extend(
        path
        for path in stdlib.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and "site-packages" not in path.parts
    )
    result = []
    for path in sorted(
        set(candidates), key=lambda item: item.relative_to(root).as_posix().encode()
    ):
        relative = path.relative_to(root).as_posix()
        result.append(
            _logical_file(
                relative,
                f"runtime/{relative}",
                path.read_bytes(),
                _is_executable(path),
            )
        )
    return tuple(result)


def _distribution(name: str) -> importlib.metadata.Distribution:
    try:
        return importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise ManifestGenerationError(f"missing distribution {name!r}") from error


def _record_path(distribution: importlib.metadata.Distribution) -> Path:
    path = Path(distribution._path) / "RECORD"  # type: ignore[attr-defined]
    if not path.is_file() or path.is_symlink():
        raise ManifestGenerationError(
            f"distribution {distribution.metadata['Name']!r} lacks a regular RECORD"
        )
    return path


def _record_rows(record: bytes) -> tuple[tuple[str, str, str], ...]:
    try:
        text = record.decode("utf-8", errors="strict")
        rows = tuple(tuple(row) for row in csv.reader(io.StringIO(text)))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ManifestGenerationError("distribution RECORD is malformed") from error
    if not rows or any(len(row) != 3 for row in rows):
        raise ManifestGenerationError("distribution RECORD has an invalid image")
    return rows  # type: ignore[return-value]


def _record_digest(encoded: str) -> str:
    try:
        algorithm, value = encoded.split("=", 1)
        if algorithm != "sha256":
            raise ValueError(algorithm)
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).hex()
    except ValueError as error:
        raise ManifestGenerationError("RECORD uses a non-SHA256 digest") from error


def _record_info(name: str) -> dict[str, Any]:
    distribution = _distribution(name)
    record = _record_path(distribution).read_bytes()
    return {
        "distribution": distribution.metadata["Name"],
        "version": distribution.version,
        "record_sha256": _sha256(record),
    }


def _source_omissions(component: SourceComponentLock) -> tuple[str, ...]:
    distribution = _distribution(component.distribution)
    dist_info = Path(distribution._path).name  # type: ignore[attr-defined]
    values = {
        f"{dist_info}/METADATA",
        f"{dist_info}/RECORD",
        f"{dist_info}/direct_url.json",
    }
    if component.distribution == "openbench":
        values.add(f"{dist_info}/entry_points.txt")
    return tuple(sorted(values, key=str.encode))


def _record_closures(names: Sequence[str]) -> tuple[dict[str, Any], ...]:
    result = []
    for name in names:
        info = _record_info(name)
        component = _COMPONENT_BY_DISTRIBUTION.get(
            _canonical_distribution_name(info["distribution"])
        )
        result.append(
            {
                **info,
                "destination_prefix": SITE_PACKAGES_DESTINATION,
                "omitted_paths": (
                    list(_source_omissions(component)) if component else []
                ),
            }
        )
    return tuple(sorted(result, key=lambda item: item["distribution"].lower().encode()))


def _lock_value(
    descriptor: StockDistributionDescriptor,
    record_closures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "format": "aiperf-evaluator-lock-v2",
        "provider": descriptor.provider_id,
        "distribution": descriptor.distribution_id,
        "python_environment": {
            "implementation": "cpython",
            "version": PYTHON_VERSION,
            "abi": PYTHON_ABI,
            "platform": PLATFORM,
        },
        "sources": [
            {
                "package": component.distribution,
                "version": component.version,
                "commit": component.commit,
                "source_tree_sha256": component.source_tree_sha256,
            }
            for component in descriptor.components
        ],
        "audit_scope": {
            "task": "gsm8k",
            "asset": "openai_gsm8k_main_test_canary",
            "external_effects": ["model.generate"],
            "network": "rust_host_only",
        },
        "record_closures": [
            {
                "distribution": item["distribution"],
                "version": item["version"],
                "record_sha256": item["record_sha256"],
                "omitted_paths": item["omitted_paths"],
            }
            for item in record_closures
        ],
        "installation": {
            "kind": "rust-materialized-relocatable-rootfs",
            "verify_every_record_entry": True,
            "reject_symlinks_and_unhashed_files": True,
            "source_attestation_overlay": True,
        },
    }


def _provider_source_digest(descriptor: StockDistributionDescriptor) -> str:
    return canonical_sha256(
        [
            {
                "distribution": component.distribution,
                "version": component.version,
                "commit": component.commit,
                "source_tree_sha256": component.source_tree_sha256,
            }
            for component in descriptor.components
        ]
    )


def _verify_component_sources() -> None:
    for component in _COMPONENT_BY_DISTRIBUTION.values():
        spec = importlib.util.find_spec(component.import_package)
        if spec is None or not spec.submodule_search_locations:
            raise ManifestGenerationError(
                f"source component {component.import_package!r} is unavailable"
            )
        locations = tuple(spec.submodule_search_locations)
        if len(locations) != 1:
            raise ManifestGenerationError("source component is namespace ambiguous")
        supplements = (
            {
                relative: base64.b64decode(content, validate=True)
                for relative, content in _INSPECT_SOURCE_SUPPLEMENTS_BASE64.items()
            }
            if component.distribution == "inspect-ai"
            else {}
        )
        actual = _projected_source_tree_sha256(
            Path(locations[0]), component.source_tree_excluded_paths, supplements
        )
        if actual != component.source_tree_sha256:
            raise ManifestGenerationError(
                f"source tree drift for {component.distribution!r}: {actual}"
            )


def _projected_source_tree_sha256(
    root: Path,
    excluded: Sequence[str],
    supplements: Mapping[str, bytes],
) -> str:
    allowed = {".py", ".json", ".toml", ".yaml", ".yml"}
    semantic = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in allowed
        and "__pycache__" not in path.parts
        and path.relative_to(root).as_posix() not in excluded
    }
    for relative, content in supplements.items():
        if relative in semantic:
            raise ManifestGenerationError(f"source supplement collision: {relative}")
        semantic[relative] = content
    digest = hashlib.sha256()
    for relative, content in sorted(
        semantic.items(), key=lambda item: item[0].encode()
    ):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _embedded_file(resource: str, destination: str, content: bytes) -> dict[str, Any]:
    return {
        "resource": resource,
        "destination": _normalize_relative(destination),
        "artifact_content_sha256": _sha256(content),
        "executable": False,
        "content_base64": base64.b64encode(content).decode("ascii"),
    }


def _worker_resource_contents(
    lock_overrides: Mapping[Path, bytes],
) -> tuple[tuple[Path, str, bytes], ...]:
    roots = [
        ROOT / "src/aiperf/__init__.py",
        ROOT / "src/aiperf/accuracy/__init__.py",
    ]
    evaluation = ROOT / "src/aiperf/accuracy/evaluation"
    roots.extend(
        path
        for path in evaluation.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in _ALLOWED_WORKER_SUFFIXES
        and path.name != "stock_distributions.json"
    )
    result = []
    for path in sorted(set(roots), key=lambda item: item.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix()
        content = lock_overrides.get(path, path.read_bytes())
        destination = f"{SITE_PACKAGES_DESTINATION}/{relative.removeprefix('src/')}"
        result.append((path, _normalize_relative(destination), content))
    return tuple(result)


def _worker_source_digest(resources: Sequence[tuple[Path, str, bytes]]) -> str:
    evaluation = ROOT / "src/aiperf/accuracy/evaluation"
    semantic = []
    for path, _, content in resources:
        if not path.is_relative_to(evaluation):
            continue
        relative = path.relative_to(evaluation).as_posix()
        if path.suffix not in {".py", ".json", ".toml", ".yaml", ".yml"}:
            continue
        semantic.append((relative, content))
    digest = hashlib.sha256()
    for relative, content in sorted(semantic, key=lambda item: item[0].encode()):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _metadata_overlay(component: SourceComponentLock) -> dict[str, Any]:
    distribution = _distribution(component.distribution)
    record_path = _record_path(distribution)
    upstream_record = record_path.read_bytes()
    rows = _record_rows(upstream_record)
    omitted = set(_source_omissions(component))
    dist_info = Path(distribution._path).name  # type: ignore[attr-defined]
    metadata_relative = f"{dist_info}/METADATA"
    record_relative = f"{dist_info}/RECORD"
    attestation_relative = f"{dist_info}/aiperf_source_attestation.json"
    metadata = (Path(distribution._path) / "METADATA").read_bytes()  # type: ignore[attr-defined]
    attestation = canonical_dumps(
        {
            "distribution": component.distribution,
            "version": component.version,
            "commit": component.commit,
            "source_tree_sha256": component.source_tree_sha256,
        }
    )
    final_rows = [row for row in rows if row[0] not in omitted]
    for relative, content in (
        (metadata_relative, metadata),
        (attestation_relative, attestation),
    ):
        encoded = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .decode("ascii")
            .rstrip("=")
        )
        final_rows.append((relative, f"sha256={encoded}", str(len(content))))
    final_rows.append((record_relative, "", ""))
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(sorted(final_rows, key=lambda row: row[0].encode("utf-8")))
    replacement_record = output.getvalue().encode("utf-8")
    policy = (
        "inspect_entry_point_policy_deny_v1"
        if component.distribution == "openbench"
        else "aiperf_source_attestation_v1"
    )
    replacements = (
        (metadata_relative, metadata),
        (attestation_relative, attestation),
        (record_relative, replacement_record),
    )
    return {
        "policy_id": policy,
        "distribution": distribution.metadata["Name"],
        "version": component.version,
        "upstream_record_sha256": _sha256(upstream_record),
        "omitted_paths": sorted(omitted, key=str.encode),
        "replacement_files": [
            _embedded_file(
                f"generated://metadata-overlay/{component.distribution}/{relative}",
                f"{SITE_PACKAGES_DESTINATION}/{relative}",
                content,
            )
            for relative, content in replacements
        ],
    }


def _expand_record_closure(
    closure: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], Path], ...]:
    distribution = _distribution(str(closure["distribution"]))
    if distribution.version != closure["version"]:
        raise ManifestGenerationError(f"version drift for {closure['distribution']!r}")
    record_path = _record_path(distribution)
    record = record_path.read_bytes()
    if _sha256(record) != closure["record_sha256"]:
        raise ManifestGenerationError(f"RECORD drift for {closure['distribution']!r}")
    omitted = set(closure["omitted_paths"])
    result = []
    for relative, encoded, size in _record_rows(record):
        if relative in omitted:
            continue
        source = Path(distribution.locate_file(PurePosixPath(relative)))
        if not source.is_file():
            raise ManifestGenerationError(f"RECORD file is absent: {source}")
        content = source.read_bytes()
        expected = _sha256(record) if not encoded else _record_digest(encoded)
        if _sha256(content) != expected:
            raise ManifestGenerationError(f"RECORD content drift: {source}")
        if size and len(content) != int(size):
            raise ManifestGenerationError(f"RECORD size drift: {source}")
        destination = _destination(str(closure["destination_prefix"]), relative)
        result.append(
            (
                _logical_file(
                    destination,
                    destination,
                    content,
                    _is_executable(source),
                ),
                source,
            )
        )
    return tuple(result)


def _elf_inputs(
    runtime_root: Path,
    runtime_files: Sequence[Mapping[str, Any]],
    record_closures: Sequence[Mapping[str, Any]],
) -> tuple[Path, ...]:
    result: set[Path] = set()
    for item in runtime_files:
        source = runtime_root / str(item["source_relative_path"])
        if source.read_bytes()[:4] == b"\x7fELF":
            result.add(source)
    for closure in record_closures:
        for _, source in _expand_record_closure(closure):
            try:
                if source.read_bytes()[:4] == b"\x7fELF":
                    result.add(source)
            except OSError as error:
                raise ManifestGenerationError(
                    f"failed to inspect ELF {source}"
                ) from error
    return tuple(sorted(result, key=lambda path: os.fsencode(path)))


def _ldd_dependencies(path: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["ldd", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "not a dynamic executable" in result.stderr:
            return ()
        raise ManifestGenerationError(f"ldd failed for {path}: {result.stderr.strip()}")
    dependencies: set[Path] = set()
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if "=> not found" in line:
            # Managed CPython carries optional stdlib extensions (for example
            # ``_crypt``) whose platform library is absent. They are not in the
            # audited provider import graph; the contained pair proof detects
            # any unresolved dependency that is actually selected.
            continue
        candidate: str | None = None
        if "=>" in line:
            right = line.split("=>", 1)[1].strip()
            if right.startswith("/"):
                candidate = right.split(" (", 1)[0]
        elif line.startswith("/"):
            candidate = line.split(" (", 1)[0]
        if candidate:
            dependencies.add(Path(candidate))
    readelf = subprocess.run(
        ["readelf", "-l", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    if match := _ELF_INTERPRETER.search(readelf.stdout):
        dependencies.add(Path(match.group(1)))
    return tuple(sorted(dependencies, key=lambda item: os.fsencode(item)))


def _system_files(
    runtime_root: Path,
    runtime_files: Sequence[Mapping[str, Any]],
    record_closures: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    site_packages = Path(next(iter(importlib.metadata.distributions())).locate_file(""))
    queue = list(_elf_inputs(runtime_root, runtime_files, record_closures))
    system: dict[str, Path] = {}
    visited: set[Path] = set()
    while queue:
        current = queue.pop()
        resolved_current = current.resolve(strict=True)
        if resolved_current in visited:
            continue
        visited.add(resolved_current)
        for dependency in _ldd_dependencies(current):
            resolved = dependency.resolve(strict=True)
            if resolved.is_relative_to(runtime_root) or resolved.is_relative_to(
                site_packages
            ):
                continue
            destination = _normalize_relative(dependency.as_posix().lstrip("/"))
            prior = system.get(destination)
            if prior is not None and prior.read_bytes() != resolved.read_bytes():
                raise ManifestGenerationError(
                    f"system destination collision at {destination!r}"
                )
            system[destination] = resolved
            queue.append(resolved)
    return tuple(
        _logical_file(
            destination,
            destination,
            source.read_bytes(),
            _is_executable(source),
        )
        for destination, source in sorted(
            system.items(), key=lambda item: item[0].encode()
        )
    )


def _insert_inventory(
    inventory: dict[str, tuple[str, bool]],
    destination: str,
    digest: str,
    executable: bool,
) -> None:
    normalized = _normalize_relative(destination)
    prior = inventory.get(normalized)
    value = (digest, executable)
    if prior is not None and prior != value:
        raise ManifestGenerationError(f"launch destination collision: {normalized}")
    if prior is not None:
        raise ManifestGenerationError(f"duplicate launch destination: {normalized}")
    inventory[normalized] = value


def _expanded_inventory(
    shared_closure_ids: Sequence[str],
    shared_closures: Mapping[str, Mapping[str, Any]],
    record_closures: Sequence[Mapping[str, Any]],
    embedded_files: Sequence[Mapping[str, Any]],
    overlays: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, bool]]:
    inventory: dict[str, tuple[str, bool]] = {}
    for closure_id in shared_closure_ids:
        for item in shared_closures[closure_id]["files"]:
            _insert_inventory(
                inventory,
                str(item["destination"]),
                str(item["artifact_content_sha256"]),
                bool(item["executable"]),
            )
    for closure in record_closures:
        for item, _ in _expand_record_closure(closure):
            _insert_inventory(
                inventory,
                item["destination"],
                item["artifact_content_sha256"],
                item["executable"],
            )
    for item in embedded_files:
        _insert_inventory(
            inventory,
            str(item["destination"]),
            str(item["artifact_content_sha256"]),
            bool(item["executable"]),
        )
    for overlay in overlays:
        for item in overlay["replacement_files"]:
            _insert_inventory(
                inventory,
                str(item["destination"]),
                str(item["artifact_content_sha256"]),
                bool(item["executable"]),
            )
    return inventory


def _closure_sha256(inventory: Mapping[str, tuple[str, bool]]) -> str:
    digest = hashlib.sha256()
    for path, (content_digest, _) in sorted(
        inventory.items(), key=lambda item: item[0].encode("utf-8")
    ):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _distribution_entry(
    descriptor: StockDistributionDescriptor,
    lock_bytes: bytes,
    record_closures: Sequence[Mapping[str, Any]],
    shared_closure_ids: Sequence[str],
    shared_closures: Mapping[str, Mapping[str, Any]],
    worker_resources: Sequence[tuple[Path, str, bytes]],
) -> dict[str, Any]:
    embedded_files = [
        _embedded_file(path.relative_to(ROOT).as_posix(), destination, content)
        for path, destination, content in worker_resources
    ]
    embedded_files.append(
        _embedded_file(
            GSM8K_CANARY_SOURCE.relative_to(ROOT).as_posix(),
            GSM8K_CANARY_DESTINATION,
            GSM8K_CANARY_SOURCE.read_bytes(),
        )
    )
    if any(
        component.distribution == "inspect-ai" for component in descriptor.components
    ):
        embedded_files.extend(
            _embedded_file(
                f"pinned-source://inspect-ai/{relative}",
                f"{SITE_PACKAGES_DESTINATION}/inspect_ai/{relative}",
                base64.b64decode(encoded, validate=True),
            )
            for relative, encoded in sorted(
                _INSPECT_SOURCE_SUPPLEMENTS_BASE64.items(),
                key=lambda item: item[0].encode(),
            )
        )
    overlays = [_metadata_overlay(component) for component in descriptor.components]
    inventory = _expanded_inventory(
        shared_closure_ids,
        shared_closures,
        record_closures,
        embedded_files,
        overlays,
    )
    manifest_resource = descriptor.task_manifest_resource
    manifest_path = MANIFEST_DIR / manifest_resource
    manifest_bytes = manifest_path.read_bytes()
    operation = OPERATION_DIRECTION_SCHEMA_SHA256["model.generate"]
    return {
        "provider_id": descriptor.provider_id,
        "distribution_id": descriptor.distribution_id,
        "package": descriptor.package,
        "package_version": descriptor.package_version,
        "provider_source_sha256": _provider_source_digest(descriptor),
        "worker_source_sha256": _worker_source_digest(worker_resources),
        "dependency_lock_sha256": _sha256(lock_bytes),
        "oci_digest": None,
        "config_schema_version": descriptor.config_schema_version,
        "config_schema_sha256": descriptor.config_schema_sha256,
        "task_manifest": {
            "resource": f"aiperf.accuracy.evaluation.manifests/{manifest_resource}",
            "artifact_content_sha256": _sha256(manifest_bytes),
            "executable_tasks": list(
                (
                    task_manifest(descriptor).get("environments", {})
                    if descriptor.provider_id == "nemo_evaluator"
                    else task_manifest(descriptor).get("tasks", {})
                ).keys()
            ),
        },
        "operations": [
            {
                "operation_id": "model.generate",
                "request_schema_sha256": operation["request"],
                "response_schema_sha256": operation["response"],
                "stream_schema_sha256": operation["stream"],
                "endpoint_capability": "chat",
            }
        ],
        "isolation": {
            "profile_id": ISOLATION_PROFILE,
            "bubblewrap": "/usr/bin/bwrap",
            "bubblewrap_sha256": _sha256(Path("/usr/bin/bwrap").read_bytes()),
            "resource_limits": {
                "address_space_bytes": 16 * 1024 * 1024 * 1024,
                "file_size_bytes": 8 * 1024 * 1024 * 1024,
                "open_files": 4096,
                "processes": 1024,
                "cpu_seconds": 86400,
            },
        },
        "launch": {
            "shared_closure_ids": list(shared_closure_ids),
            "python_environment": {
                "implementation": "cpython",
                "version": PYTHON_VERSION,
                "abi": PYTHON_ABI,
                "platform": PLATFORM,
            },
            "program": PROGRAM_DESTINATION,
            "args": list(descriptor.fixed_argv),
            "environment": dict(descriptor.clean_environment),
            "current_dir": CURRENT_DIR,
            "record_closures": list(record_closures),
            "embedded_files": embedded_files,
            "metadata_overlays": overlays,
        },
        "launch_closure_sha256": _closure_sha256(inventory),
    }


def generate() -> tuple[dict[Path, bytes], bytes]:
    """Generate lock resources and the complete stock launch manifest."""
    _verify_component_sources()
    runtime_root = _runtime_root()
    runtime_files = _runtime_files(runtime_root)
    descriptors = (
        (NEMO_EVALUATOR_DISTRIBUTION, NEMO_RECORD_DISTRIBUTIONS),
        (OPENBENCH_DISTRIBUTION, OPENBENCH_RECORD_DISTRIBUTIONS),
    )
    record_sets: dict[str, tuple[dict[str, Any], ...]] = {}
    lock_bytes: dict[Path, bytes] = {}
    for descriptor, names in descriptors:
        closures = _record_closures(names)
        record_sets[descriptor.distribution_id] = closures
        path = MANIFEST_DIR / descriptor.dependency_lock_resource
        lock_bytes[path] = _json_bytes(_lock_value(descriptor, closures))
    worker_resources = _worker_resource_contents(lock_bytes)
    shared_closures: dict[str, dict[str, Any]] = {
        RUNTIME_CLOSURE_ID: {
            "resolver": {
                "kind": "python_runtime_root",
                "implementation": "cpython",
                "version": PYTHON_VERSION,
                "abi": PYTHON_ABI,
                "platform": PLATFORM,
                "executable_relative_path": "bin/python3.12",
                "executable_sha256": _sha256(
                    (runtime_root / "bin/python3.12").read_bytes()
                ),
            },
            "files": list(runtime_files),
        }
    }
    system_ids: dict[str, str] = {}
    for descriptor, _ in descriptors:
        closure_id = f"system_{PLATFORM.replace('-', '_')}_{descriptor.provider_id}"
        system_ids[descriptor.distribution_id] = closure_id
        shared_closures[closure_id] = {
            "resolver": {"kind": "system_root", "platform": PLATFORM},
            "files": list(
                _system_files(
                    runtime_root,
                    runtime_files,
                    record_sets[descriptor.distribution_id],
                )
            ),
        }
    entries = []
    for descriptor, _ in descriptors:
        entries.append(
            _distribution_entry(
                descriptor,
                lock_bytes[MANIFEST_DIR / descriptor.dependency_lock_resource],
                record_sets[descriptor.distribution_id],
                (RUNTIME_CLOSURE_ID, system_ids[descriptor.distribution_id]),
                shared_closures,
                worker_resources,
            )
        )
    value = {
        "schema_version": "aiperf-stock-evaluator-distributions-v1",
        "shared_closures": shared_closures,
        "distributions": entries,
    }
    return lock_bytes, _json_bytes(value)


def _write_or_check(write: bool) -> None:
    lock_bytes, manifest = generate()
    outputs = {**lock_bytes, OUTPUT: manifest}
    mismatches = []
    for path, content in outputs.items():
        if write:
            path.write_bytes(content)
        elif not path.is_file() or path.read_bytes() != content:
            mismatches.append(path.relative_to(ROOT).as_posix())
    if mismatches:
        raise ManifestGenerationError(
            "generated evaluator resources drifted: " + ", ".join(mismatches)
        )


def _source_for_shared(
    closure: Mapping[str, Any], runtime_root: Path, item: Mapping[str, Any]
) -> Path:
    resolver = closure["resolver"]
    if resolver["kind"] == "python_runtime_root":
        return runtime_root / str(item["source_relative_path"])
    if resolver["kind"] == "system_root":
        return Path("/") / str(item["source_relative_path"])
    raise ManifestGenerationError("unknown shared-closure resolver")


def materialize(distribution_id: str, root: Path) -> None:
    """Materialize one generated closure for a no-host-path subprocess proof."""
    _, manifest_bytes = generate()
    manifest = json.loads(manifest_bytes)
    entry = next(
        (
            item
            for item in manifest["distributions"]
            if item["distribution_id"] == distribution_id
        ),
        None,
    )
    if entry is None:
        raise ManifestGenerationError(f"unknown distribution {distribution_id!r}")
    root.mkdir(parents=True, exist_ok=False)
    runtime_root = _runtime_root()
    sources: dict[str, tuple[bytes | Path, bool]] = {}

    def add(destination: str, source: bytes | Path, executable: bool) -> None:
        normalized = _normalize_relative(destination)
        if normalized in sources:
            raise ManifestGenerationError(
                f"duplicate materialization path {normalized}"
            )
        sources[normalized] = (source, executable)

    for closure_id in entry["launch"]["shared_closure_ids"]:
        closure = manifest["shared_closures"][closure_id]
        for item in closure["files"]:
            add(
                item["destination"],
                _source_for_shared(closure, runtime_root, item),
                item["executable"],
            )
    for closure in entry["launch"]["record_closures"]:
        for item, source in _expand_record_closure(closure):
            add(item["destination"], source, item["executable"])
    for item in entry["launch"]["embedded_files"]:
        add(
            item["destination"],
            base64.b64decode(item["content_base64"], validate=True),
            item["executable"],
        )
    for overlay in entry["launch"]["metadata_overlays"]:
        for item in overlay["replacement_files"]:
            add(
                item["destination"],
                base64.b64decode(item["content_base64"], validate=True),
                item["executable"],
            )
    for destination, (source, executable) in sources.items():
        target = root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(source, bytes):
            target.write_bytes(source)
        else:
            shutil.copyfile(source.resolve(strict=True), target)
        target.chmod(0o755 if executable else 0o644)
    # Bubblewrap processes mount operations after the immutable root is bound
    # read-only at ``/``.  Every mountpoint must therefore already exist in the
    # materialized image; empty directories are intentionally outside the
    # regular-file closure attested by Rust.
    for relative in (CURRENT_DIR, "staging", "proc", "dev", "run/aiperf"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    actual: dict[str, tuple[str, bool]] = {}
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ManifestGenerationError(f"materialized special/linked file: {path}")
        relative = path.relative_to(root).as_posix()
        actual[relative] = (_sha256(path.read_bytes()), bool(metadata.st_mode & 0o111))
    if _closure_sha256(actual) != entry["launch_closure_sha256"]:
        raise ManifestGenerationError("materialized launch-closure digest drift")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--materialize", type=Path)
    parser.add_argument("--distribution")
    args = parser.parse_args(argv)
    if args.materialize is not None:
        if not args.distribution:
            parser.error("--materialize requires --distribution")
        materialize(args.distribution, args.materialize)
    else:
        _write_or_check(args.write)


if __name__ == "__main__":
    main()
