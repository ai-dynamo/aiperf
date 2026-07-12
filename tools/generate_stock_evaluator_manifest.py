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
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from packaging.markers import default_environment
from packaging.requirements import Requirement

from aiperf.accuracy.evaluation.canonical import canonical_dumps, canonical_sha256
from aiperf.accuracy.evaluation.distributions import (
    ISOLATION_PROFILE,
    MAX_PROCESSES,
    NEMO_EVALUATOR_DISTRIBUTION,
    OPENBENCH_DISTRIBUTION,
    SOURCE_TREE_DIGEST_POLICY,
    STOCK_DISTRIBUTIONS,
    SourceComponentLock,
    SourceOverlayLock,
    StockDistributionDescriptor,
    distribution_identity_components,
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
NEMO_DEPLOYMENT_ROOT_ID = "nvidia_nemo_evaluator_0_4_locked"
OPENBENCH_DEPLOYMENT_ROOT_ID = "groq_openbench_0_5_3_inspect_0_3_141_locked"
SYSTEM_DEPLOYMENT_ROOT_ID = "system_linux_x86_64"
SITE_PACKAGES_DESTINATION = "runtime/lib/python3.12/site-packages"
SITE_PACKAGES_RELATIVE = "lib/python3.12/site-packages"
PROGRAM_DESTINATION = "runtime/bin/python3.12"
CURRENT_DIR = "work"
STOCK_MANIFEST_RESOURCE = (
    "aiperf.accuracy.evaluation.manifests/stock_distributions.json"
)
GSM8K_CANARY_SOURCE = (
    ROOT / "src/aiperf/accuracy/evaluation/manifests/assets/gsm8k_canary.jsonl"
)
GSM8K_CANARY_DESTINATION = "assets/gsm8k_canary.jsonl"
SOURCE_OVERLAY_DIR = ROOT / "src/aiperf/accuracy/evaluation/source_overlays"
RESOURCE_BOOTSTRAP_SOURCE = (
    ROOT / "src/aiperf/accuracy/evaluation/resource_bootstrap.py"
)
RESOURCE_BOOTSTRAP_DESTINATION = (
    "runtime/libexec/aiperf-evaluator-resource-bootstrap.py"
)
NEMO_ENVIRONMENT_LOCK = ROOT / "tools/stock_evaluators/nemo/uv.lock"
OPENBENCH_ENVIRONMENT_LOCK = ROOT / "tools/stock_evaluators/openbench/uv.lock"
AUDITED_DIRECT_DEPENDENCIES = {
    NEMO_EVALUATOR_DISTRIBUTION.distribution_id: ("orjson", "scipy"),
    OPENBENCH_DISTRIBUTION.distribution_id: ("orjson",),
}
GSM8K_SCORE_PROJECTION_ID = "gsm8k_binary_score_v1"
GSM8K_SCORE_SCHEMA_SHA256 = (
    "d156e6577305139bac7f48946996fa35d489a381a87bce4c58d18c47d8d9eeb5"
)
GSM8K_SCORE_SCHEMA_CANONICAL = (
    b'{"$schema":"https://json-schema.org/draft/2020-12/schema",'
    b'"additionalProperties":false,"properties":{"value":{"enum":[0,1],'
    b'"type":"number"}},"required":["value"],"type":"object"}'
)


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


_COMPONENT_BY_DISTRIBUTION = {
    _canonical_distribution_name(component.distribution): component
    for descriptor in STOCK_DISTRIBUTIONS.values()
    for component in descriptor.components
}
_ELF_INTERPRETER = re.compile(r"Requesting program interpreter:\s*([^\]]+)\]")
_ALLOWED_WORKER_SUFFIXES = {
    ".py",
    ".json",
    ".jsonl",
    ".patch",
    ".toml",
    ".yaml",
    ".yml",
}
_INSPECT_SOURCE_SUPPLEMENTS_BASE64 = {
    "tool/_tools/_computer/_resources/image_home_dir/.config/Code/User/settings.json": "ewogICJzZWN1cml0eS53b3Jrc3BhY2UudHJ1c3QuZW5hYmxlZCI6IGZhbHNlLAogICJ1cGRhdGUuc2hvd1JlbGVhc2VOb3RlcyI6IGZhbHNlLAogICJlZGl0b3IuY3Vyc29yQmxpbmtpbmciOiAic29saWQiLAogICJlZGl0b3IuY3Vyc29yV2lkdGgiOiAzLAogICJ3b3JrYmVuY2guY29sb3JDdXN0b21pemF0aW9ucyI6IHsKICAgICJlZGl0b3JDdXJzb3IuZm9yZWdyb3VuZCI6ICIjRkYwMDAwIgogIH0KfQo=",
    "tool/_tools/_computer/_resources/tool/.vscode/settings.json": "ewogICJjU3BlbGwud29yZHMiOiBbCiAgICAiRldYR0EiLAogICAgImdldG1vdXNlbG9jYXRpb24iLAogICAgImtleXVwIiwKICAgICJtb3VzZWRvd24iLAogICAgIm1vdXNlbW92ZSIsCiAgICAibW91c2V1cCIsCiAgICAic2Nyb3QiLAogICAgIldYR0EiCiAgXQp9Cg==",
    "tool/_tools/_computer/_resources/tool/pyproject.toml": "W2J1aWxkLXN5c3RlbV0KcmVxdWlyZXMgPSBbInNldHVwdG9vbHM+PTY0IiwgInNldHVwdG9vbHNfc2NtW3RvbWxdPj04Il0KYnVpbGQtYmFja2VuZCA9ICJzZXR1cHRvb2xzLmJ1aWxkX21ldGEiCgpbdG9vbC5zZXR1cHRvb2xzX3NjbV0KClt0b29sLnNldHVwdG9vbHMucGFja2FnZXMuZmluZF0Kd2hlcmUgPSBbIi4iXQppbmNsdWRlID0gWyJpbnNwZWN0X2FpKiJdCgpbdG9vbC5ydWZmXQpzcmMgPSBbIi4iXQoKW3Rvb2wucnVmZi5saW50XQpzZWxlY3QgPSBbCiAgICAiRSIsICAgICAgIyBweWNvZGVzdHlsZSBlcnJvcnMKICAgICJXIiwgICAgICAjIHB5Y29kZXN0eWxlIHdhcm5pbmdzCiAgICAiRiIsICAgICAgIyBmbGFrZTgKICAgICJEIiwgICAgICAjIHB5ZG9jc3R5bGUKICAgICJJIiwgICAgICAjIGlzb3J0CiAgICAiU0lNMTAxIiwgIyBkdXBsaWNhdGUgaXNpbnN0YW5jZQogICAgIyAiUkVUIiwgIyBmbGFrZTgtcmV0dXJuCiAgICAjICJSVUYiLCAjIHJ1ZmYgcnVsZXMKXQppZ25vcmUgPSBbIkUyMDMiLCAiRTUwMSIsICJEMTAiLCAiRDIxMiIsICJENDE1Il0KClt0b29sLnJ1ZmYubGludC5weWRvY3N0eWxlXQpjb252ZW50aW9uID0gImdvb2dsZSIKClt0b29sLnB5dGVzdC5pbmlfb3B0aW9uc10KbWludmVyc2lvbiA9ICI3LjAiCmFkZG9wdHMgPSAiLXJBIC0tZG9jdGVzdC1tb2R1bGVzIC0tY29sb3I9eWVzIgpkb2N0ZXN0X29wdGlvbmZsYWdzID0gWyJOT1JNQUxJWkVfV0hJVEVTUEFDRSIsICJJR05PUkVfRVhDRVBUSU9OX0RFVEFJTCJdCmFzeW5jaW9fbW9kZSA9ICJhdXRvIgphc3luY2lvX2RlZmF1bHRfZml4dHVyZV9sb29wX3Njb3BlID0gImZ1bmN0aW9uIgpsb2dfbGV2ZWwgPSAid2FybmluZyIKClt0b29sLm15cHldCndhcm5fdW51c2VkX2lnbm9yZXMgPSB0cnVlCm5vX2ltcGxpY2l0X3JlZXhwb3J0ID0gdHJ1ZQpzdHJpY3RfZXF1YWxpdHkgPSB0cnVlCndhcm5fcmVkdW5kYW50X2Nhc3RzID0gdHJ1ZQp3YXJuX3VudXNlZF9jb25maWdzID0gdHJ1ZQpkaXNhbGxvd19hbnlfZXhwbGljaXQgPSB0cnVlCmRpc2FsbG93X2FueV9nZW5lcmljcyA9IHRydWUKZGlzYWxsb3dfc3ViY2xhc3NpbmdfYW55ID0gdHJ1ZQpwbHVnaW5zPVsicHlkYW50aWMubXlweSJdCgoKW3Rvb2wucHlkYW50aWMtbXlweV0KaW5pdF9mb3JiaWRfZXh0cmEgPSB0cnVlCmluaXRfdHlwZWQgPSB0cnVlCgpbdG9vbC5jaGVjay13aGVlbC1jb250ZW50c10KaWdub3JlID0gWyJXMDAyIiwgIlcwMDkiXQoKW3Byb2plY3RdCm5hbWUgPSAid2ViX2Jyb3dzZXJfdG9vbF9jb250YWluZXIiCnJlcXVpcmVzLXB5dGhvbiA9ICI+PTMuMTAiCmR5bmFtaWMgPSBbInZlcnNpb24iLCAiZGVwZW5kZW5jaWVzIl0KCgpbcHJvamVjdC5vcHRpb25hbC1kZXBlbmRlbmNpZXNdCmRldiA9IFsicHl0ZXN0Il0=",
}


class ManifestGenerationError(RuntimeError):
    """The installed closure did not match the frozen generator contract."""


@dataclass(frozen=True)
class ProviderEnvironment:
    """One explicit complete virtual-environment input to manifest generation."""

    prefix: Path
    site_packages: Path
    runtime_root: Path
    distributions: Mapping[str, importlib.metadata.Distribution]
    resolution_lock: Path


@dataclass(frozen=True)
class DeploymentSourceFile:
    """One verified regular source file staged under a deployment root."""

    relative_path: str
    source: Path
    artifact_content_sha256: str
    executable: bool


@dataclass(frozen=True)
class DeploymentSourceRoot:
    """One immutable deployment-owned source root consumed by the runner."""

    id: str
    kind: Literal["python_runtime", "python_environment", "system"]
    relative_path: str
    files: tuple[DeploymentSourceFile, ...]


def _provider_environment(prefix: Path, resolution_lock: Path) -> ProviderEnvironment:
    prefix = prefix.expanduser().resolve(strict=True)
    if not prefix.is_absolute():
        raise ManifestGenerationError("provider environment prefix must be absolute")
    executable = prefix / "bin/python3.12"
    site_packages = prefix / "lib/python3.12/site-packages"
    if not executable.exists() or not site_packages.is_dir():
        raise ManifestGenerationError(
            f"provider environment is not a CPython 3.12 virtualenv: {prefix}"
        )
    runtime_executable = executable.resolve(strict=True)
    runtime_root = runtime_executable.parent.parent
    if runtime_executable != runtime_root / "bin/python3.12":
        raise ManifestGenerationError("provider environment runtime layout drifted")
    observed: dict[str, importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        name = _canonical_distribution_name(distribution.metadata["Name"])
        if name in observed:
            raise ManifestGenerationError(
                f"provider environment contains duplicate distribution {name!r}"
            )
        observed[name] = distribution
    if not observed:
        raise ManifestGenerationError("provider environment contains no distributions")
    resolution_lock = resolution_lock.resolve(strict=True)
    return ProviderEnvironment(
        prefix=prefix,
        site_packages=site_packages.resolve(strict=True),
        runtime_root=runtime_root.resolve(strict=True),
        distributions=observed,
        resolution_lock=resolution_lock,
    )


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


def _runtime_root(environments: Sequence[ProviderEnvironment]) -> Path:
    roots = {environment.runtime_root for environment in environments}
    if len(roots) != 1:
        raise ManifestGenerationError(
            "provider environments do not share the pinned CPython runtime"
        )
    root = next(iter(roots))
    executable = root / "bin/python3.12"
    result = subprocess.run(
        [
            str(executable),
            "-I",
            "-c",
            "import sys,sysconfig;print(sys.implementation.name);print('.'.join(map(str,sys.version_info[:3])));print(sysconfig.get_config_var('SOABI'))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.splitlines() != ["cpython", PYTHON_VERSION, PYTHON_ABI]:
        raise ManifestGenerationError("generator is not running on pinned CPython")
    if not executable.is_file() or executable.is_symlink():
        raise ManifestGenerationError("pinned CPython executable is not a regular file")
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


def _distribution(
    name: str, environment: ProviderEnvironment
) -> importlib.metadata.Distribution:
    distribution = environment.distributions.get(_canonical_distribution_name(name))
    if distribution is None:
        raise ManifestGenerationError(
            f"missing distribution {name!r} in {environment.prefix}"
        )
    return distribution


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


def _record_info(name: str, environment: ProviderEnvironment) -> dict[str, Any]:
    distribution = _distribution(name, environment)
    record = _record_path(distribution).read_bytes()
    return {
        "distribution": distribution.metadata["Name"],
        "version": distribution.version,
        "record_sha256": _sha256(record),
    }


def _source_omissions(
    component: SourceComponentLock,
    environment: ProviderEnvironment,
    existing_overlay_targets: Sequence[str] = (),
) -> tuple[str, ...]:
    distribution = _distribution(component.distribution, environment)
    dist_info = Path(distribution._path).name  # type: ignore[attr-defined]
    values = {
        f"{dist_info}/METADATA",
        f"{dist_info}/RECORD",
        f"{dist_info}/direct_url.json",
    }
    if component.distribution == "openbench":
        values.add(f"{dist_info}/entry_points.txt")
    values.update(
        f"{component.import_package}/{relative}"
        for relative in existing_overlay_targets
    )
    return tuple(sorted(values, key=str.encode))


def _record_closures(
    names: Sequence[str],
    environment: ProviderEnvironment,
    projections: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    result = []
    for name in names:
        info = _record_info(name, environment)
        component = _COMPONENT_BY_DISTRIBUTION.get(
            _canonical_distribution_name(info["distribution"])
        )
        projection = (
            projections.get(_canonical_distribution_name(component.distribution))
            if component is not None
            else None
        )
        result.append(
            {
                **info,
                "destination_prefix": SITE_PACKAGES_DESTINATION,
                "omitted_paths": (
                    list(
                        _source_omissions(
                            component,
                            environment,
                            projection["existing_overlay_targets"]
                            if projection
                            else (),
                        )
                    )
                    if component
                    else []
                ),
                "_environment": environment,
            }
        )
    return tuple(sorted(result, key=lambda item: item["distribution"].lower().encode()))


def _dependency_closure(
    environment: ProviderEnvironment, root_distributions: Sequence[str]
) -> tuple[str, ...]:
    """Resolve the exact base dependency graph from installed METADATA."""
    marker_environment = default_environment()
    marker_environment.update(
        {
            "python_version": "3.12",
            "python_full_version": PYTHON_VERSION,
            "sys_platform": "linux",
            "platform_machine": "x86_64",
        }
    )
    pending: list[tuple[str, frozenset[str]]] = [
        (distribution, frozenset()) for distribution in root_distributions
    ]
    selected: dict[str, frozenset[str]] = {}
    while pending:
        raw_name, extras = pending.pop()
        name = _canonical_distribution_name(raw_name)
        prior = selected.get(name, frozenset())
        merged = prior | extras
        if name in selected and merged == prior:
            continue
        selected[name] = merged
        distribution = _distribution(name, environment)
        for raw_requirement in distribution.requires or ():
            requirement = Requirement(raw_requirement)
            marker = requirement.marker
            applies = marker is None or marker.evaluate(
                {**marker_environment, "extra": ""}
            )
            if not applies:
                applies = any(
                    marker.evaluate({**marker_environment, "extra": extra})
                    for extra in merged
                )
            if applies:
                dependency = _distribution(requirement.name, environment)
                if (
                    requirement.specifier
                    and dependency.version not in requirement.specifier
                ):
                    raise ManifestGenerationError(
                        f"installed dependency {requirement.name!r} {dependency.version} violates {requirement.specifier}"
                    )
                pending.append((requirement.name, frozenset(requirement.extras)))
    installed = set(environment.distributions)
    if installed != set(selected):
        extra = sorted(installed - set(selected))
        missing = sorted(set(selected) - installed)
        raise ManifestGenerationError(
            f"provider environment is not the exact dependency closure; extra={extra}, missing={missing}"
        )
    return tuple(
        environment.distributions[name].metadata["Name"]
        for name in sorted(selected, key=str.encode)
    )


def _lock_value(
    descriptor: StockDistributionDescriptor,
    record_closures: Sequence[Mapping[str, Any]],
    environment: ProviderEnvironment,
) -> dict[str, Any]:
    record_identity = [
        {
            "distribution": item["distribution"],
            "version": item["version"],
            "record_sha256": item["record_sha256"],
        }
        for item in record_closures
    ]
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
                "base_source_tree_sha256": component.base_source_tree_sha256,
                "source_tree_digest_policy": SOURCE_TREE_DIGEST_POLICY,
                "overlays": [
                    {
                        "overlay_id": overlay.overlay_id,
                        "resource": overlay.resource,
                        "artifact_content_sha256": overlay.resource_sha256,
                        "targets": list(overlay.targets),
                    }
                    for overlay in component.overlays
                ],
                "effective_source_tree_sha256": component.source_tree_sha256,
            }
            for component in descriptor.components
        ],
        "resolution": {
            "kind": "uv-lock-v1",
            "artifact_content_sha256": _sha256(
                environment.resolution_lock.read_bytes()
            ),
            "environment_record_set_sha256": canonical_sha256(record_identity),
        },
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
                "base_source_tree_sha256": component.base_source_tree_sha256,
                "overlays": [
                    {
                        "overlay_id": overlay.overlay_id,
                        "artifact_content_sha256": overlay.resource_sha256,
                    }
                    for overlay in component.overlays
                ],
                "effective_source_tree_sha256": component.source_tree_sha256,
            }
            for component in descriptor.components
        ]
    )


def _verify_component_sources(
    descriptor: StockDistributionDescriptor, environment: ProviderEnvironment
) -> dict[str, dict[str, Any]]:
    projections: dict[str, dict[str, Any]] = {}
    for component in descriptor.components:
        distribution = _distribution(component.distribution, environment)
        root = Path(distribution.locate_file(component.import_package))
        if not root.is_dir() or root.is_symlink():
            raise ManifestGenerationError(
                f"source component {component.import_package!r} is unavailable"
            )
        supplements = (
            {
                relative: base64.b64decode(content, validate=True)
                for relative, content in _INSPECT_SOURCE_SUPPLEMENTS_BASE64.items()
            }
            if component.distribution == "inspect-ai"
            else {}
        )
        expected_restored = {
            item.relative_path: item.content_sha256
            for item in component.restored_base_files
        }
        actual_restored = {
            relative: _sha256(content) for relative, content in supplements.items()
        }
        if actual_restored != expected_restored:
            raise ManifestGenerationError(
                f"restored base-file inventory drift for {component.distribution!r}"
            )
        actual_base = _projected_source_tree_sha256(
            root, component.source_tree_excluded_paths, supplements
        )
        if actual_base != component.base_source_tree_sha256:
            raise ManifestGenerationError(
                f"base source tree drift for {component.distribution!r}: {actual_base}"
            )
        with tempfile.TemporaryDirectory(
            prefix=f"aiperf-{component.import_package}-overlay-"
        ) as temporary:
            effective_root = Path(temporary) / component.import_package
            shutil.copytree(root, effective_root, symlinks=False)
            for relative, content in supplements.items():
                target = effective_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ManifestGenerationError(
                        f"restored base file collided with wheel content: {relative}"
                    )
                target.write_bytes(content)
            entries: list[dict[str, Any]] = []
            existing_targets: set[str] = set()
            for ordinal, overlay in enumerate(component.overlays):
                entry = _apply_source_overlay(
                    effective_root,
                    overlay,
                    component.source_tree_excluded_paths,
                    ordinal,
                )
                entries.append(entry)
                existing_targets.update(
                    target["relative_path"]
                    for target in entry["targets"]
                    if target["base_content_sha256"] is not None
                )
            effective = _projected_source_tree_sha256(
                effective_root, component.source_tree_excluded_paths, {}
            )
            if effective != component.source_tree_sha256:
                raise ManifestGenerationError(
                    f"effective source tree drift for {component.distribution!r}: {effective}"
                )
            replacement_contents = {
                relative: (effective_root / relative).read_bytes()
                for overlay in component.overlays
                for relative in overlay.targets
            }
        projections[_canonical_distribution_name(component.distribution)] = {
            "base_source_tree_sha256": actual_base,
            "restored_base_files": supplements,
            "overlays": entries,
            "effective_source_tree_sha256": effective,
            "existing_overlay_targets": tuple(sorted(existing_targets, key=str.encode)),
            "replacement_contents": replacement_contents,
        }
    return projections


def _apply_source_overlay(
    root: Path,
    overlay: SourceOverlayLock,
    excluded: Sequence[str],
    ordinal: int,
) -> dict[str, Any]:
    patch = SOURCE_OVERLAY_DIR / overlay.resource
    content = patch.read_bytes()
    if _sha256(content) != overlay.resource_sha256:
        raise ManifestGenerationError(
            f"source overlay digest drift for {overlay.overlay_id!r}"
        )
    before = _semantic_tree(root, excluded, {})
    result = subprocess.run(
        ["git", "apply", "--recount", "-p1", str(patch)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ManifestGenerationError(
            f"source overlay {overlay.overlay_id!r} did not apply: {result.stderr.strip()}"
        )
    after = _semantic_tree(root, excluded, {})
    changed = tuple(
        sorted(
            {
                relative
                for relative in set(before) | set(after)
                if before.get(relative) != after.get(relative)
            },
            key=str.encode,
        )
    )
    if changed != tuple(sorted(overlay.targets, key=str.encode)):
        raise ManifestGenerationError(
            f"source overlay {overlay.overlay_id!r} target set drifted: {changed}"
        )
    if any(relative not in after for relative in changed):
        raise ManifestGenerationError("source overlays may not delete semantic files")
    return {
        "ordinal": ordinal,
        "overlay_id": overlay.overlay_id,
        "resource": overlay.resource,
        "artifact_content_sha256": overlay.resource_sha256,
        "targets": [
            {
                "relative_path": relative,
                "base_content_sha256": (
                    _sha256(before[relative]) if relative in before else None
                ),
                "effective_content_sha256": _sha256(after[relative]),
            }
            for relative in overlay.targets
        ],
    }


def _projected_source_tree_sha256(
    root: Path,
    excluded: Sequence[str],
    supplements: Mapping[str, bytes],
) -> str:
    semantic = _semantic_tree(root, excluded, supplements)
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


def _semantic_tree(
    root: Path, excluded: Sequence[str], supplements: Mapping[str, bytes]
) -> dict[str, bytes]:
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
    return semantic


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
        if path.suffix not in {
            ".py",
            ".json",
            ".toml",
            ".yaml",
            ".yml",
        }:
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


def _metadata_overlay(
    component: SourceComponentLock,
    environment: ProviderEnvironment,
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    distribution = _distribution(component.distribution, environment)
    record_path = _record_path(distribution)
    upstream_record = record_path.read_bytes()
    rows = _record_rows(upstream_record)
    omitted = set(
        _source_omissions(
            component,
            environment,
            projection["existing_overlay_targets"],
        )
    )
    dist_info = Path(distribution._path).name  # type: ignore[attr-defined]
    metadata_relative = f"{dist_info}/METADATA"
    record_relative = f"{dist_info}/RECORD"
    attestation_relative = f"{dist_info}/aiperf_source_attestation.json"
    metadata = (Path(distribution._path) / "METADATA").read_bytes()  # type: ignore[attr-defined]
    attestation_value = {
        "format": "aiperf-pinned-source-overlay-v1",
        "distribution": component.distribution,
        "version": component.version,
        "base": {
            "commit": component.commit,
            "digest_policy": SOURCE_TREE_DIGEST_POLICY,
            "source_tree_sha256": component.base_source_tree_sha256,
            "excluded_relative_paths": list(component.source_tree_excluded_paths),
            "restored_files": [
                {
                    "relative_path": item.relative_path,
                    "artifact_content_sha256": item.content_sha256,
                }
                for item in component.restored_base_files
            ],
        },
        "overlay_policy": "aiperf-unified-diff-overlay-v1",
        "overlays": projection["overlays"],
        "effective_source_tree_sha256": component.source_tree_sha256,
        "effective_source_tree_digest_policy": SOURCE_TREE_DIGEST_POLICY,
    }
    attestation = canonical_dumps(attestation_value)
    source_replacements: dict[str, bytes] = {
        f"{component.import_package}/{relative}": content
        for relative, content in projection["replacement_contents"].items()
    }
    source_replacements.update(
        {
            f"{component.import_package}/{relative}": content
            for relative, content in projection["restored_base_files"].items()
        }
    )
    final_rows = [row for row in rows if row[0] not in omitted]
    replacements = {
        **source_replacements,
        metadata_relative: metadata,
        attestation_relative: attestation,
    }
    for relative, content in replacements.items():
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
    replacements[record_relative] = replacement_record
    return {
        "policy_id": "pinned_source_overlay_v1",
        "distribution": distribution.metadata["Name"],
        "version": component.version,
        "upstream_record_sha256": _sha256(upstream_record),
        "omitted_paths": sorted(omitted, key=str.encode),
        "base_source": attestation_value["base"],
        "overlay_policy": attestation_value["overlay_policy"],
        "overlays": projection["overlays"],
        "effective_source_tree_sha256": component.source_tree_sha256,
        "effective_source_tree_digest_policy": SOURCE_TREE_DIGEST_POLICY,
        "replacement_files": [
            _embedded_file(
                f"generated://source-overlay/{component.distribution}/{relative}",
                f"{SITE_PACKAGES_DESTINATION}/{relative}",
                content,
            )
            for relative, content in sorted(
                replacements.items(), key=lambda item: item[0].encode()
            )
        ],
    }


def _expand_record_closure(
    closure: Mapping[str, Any],
    environment: ProviderEnvironment,
) -> tuple[tuple[dict[str, Any], Path], ...]:
    distribution = _distribution(str(closure["distribution"]), environment)
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
    environment: ProviderEnvironment,
) -> tuple[Path, ...]:
    result: set[Path] = set()
    for item in runtime_files:
        source = runtime_root / str(item["source_relative_path"])
        if source.read_bytes()[:4] == b"\x7fELF":
            result.add(source)
    for closure in record_closures:
        for _, source in _expand_record_closure(closure, environment):
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
    environment: ProviderEnvironment,
) -> tuple[dict[str, Any], ...]:
    package_roots = {environment.site_packages}
    queue = list(_elf_inputs(runtime_root, runtime_files, record_closures, environment))
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
            if resolved.is_relative_to(runtime_root) or any(
                resolved.is_relative_to(package_root) for package_root in package_roots
            ):
                continue
            if not _is_allowed_system_source(resolved):
                raise ManifestGenerationError(
                    f"ELF dependency escaped pinned runtime/package roots and system library prefixes: {resolved}"
                )
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


def _is_allowed_system_source(path: Path) -> bool:
    """Accept only deployment-owned dynamic-library roots, never host worktrees."""
    parts = path.parts
    return (
        len(parts) >= 3
        and parts[0] == "/"
        and (
            parts[1] in {"lib", "lib64"}
            or (len(parts) >= 4 and parts[1] == "usr" and parts[2] in {"lib", "lib64"})
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
    environment: ProviderEnvironment,
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
        for item, _ in _expand_record_closure(closure, environment):
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


def _validate_public_projection(public_projection: Mapping[str, Any]) -> None:
    """Bind every stock public score to the reviewed executable validator."""
    score_schemas = public_projection.get("score_schemas")
    if not isinstance(score_schemas, list) or len(score_schemas) != 1:
        raise ManifestGenerationError(
            "stock public projection must contain exactly one score schema"
        )
    score = score_schemas[0]
    if not isinstance(score, dict):
        raise ManifestGenerationError("stock public score schema must be an object")
    if score.get("projection_id") != GSM8K_SCORE_PROJECTION_ID:
        raise ManifestGenerationError(
            "stock public score schema named an unreviewed executable validator"
        )
    schema = score.get("schema")
    if canonical_dumps(schema) != GSM8K_SCORE_SCHEMA_CANONICAL:
        raise ManifestGenerationError(
            "stock GSM8K public score schema drifted from the reviewed object"
        )
    actual_sha256 = canonical_sha256(schema)
    if actual_sha256 != GSM8K_SCORE_SCHEMA_SHA256:
        raise ManifestGenerationError(
            f"stock GSM8K public score schema digest drifted: {actual_sha256}"
        )
    if score.get("schema_sha256") != actual_sha256:
        raise ManifestGenerationError(
            "stock GSM8K public score schema advertised the wrong digest"
        )


def _distribution_entry(
    descriptor: StockDistributionDescriptor,
    lock_bytes: bytes,
    record_closures: Sequence[Mapping[str, Any]],
    shared_closure_ids: Sequence[str],
    shared_closures: Mapping[str, Mapping[str, Any]],
    worker_resources: Sequence[tuple[Path, str, bytes]],
    environment: ProviderEnvironment,
    projections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    embedded_files = [
        _embedded_file(path.relative_to(ROOT).as_posix(), destination, content)
        for path, destination, content in worker_resources
    ]
    embedded_files.append(
        _embedded_file(
            RESOURCE_BOOTSTRAP_SOURCE.relative_to(ROOT).as_posix(),
            RESOURCE_BOOTSTRAP_DESTINATION,
            RESOURCE_BOOTSTRAP_SOURCE.read_bytes(),
        )
    )
    embedded_files.append(
        _embedded_file(
            GSM8K_CANARY_SOURCE.relative_to(ROOT).as_posix(),
            GSM8K_CANARY_DESTINATION,
            GSM8K_CANARY_SOURCE.read_bytes(),
        )
    )
    overlays = [
        _metadata_overlay(
            component,
            environment,
            projections[_canonical_distribution_name(component.distribution)],
        )
        for component in descriptor.components
    ]
    inventory = _expanded_inventory(
        shared_closure_ids,
        shared_closures,
        record_closures,
        embedded_files,
        overlays,
        environment,
    )
    manifest_resource = descriptor.task_manifest_resource
    manifest_path = MANIFEST_DIR / manifest_resource
    manifest_bytes = manifest_path.read_bytes()
    manifest_value = task_manifest(descriptor)
    manifest_entries = (
        manifest_value.get("environments", {})
        if descriptor.provider_id == "nemo_evaluator"
        else manifest_value.get("tasks", {})
    )
    executable_entries = {
        name: value
        for name, value in manifest_entries.items()
        if isinstance(value, dict) and value.get("executable") is True
    }
    if len(executable_entries) != 1:
        raise ManifestGenerationError(
            "stock distribution must expose exactly one audited executable task"
        )
    public_projection = next(iter(executable_entries.values())).get("public_projection")
    if not isinstance(public_projection, dict):
        raise ManifestGenerationError("executable task omitted public projection")
    _validate_public_projection(public_projection)
    operation = OPERATION_DIRECTION_SCHEMA_SHA256["model.generate"]
    worker_source_sha256 = _worker_source_digest(worker_resources)
    dependency_lock_sha256 = _sha256(lock_bytes)
    return {
        "provider_id": descriptor.provider_id,
        "distribution_id": descriptor.distribution_id,
        "package": descriptor.package,
        "package_version": descriptor.package_version,
        "provider_source_sha256": _provider_source_digest(descriptor),
        "identity_components": [
            component.to_wire()
            for component in distribution_identity_components(
                descriptor,
                worker_source_sha256=worker_source_sha256,
                dependency_lock_sha256=dependency_lock_sha256,
            )
        ],
        "worker_source_sha256": worker_source_sha256,
        "dependency_lock_sha256": dependency_lock_sha256,
        "oci_digest": None,
        "config_schema_version": descriptor.config_schema_version,
        "config_schema_sha256": descriptor.config_schema_sha256,
        "task_manifest": {
            "resource": f"aiperf.accuracy.evaluation.manifests/{manifest_resource}",
            "artifact_content_sha256": _sha256(manifest_bytes),
            "executable_tasks": list(executable_entries),
        },
        "public_projection": public_projection,
        "operations": [
            {
                "operation_id": "model.generate",
                "request_schema_sha256": operation["request"],
                "response_schema_sha256": operation["response"],
                "stream_schema_sha256": operation["stream"],
                "endpoint_capability": "chat",
            }
        ],
        "compatibility_proxy": (
            {
                "dialects": list(descriptor.compatibility_proxy.dialects),
                "routes": [
                    {
                        "dialect": route.dialect,
                        "selector": route.selector,
                        "service_id": route.service_id,
                        "purpose": route.purpose,
                        "semantic_operation_id": route.semantic_operation_id,
                        "restricted_payload": route.restricted_payload,
                    }
                    for route in descriptor.compatibility_proxy.routes
                ],
                "grant_limits": {
                    "max_operations": descriptor.compatibility_proxy.max_operations,
                    "max_concurrent_operations": descriptor.compatibility_proxy.max_concurrent_operations,
                    "max_request_bytes": descriptor.compatibility_proxy.max_request_bytes,
                    "max_response_bytes": descriptor.compatibility_proxy.max_response_bytes,
                    "max_stream_events": descriptor.compatibility_proxy.max_stream_events,
                    "expires_after_ms": descriptor.compatibility_proxy.expires_after_ms,
                },
            }
            if descriptor.compatibility_proxy is not None
            else None
        ),
        "isolation": {
            "profile_id": ISOLATION_PROFILE,
            "bubblewrap": "/usr/bin/bwrap",
            "bubblewrap_sha256": _sha256(Path("/usr/bin/bwrap").read_bytes()),
            "resource_limits": {
                "address_space_bytes": 16 * 1024 * 1024 * 1024,
                "file_size_bytes": 8 * 1024 * 1024 * 1024,
                "open_files": 4096,
                "processes": MAX_PROCESSES,
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
            "record_closures": [
                {
                    key: value
                    for key, value in closure.items()
                    if not key.startswith("_")
                }
                for closure in record_closures
            ],
            "embedded_files": embedded_files,
            "metadata_overlays": overlays,
        },
        "launch_closure_sha256": _closure_sha256(inventory),
    }


def generate(
    *, nemo_root: Path, openbench_root: Path
) -> tuple[dict[Path, bytes], bytes]:
    """Generate lock resources and the complete stock launch manifest."""
    environments = {
        NEMO_EVALUATOR_DISTRIBUTION.distribution_id: _provider_environment(
            nemo_root, NEMO_ENVIRONMENT_LOCK
        ),
        OPENBENCH_DISTRIBUTION.distribution_id: _provider_environment(
            openbench_root, OPENBENCH_ENVIRONMENT_LOCK
        ),
    }
    runtime_root = _runtime_root(tuple(environments.values()))
    runtime_files = _runtime_files(runtime_root)
    descriptors = (NEMO_EVALUATOR_DISTRIBUTION, OPENBENCH_DISTRIBUTION)
    record_sets: dict[str, tuple[dict[str, Any], ...]] = {}
    projections: dict[str, dict[str, dict[str, Any]]] = {}
    lock_bytes: dict[Path, bytes] = {}
    for descriptor in descriptors:
        environment = environments[descriptor.distribution_id]
        projection = _verify_component_sources(descriptor, environment)
        projections[descriptor.distribution_id] = projection
        names = _dependency_closure(
            environment,
            (
                descriptor.package,
                *AUDITED_DIRECT_DEPENDENCIES[descriptor.distribution_id],
            ),
        )
        closures = _record_closures(names, environment, projection)
        record_sets[descriptor.distribution_id] = closures
        path = MANIFEST_DIR / descriptor.dependency_lock_resource
        lock_bytes[path] = _json_bytes(_lock_value(descriptor, closures, environment))
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
    for descriptor in descriptors:
        closure_id = f"system_{PLATFORM.replace('-', '_')}_{descriptor.provider_id}"
        system_ids[descriptor.distribution_id] = closure_id
        shared_closures[closure_id] = {
            "resolver": {"kind": "system_root", "platform": PLATFORM},
            "files": list(
                _system_files(
                    runtime_root,
                    runtime_files,
                    record_sets[descriptor.distribution_id],
                    environments[descriptor.distribution_id],
                )
            ),
        }
    entries = []
    for descriptor in descriptors:
        entries.append(
            _distribution_entry(
                descriptor,
                lock_bytes[MANIFEST_DIR / descriptor.dependency_lock_resource],
                record_sets[descriptor.distribution_id],
                (RUNTIME_CLOSURE_ID, system_ids[descriptor.distribution_id]),
                shared_closures,
                worker_resources,
                environments[descriptor.distribution_id],
                projections[descriptor.distribution_id],
            )
        )
    value = {
        "schema_version": "aiperf-stock-evaluator-distributions-v1",
        "shared_closures": shared_closures,
        "distributions": entries,
    }
    return lock_bytes, _json_bytes(value)


def _verified_deployment_source_file(
    *,
    relative_path: str,
    source: Path,
    artifact_content_sha256: str,
    executable: bool,
) -> DeploymentSourceFile:
    relative_path = _normalize_relative(relative_path)
    try:
        metadata = source.lstat()
    except OSError as error:
        raise ManifestGenerationError(
            f"deployment source is unavailable: {source}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode) or source.is_symlink():
        raise ManifestGenerationError(
            f"deployment source is not a regular non-symlink file: {source}"
        )
    source = source.resolve(strict=True)
    content = source.read_bytes()
    if _sha256(content) != artifact_content_sha256:
        raise ManifestGenerationError(f"deployment source digest drifted: {source}")
    if _is_executable(source) != executable:
        raise ManifestGenerationError(
            f"deployment source executable mode drifted: {source}"
        )
    return DeploymentSourceFile(
        relative_path=relative_path,
        source=source,
        artifact_content_sha256=artifact_content_sha256,
        executable=executable,
    )


def _upstream_record_source_files(
    record_closures: Sequence[Mapping[str, Any]],
    environment: ProviderEnvironment,
) -> tuple[DeploymentSourceFile, ...]:
    """Return every original RECORD row, including materialization omissions."""
    files: dict[str, DeploymentSourceFile] = {}
    for closure in record_closures:
        distribution = _distribution(str(closure["distribution"]), environment)
        if distribution.version != closure["version"]:
            raise ManifestGenerationError(
                f"version drift for {closure['distribution']!r}"
            )
        record_path = _record_path(distribution)
        record = record_path.read_bytes()
        if _sha256(record) != closure["record_sha256"]:
            raise ManifestGenerationError(
                f"RECORD drift for {closure['distribution']!r}"
            )
        rows = _record_rows(record)
        row_paths = {row[0] for row in rows}
        if len(row_paths) != len(rows):
            raise ManifestGenerationError(
                f"RECORD contains duplicate rows for {closure['distribution']!r}"
            )
        if not set(closure["omitted_paths"]).issubset(row_paths):
            raise ManifestGenerationError(
                f"RECORD omissions drifted for {closure['distribution']!r}"
            )
        for relative, encoded, size in rows:
            source = Path(distribution.locate_file(PurePosixPath(relative)))
            try:
                resolved = source.resolve(strict=True)
            except OSError as error:
                raise ManifestGenerationError(
                    f"RECORD file is absent: {source}"
                ) from error
            if not resolved.is_relative_to(environment.prefix):
                raise ManifestGenerationError(
                    f"RECORD source escaped its Python environment: {source}"
                )
            if encoded:
                expected = _record_digest(encoded)
            else:
                if resolved != record_path.resolve(strict=True):
                    raise ManifestGenerationError(
                        "only a distribution RECORD may omit its own digest"
                    )
                expected = _sha256(record)
            if size and resolved.stat().st_size != int(size):
                raise ManifestGenerationError(f"RECORD size drift: {source}")
            relative_path = _destination(SITE_PACKAGES_RELATIVE, relative)
            item = _verified_deployment_source_file(
                relative_path=relative_path,
                source=source,
                artifact_content_sha256=expected,
                executable=_is_executable(source),
            )
            if relative_path in files:
                raise ManifestGenerationError(
                    f"duplicate provider-environment source path: {relative_path}"
                )
            files[relative_path] = item
    return tuple(files[path] for path in sorted(files, key=str.encode))


def _manifest_shared_source_files(
    closure: Mapping[str, Any],
    *,
    runtime_root: Path,
    root_prefix: str,
) -> tuple[DeploymentSourceFile, ...]:
    files = []
    for item in closure["files"]:
        relative = _normalize_relative(str(item["source_relative_path"]))
        expected_destination = (
            _destination(root_prefix, relative) if root_prefix else relative
        )
        if item["destination"] != expected_destination:
            raise ManifestGenerationError(
                f"shared closure destination drifted for {relative!r}"
            )
        files.append(
            _verified_deployment_source_file(
                relative_path=relative,
                source=_source_for_shared(closure, runtime_root, item),
                artifact_content_sha256=str(item["artifact_content_sha256"]),
                executable=bool(item["executable"]),
            )
        )
    if len({item.relative_path for item in files}) != len(files):
        raise ManifestGenerationError("shared closure contains duplicate source paths")
    return tuple(sorted(files, key=lambda item: item.relative_path.encode()))


def deployment_source_roots(
    *, nemo_root: Path, openbench_root: Path
) -> tuple[DeploymentSourceRoot, ...]:
    """Return the four exact source roots used to build a runner deployment.

    Generation re-verifies the complete dependency closure, upstream RECORDs,
    source overlays, and effective task/worker manifests. The committed lock
    and aggregate manifest bytes must then match that fresh result before any
    source path is released to deployment tooling.
    """
    lock_bytes, manifest_bytes = generate(
        nemo_root=nemo_root, openbench_root=openbench_root
    )
    outputs = {**lock_bytes, OUTPUT: manifest_bytes}
    mismatches = [
        path.relative_to(ROOT).as_posix()
        for path, content in outputs.items()
        if not path.is_file() or path.read_bytes() != content
    ]
    if mismatches:
        raise ManifestGenerationError(
            "generated evaluator resources drifted: " + ", ".join(mismatches)
        )

    environments = {
        NEMO_DEPLOYMENT_ROOT_ID: _provider_environment(
            nemo_root, NEMO_ENVIRONMENT_LOCK
        ),
        OPENBENCH_DEPLOYMENT_ROOT_ID: _provider_environment(
            openbench_root, OPENBENCH_ENVIRONMENT_LOCK
        ),
    }
    runtime_root = _runtime_root(tuple(environments.values()))
    manifest = json.loads(manifest_bytes)
    shared = manifest["shared_closures"]
    runtime_closure = shared.get(RUNTIME_CLOSURE_ID)
    if (
        not isinstance(runtime_closure, dict)
        or runtime_closure.get("resolver", {}).get("kind") != "python_runtime_root"
    ):
        raise ManifestGenerationError("pinned deployment runtime closure is absent")
    runtime_files = _manifest_shared_source_files(
        runtime_closure,
        runtime_root=runtime_root,
        root_prefix="runtime",
    )

    entries = {entry["distribution_id"]: entry for entry in manifest["distributions"]}
    if set(entries) != set(environments):
        raise ManifestGenerationError("deployment provider distribution set drifted")
    provider_files = {
        distribution_id: _upstream_record_source_files(
            entries[distribution_id]["launch"]["record_closures"], environment
        )
        for distribution_id, environment in environments.items()
    }

    system_files: dict[str, DeploymentSourceFile] = {}
    system_closure_count = 0
    for closure in shared.values():
        if closure.get("resolver", {}).get("kind") != "system_root":
            continue
        system_closure_count += 1
        for item in _manifest_shared_source_files(
            closure,
            runtime_root=runtime_root,
            root_prefix="",
        ):
            prior = system_files.get(item.relative_path)
            if prior is not None:
                if (
                    prior.artifact_content_sha256 != item.artifact_content_sha256
                    or prior.executable != item.executable
                ):
                    raise ManifestGenerationError(
                        f"merged system destination collision: {item.relative_path}"
                    )
                continue
            system_files[item.relative_path] = item
    if system_closure_count != len(environments) or not system_files:
        raise ManifestGenerationError("deployment system closure set drifted")

    roots = (
        DeploymentSourceRoot(
            id=RUNTIME_CLOSURE_ID,
            kind="python_runtime",
            relative_path="runtime",
            files=runtime_files,
        ),
        DeploymentSourceRoot(
            id=NEMO_DEPLOYMENT_ROOT_ID,
            kind="python_environment",
            relative_path="nemo",
            files=provider_files[NEMO_DEPLOYMENT_ROOT_ID],
        ),
        DeploymentSourceRoot(
            id=OPENBENCH_DEPLOYMENT_ROOT_ID,
            kind="python_environment",
            relative_path="openbench",
            files=provider_files[OPENBENCH_DEPLOYMENT_ROOT_ID],
        ),
        DeploymentSourceRoot(
            id=SYSTEM_DEPLOYMENT_ROOT_ID,
            kind="system",
            relative_path="system",
            files=tuple(
                system_files[path] for path in sorted(system_files, key=str.encode)
            ),
        ),
    )
    if any(not root.files for root in roots):
        raise ManifestGenerationError("deployment source root was empty")
    return roots


def _write_or_check(write: bool, *, nemo_root: Path, openbench_root: Path) -> None:
    lock_bytes, manifest = generate(nemo_root=nemo_root, openbench_root=openbench_root)
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


def materialize(
    distribution_id: str,
    root: Path,
    *,
    nemo_root: Path,
    openbench_root: Path,
) -> None:
    """Materialize one generated closure for a no-host-path subprocess proof."""
    environments = {
        NEMO_EVALUATOR_DISTRIBUTION.distribution_id: _provider_environment(
            nemo_root, NEMO_ENVIRONMENT_LOCK
        ),
        OPENBENCH_DISTRIBUTION.distribution_id: _provider_environment(
            openbench_root, OPENBENCH_ENVIRONMENT_LOCK
        ),
    }
    _, manifest_bytes = generate(nemo_root=nemo_root, openbench_root=openbench_root)
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
    runtime_root = _runtime_root(tuple(environments.values()))
    environment = environments[distribution_id]
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
        for item, source in _expand_record_closure(closure, environment):
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
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--openbench-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.materialize is not None:
        if not args.distribution:
            parser.error("--materialize requires --distribution")
        materialize(
            args.distribution,
            args.materialize,
            nemo_root=args.nemo_root,
            openbench_root=args.openbench_root,
        )
    else:
        _write_or_check(
            args.write,
            nemo_root=args.nemo_root,
            openbench_root=args.openbench_root,
        )


if __name__ == "__main__":
    main()
