# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable stock evaluator distribution descriptors and source attestation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

import orjson

from aiperf.accuracy.evaluation.canonical import canonical_dumps, canonical_sha256
from aiperf.accuracy.evaluation.contracts import EvaluationWorkerIdentity

WORKER_MODULE = "aiperf.accuracy.evaluation.worker"
CONTROL_READ_FD = 3
CONTROL_WRITE_FD = 4
STAGING_ROOT = "/staging"
ISOLATION_PROFILE = "linux-bubblewrap-rootfs-process-tree-v2"
WORKER_OPERATIONS = (
    "plan_session",
    "bind_assets",
    "next_units",
    "instantiate_units",
    "start_units",
    "poll_events",
    "submit_host_events",
    "cancel_units",
    "finalize_session",
    "shutdown",
)


@dataclass(frozen=True)
class SourceComponentLock:
    """Exact installed Python package/source identity."""

    distribution: str
    import_package: str
    version: str
    commit: str
    source_tree_sha256: str
    source_tree_excluded_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class DistributionEvidence:
    """Worker-observed evidence independently checked by the Rust factory."""

    provider_source_sha256: str
    worker_source_sha256: str
    dependency_lock_sha256: str


@dataclass(frozen=True)
class StockDistributionDescriptor:
    """Factory-owned launch identity for one immutable provider closure."""

    provider_id: str
    distribution_id: str
    package: str
    package_version: str
    components: tuple[SourceComponentLock, ...]
    dependency_lock_resource: str
    config_schema_version: int
    config_schema: dict[str, Any]
    task_manifest_resource: str

    @property
    def config_schema_sha256(self) -> str:
        """Return the exact authored-config schema fingerprint."""
        return canonical_sha256(self.config_schema)

    @property
    def fixed_argv(self) -> tuple[str, ...]:
        """Return literal worker arguments owned by the stock factory."""
        return (
            "-I",
            "-m",
            WORKER_MODULE,
            "--provider",
            self.provider_id,
            "--distribution",
            self.distribution_id,
            "--read-fd",
            str(CONTROL_READ_FD),
            "--write-fd",
            str(CONTROL_WRITE_FD),
            "--staging-root",
            STAGING_ROOT,
        )

    @property
    def clean_environment(self) -> MappingProxyType[str, str]:
        """Return the complete non-secret production worker environment."""
        return MappingProxyType(
            {
                "PATH": "/runtime/bin",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "XDG_DATA_HOME": f"{STAGING_ROOT}/.xdg-data",
                "XDG_CACHE_HOME": f"{STAGING_ROOT}/.xdg-cache",
            }
        )

    def verify_installed_closure(self) -> DistributionEvidence:
        """Fail closed unless every package/version/commit/source byte is exact."""
        component_evidence: list[dict[str, str]] = []
        for component in self.components:
            try:
                installed_version = importlib.metadata.version(component.distribution)
            except importlib.metadata.PackageNotFoundError as error:
                raise DistributionVerificationError(
                    f"required evaluator package {component.distribution!r} is absent"
                ) from error
            if installed_version != component.version:
                raise DistributionVerificationError(
                    f"evaluator package {component.distribution!r} version drift"
                )
            spec = importlib.util.find_spec(component.import_package)
            if spec is None or not spec.submodule_search_locations:
                raise DistributionVerificationError(
                    f"evaluator package {component.import_package!r} has no package tree"
                )
            locations = tuple(spec.submodule_search_locations)
            if len(locations) != 1:
                raise DistributionVerificationError(
                    f"evaluator package {component.import_package!r} is namespace-ambiguous"
                )
            actual_tree = source_tree_sha256(
                Path(locations[0]),
                excluded_relative_paths=frozenset(component.source_tree_excluded_paths),
            )
            if actual_tree != component.source_tree_sha256:
                raise DistributionVerificationError(
                    f"evaluator package {component.distribution!r} source digest drift"
                )
            _verify_source_attestation(component)
            component_evidence.append(
                {
                    "distribution": component.distribution,
                    "version": component.version,
                    "commit": component.commit,
                    "source_tree_sha256": actual_tree,
                }
            )
        lock_bytes = _resource_bytes(self.dependency_lock_resource)
        task_manifest = _resource_bytes(self.task_manifest_resource)
        # Loading now proves packaged manifests are valid strict JSON; their
        # bytes, not a mutable filesystem path, are the attested domain.
        orjson.loads(lock_bytes)
        orjson.loads(task_manifest)
        if self.provider_id == "openbench":
            inspect_entry_points = importlib.metadata.entry_points(group="inspect_ai")
            if tuple(inspect_entry_points):
                raise DistributionVerificationError(
                    "OpenBench AIPerf distribution contains forbidden Inspect entry points"
                )
        worker_root = Path(__file__).resolve().parent
        return DistributionEvidence(
            provider_source_sha256=canonical_sha256(component_evidence),
            worker_source_sha256=source_tree_sha256(
                worker_root,
                excluded_relative_paths=frozenset(
                    {"manifests/stock_distributions.json"}
                ),
            ),
            dependency_lock_sha256=hashlib.sha256(lock_bytes).hexdigest(),
        )

    def worker_identity(
        self, launch_nonce: str, evidence: DistributionEvidence
    ) -> EvaluationWorkerIdentity:
        """Construct the exact ``hello`` identity for a verified launch."""
        return EvaluationWorkerIdentity(
            evaluator_protocol=2,
            provider_id=self.provider_id,
            distribution_id=self.distribution_id,
            package=self.package,
            package_version=self.package_version,
            provider_source_sha256=evidence.provider_source_sha256,
            worker_source_sha256=evidence.worker_source_sha256,
            dependency_lock_sha256=evidence.dependency_lock_sha256,
            python_version=".".join(map(str, sys.version_info[:3])),
            launch_nonce=launch_nonce,
            operations=WORKER_OPERATIONS,
        )


class DistributionVerificationError(RuntimeError):
    """Installed evaluator closure did not match immutable stock evidence."""


def source_tree_sha256(
    root: Path, *, excluded_relative_paths: frozenset[str] = frozenset()
) -> str:
    """Hash semantic package files with path and length framing.

    The generated stock launch manifest excludes itself from the worker source
    domain; its raw bytes and expanded launch closure are independently pinned
    by Rust, avoiding a self-referential digest.
    """
    allowed = {".py", ".json", ".toml", ".yaml", ".yml"}
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in allowed
            and "__pycache__" not in path.parts
            and path.relative_to(root).as_posix() not in excluded_relative_paths
        ),
        key=lambda path: path.relative_to(root).as_posix().encode("utf-8"),
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def task_manifest(descriptor: StockDistributionDescriptor) -> dict[str, Any]:
    """Load the provider's frozen task/role/capability manifest."""
    value = orjson.loads(_resource_bytes(descriptor.task_manifest_resource))
    if not isinstance(value, dict):
        raise DistributionVerificationError("task manifest root must be an object")
    canonical_dumps(value)
    return value


def _verify_source_attestation(component: SourceComponentLock) -> None:
    distribution = importlib.metadata.distribution(component.distribution)
    attestation = distribution.read_text("aiperf_source_attestation.json")
    if attestation is not None:
        try:
            value = orjson.loads(attestation)
        except orjson.JSONDecodeError as error:
            raise DistributionVerificationError(
                f"evaluator package {component.distribution!r} has malformed source attestation"
            ) from error
        expected = {
            "distribution": component.distribution,
            "version": component.version,
            "commit": component.commit,
            "source_tree_sha256": component.source_tree_sha256,
        }
        if value != expected:
            raise DistributionVerificationError(
                f"evaluator package {component.distribution!r} source attestation drift"
            )
        return
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        raise DistributionVerificationError(
            f"evaluator package {component.distribution!r} lacks source attestation"
        )
    try:
        value = orjson.loads(direct_url)
        commit = value["vcs_info"]["commit_id"]
    except (KeyError, TypeError, orjson.JSONDecodeError) as error:
        raise DistributionVerificationError(
            f"evaluator package {component.distribution!r} has malformed VCS evidence"
        ) from error
    if commit != component.commit:
        raise DistributionVerificationError(
            f"evaluator package {component.distribution!r} commit drift"
        )


def _resource_bytes(name: str) -> bytes:
    return files("aiperf.accuracy.evaluation.manifests").joinpath(name).read_bytes()


_NEMO_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["environment", "solver", "solver_config", "selection"],
    "properties": {
        "environment": {"const": "gsm8k"},
        "solver": {"const": "chat"},
        "environment_config": {"type": "object", "additionalProperties": False},
        "solver_config": {
            "type": "object",
            "additionalProperties": False,
            "required": ["max_tokens"],
            "properties": {
                "max_tokens": {"type": "integer", "minimum": 1},
                "temperature": {"type": "number"},
                "top_p": {"type": "number"},
                "seed": {"type": "integer"},
                "stop": {"type": "array", "items": {"type": "string"}},
                "frequency_penalty": {"type": "number"},
                "presence_penalty": {"type": "number"},
            },
        },
        "selection": {
            "type": "object",
            "additionalProperties": False,
            "required": ["limit", "seed"],
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                "seed": {"const": 0},
            },
        },
    },
}

_OPENBENCH_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["task", "task_args", "epochs"],
    "properties": {
        "task": {"const": "gsm8k"},
        "task_args": {"type": "object", "additionalProperties": False},
        "epochs": {"type": "integer", "minimum": 1, "maximum": 8},
        "limit": {"type": "integer", "minimum": 1, "maximum": 5},
    },
}

NEMO_EVALUATOR_DISTRIBUTION = StockDistributionDescriptor(
    provider_id="nemo_evaluator",
    distribution_id="nvidia_nemo_evaluator_0_4_locked",
    package="nemo-evaluator",
    package_version="0.4.0",
    components=(
        SourceComponentLock(
            distribution="nemo-evaluator",
            import_package="nemo_evaluator",
            version="0.4.0",
            commit="a668af906b46c802984f2d471f15ca83b763092d",
            source_tree_sha256="19ec02c2ab2e3e1d4fb84f65a14c970fa3b776e536f372abc4c536e0e6219a3a",
            source_tree_excluded_paths=(
                "_registry_overrides/regenerate.py",
                "_registry_overrides/terminal_bench_2_1.json",
            ),
        ),
    ),
    dependency_lock_resource="nemo_evaluator.lock.json",
    config_schema_version=1,
    config_schema=_NEMO_SCHEMA,
    task_manifest_resource="nemo_evaluator_tasks.json",
)

OPENBENCH_DISTRIBUTION = StockDistributionDescriptor(
    provider_id="openbench",
    distribution_id="groq_openbench_0_5_3_inspect_0_3_141_locked",
    package="openbench",
    package_version="0.5.3",
    components=(
        SourceComponentLock(
            distribution="openbench",
            import_package="openbench",
            version="0.5.3",
            commit="3f190a835f7fee34ccd96e17242a36a29e0620a6",
            source_tree_sha256="bdfcc39c2423619696d359970e75611dd0aadee6c87a383961b78ab705acf1d5",
        ),
        SourceComponentLock(
            distribution="inspect-ai",
            import_package="inspect_ai",
            version="0.3.141",
            commit="bb78d82dde311b68dbfd0b49f3186b9fc13a1465",
            source_tree_sha256="6bd6016a593ebc0e976285e6416025a0c8a123d8451b0fc180da9a6a17d9794b",
        ),
    ),
    dependency_lock_resource="openbench.lock.json",
    config_schema_version=1,
    config_schema=_OPENBENCH_SCHEMA,
    task_manifest_resource="openbench_tasks.json",
)

STOCK_DISTRIBUTIONS = MappingProxyType(
    {
        NEMO_EVALUATOR_DISTRIBUTION.distribution_id: NEMO_EVALUATOR_DISTRIBUTION,
        OPENBENCH_DISTRIBUTION.distribution_id: OPENBENCH_DISTRIBUTION,
    }
)


def selected_descriptor(
    provider_id: str, distribution_id: str
) -> StockDistributionDescriptor:
    """Resolve only a factory-registered stock distribution identity."""
    descriptor = STOCK_DISTRIBUTIONS.get(distribution_id)
    if descriptor is None or descriptor.provider_id != provider_id:
        raise DistributionVerificationError(
            "unregistered provider distribution selection"
        )
    return descriptor


def production_launch_available(descriptor: StockDistributionDescriptor) -> bool:
    """Return true only when the worker-visible closure attestation passes.

    Rust separately resolves and materializes the pinned logical runtime/ELF
    closure before advertising product availability.
    """
    try:
        descriptor.verify_installed_closure()
        if not executable_tasks(descriptor):
            return False
    except DistributionVerificationError:
        return False
    return True


def executable_tasks(descriptor: StockDistributionDescriptor) -> tuple[str, ...]:
    """Return exact statically executable task IDs without importing providers."""
    manifest = task_manifest(descriptor)
    entries = (
        manifest.get("environments", {})
        if descriptor.provider_id == "nemo_evaluator"
        else manifest.get("tasks", {})
    )
    if not isinstance(entries, dict):
        raise DistributionVerificationError(
            "provider task manifest has malformed entries"
        )
    return tuple(
        name
        for name, value in entries.items()
        if isinstance(value, dict) and value.get("executable") is True
    )
