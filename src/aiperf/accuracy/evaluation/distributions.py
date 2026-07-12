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
from aiperf.accuracy.evaluation.contracts import (
    EvaluationIdentityComponent,
    EvaluationIdentityOverlay,
    EvaluationWorkerIdentity,
)

WORKER_MODULE = "aiperf.accuracy.evaluation.worker"
CONTROL_BOOTSTRAP = (
    "/runtime/lib/python3.12/site-packages/"
    "aiperf/accuracy/evaluation/control_bootstrap.py"
)
CONTROL_READ_FD = 3
CONTROL_WRITE_FD = 4
BOOTSTRAP_PROCESS_LIMIT_ENV = "AIPERF_EVALUATOR_BOOTSTRAP_PROCESS_LIMIT"
STOCK_PROCESS_LIMIT = 1024
STAGING_ROOT = "/staging"
ISOLATION_PROFILE = "linux-bubblewrap-rootfs-process-tree-v3"
SOURCE_TREE_DIGEST_POLICY = "aiperf-semantic-source-tree-sha256-v1"
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
class SourceOverlayLock:
    """One ordered, digest-pinned patch over an upstream package tree."""

    overlay_id: str
    resource: str
    resource_sha256: str
    targets: tuple[str, ...]


@dataclass(frozen=True)
class RestoredBaseFileLock:
    """One upstream-commit file omitted from the built wheel and restored exactly."""

    relative_path: str
    content_sha256: str


@dataclass(frozen=True)
class SourceComponentLock:
    """Exact installed Python package/source identity."""

    distribution: str
    import_package: str
    version: str
    commit: str
    base_source_tree_sha256: str
    source_tree_sha256: str
    source_tree_excluded_paths: tuple[str, ...] = ()
    restored_base_files: tuple[RestoredBaseFileLock, ...] = ()
    overlays: tuple[SourceOverlayLock, ...] = ()


@dataclass(frozen=True)
class DistributionEvidence:
    """Worker-observed evidence independently checked by the Rust factory."""

    provider_source_sha256: str
    worker_source_sha256: str
    dependency_lock_sha256: str


@dataclass(frozen=True)
class CompatibilityProxyRouteDescriptor:
    """One exact dialect selector to Rust-owned semantic-route grant."""

    dialect: str
    selector: str
    service_id: str
    purpose: str
    semantic_operation_id: str
    restricted_payload: bool


@dataclass(frozen=True)
class CompatibilityProxyDescriptor:
    """Exact local adapter set and maximum grant for one distribution."""

    dialects: tuple[str, ...]
    routes: tuple[CompatibilityProxyRouteDescriptor, ...]
    max_operations: int
    max_concurrent_operations: int
    max_request_bytes: int
    max_response_bytes: int
    max_stream_events: int
    expires_after_ms: int


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
    compatibility_proxy: CompatibilityProxyDescriptor | None

    @property
    def config_schema_sha256(self) -> str:
        """Return the exact authored-config schema fingerprint."""
        return canonical_sha256(self.config_schema)

    @property
    def fixed_argv(self) -> tuple[str, ...]:
        """Return literal worker arguments owned by the stock factory."""
        return (
            "-I",
            "-S",
            CONTROL_BOOTSTRAP,
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
                "HOME": f"{STAGING_ROOT}/home",
                "TMPDIR": f"{STAGING_ROOT}/tmp",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "XDG_CONFIG_HOME": f"{STAGING_ROOT}/.xdg-config",
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
                    "base_source_tree_sha256": component.base_source_tree_sha256,
                    "overlays": [
                        {
                            "overlay_id": overlay.overlay_id,
                            "artifact_content_sha256": overlay.resource_sha256,
                        }
                        for overlay in component.overlays
                    ],
                    "effective_source_tree_sha256": actual_tree,
                }
            )
        lock_bytes = _resource_bytes(self.dependency_lock_resource)
        task_manifest = _resource_bytes(self.task_manifest_resource)
        # Loading now proves packaged manifests are valid strict JSON; their
        # bytes, not a mutable filesystem path, are the attested domain.
        orjson.loads(lock_bytes)
        orjson.loads(task_manifest)
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

    Policy ``aiperf-semantic-source-tree-sha256-v1`` selects the semantic
    suffix set, sorts UTF-8 relative paths bytewise, and feeds SHA-256 with an
    unsigned 8-byte big-endian path length, path bytes, unsigned 8-byte
    content length, and raw content bytes for every file.

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


def source_identity_component(
    component: SourceComponentLock,
) -> EvaluationIdentityComponent:
    """Project one source lock into the attested evaluation identity shape."""
    return EvaluationIdentityComponent(
        name=component.import_package,
        version=component.version,
        source_sha256=component.source_tree_sha256,
        source_commit=component.commit if component.overlays else None,
        base_source_sha256=(
            component.base_source_tree_sha256 if component.overlays else None
        ),
        overlay_policy=(
            "aiperf-unified-diff-overlay-v1" if component.overlays else None
        ),
        overlays=tuple(
            EvaluationIdentityOverlay(
                overlay_id=overlay.overlay_id,
                artifact_content_sha256=overlay.resource_sha256,
            )
            for overlay in component.overlays
        ),
    )


def distribution_identity_components(
    descriptor: StockDistributionDescriptor,
    *,
    worker_source_sha256: str,
    dependency_lock_sha256: str,
) -> tuple[EvaluationIdentityComponent, ...]:
    """Return the factory-attested ordered component graph for a stock run."""
    return (
        *(source_identity_component(component) for component in descriptor.components),
        EvaluationIdentityComponent(
            name=f"{descriptor.provider_id}_task_registry",
            version="aiperf-evaluator-task-manifest-v1",
            source_sha256=hashlib.sha256(
                _resource_bytes(descriptor.task_manifest_resource)
            ).hexdigest(),
        ),
        EvaluationIdentityComponent(
            name="aiperf_evaluation_worker",
            version="evaluator-protocol-2",
            source_sha256=worker_source_sha256,
        ),
        EvaluationIdentityComponent(
            name=f"{descriptor.provider_id}_dependency_lock",
            version="aiperf-evaluator-lock-v2",
            source_sha256=dependency_lock_sha256,
        ),
    )


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
        if not isinstance(value, dict) or set(value) != {
            "format",
            "distribution",
            "version",
            "base",
            "overlay_policy",
            "overlays",
            "effective_source_tree_sha256",
            "effective_source_tree_digest_policy",
        }:
            raise DistributionVerificationError(
                f"evaluator package {component.distribution!r} source attestation image drift"
            )
        expected_base = {
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
        }
        if (
            value["format"] != "aiperf-pinned-source-overlay-v1"
            or value["distribution"] != component.distribution
            or value["version"] != component.version
            or value["base"] != expected_base
            or value["overlay_policy"] != "aiperf-unified-diff-overlay-v1"
            or value["effective_source_tree_sha256"] != component.source_tree_sha256
            or value["effective_source_tree_digest_policy"] != SOURCE_TREE_DIGEST_POLICY
        ):
            raise DistributionVerificationError(
                f"evaluator package {component.distribution!r} source attestation drift"
            )
        raw_overlays = value["overlays"]
        if not isinstance(raw_overlays, list) or len(raw_overlays) != len(
            component.overlays
        ):
            raise DistributionVerificationError("source overlay inventory drift")
        package_spec = importlib.util.find_spec(component.import_package)
        if package_spec is None or not package_spec.submodule_search_locations:
            raise DistributionVerificationError("overlaid package tree is unavailable")
        package_root = Path(tuple(package_spec.submodule_search_locations)[0])
        seen_targets: set[str] = set()
        for ordinal, (raw, expected) in enumerate(
            zip(raw_overlays, component.overlays, strict=True)
        ):
            if not isinstance(raw, dict) or set(raw) != {
                "ordinal",
                "overlay_id",
                "resource",
                "artifact_content_sha256",
                "targets",
            }:
                raise DistributionVerificationError("source overlay entry drift")
            resource = (
                Path(__file__).resolve().parent / "source_overlays" / expected.resource
            )
            if (
                raw["ordinal"] != ordinal
                or raw["overlay_id"] != expected.overlay_id
                or raw["resource"] != expected.resource
                or raw["artifact_content_sha256"] != expected.resource_sha256
                or not resource.is_file()
                or hashlib.sha256(resource.read_bytes()).hexdigest()
                != expected.resource_sha256
            ):
                raise DistributionVerificationError("source overlay provenance drift")
            targets = raw["targets"]
            if not isinstance(targets, list) or len(targets) != len(expected.targets):
                raise DistributionVerificationError("source overlay target drift")
            for target, relative in zip(targets, expected.targets, strict=True):
                if not isinstance(target, dict) or set(target) != {
                    "relative_path",
                    "base_content_sha256",
                    "effective_content_sha256",
                }:
                    raise DistributionVerificationError(
                        "source overlay target image drift"
                    )
                if relative in seen_targets or target["relative_path"] != relative:
                    raise DistributionVerificationError(
                        "source overlay target order drift"
                    )
                seen_targets.add(relative)
                base_digest = target["base_content_sha256"]
                if base_digest is not None:
                    _require_attested_sha256(base_digest, "base overlay target")
                effective_digest = _require_attested_sha256(
                    target["effective_content_sha256"], "effective overlay target"
                )
                effective_path = package_root / relative
                if (
                    not effective_path.is_file()
                    or hashlib.sha256(effective_path.read_bytes()).hexdigest()
                    != effective_digest
                ):
                    raise DistributionVerificationError(
                        "effective source overlay target digest drift"
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


def _require_attested_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DistributionVerificationError(f"{label} is not lowercase SHA-256")
    return value


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
            base_source_tree_sha256="19ec02c2ab2e3e1d4fb84f65a14c970fa3b776e536f372abc4c536e0e6219a3a",
            source_tree_sha256="f091e3989b0c21b704f7ce4143cc08f84a0a5023c8cac3da491d4129ce32bfda",
            source_tree_excluded_paths=(
                "_registry_overrides/regenerate.py",
                "_registry_overrides/terminal_bench_2_1.json",
            ),
            overlays=(
                SourceOverlayLock(
                    overlay_id="nemo-host-boundary",
                    resource="nemo_evaluator/001-host-boundary.patch",
                    resource_sha256="5abebc96f162d5322221e718c4bdabe212bfaa576b10020ef8ca03c3ae923ea2",
                    targets=(
                        "engine/host.py",
                        "engine/session.py",
                        "hosts/__init__.py",
                        "hosts/local/__init__.py",
                        "hosts/local/inference.py",
                    ),
                ),
                SourceOverlayLock(
                    overlay_id="nemo-run-evaluation-host",
                    resource="nemo_evaluator/002-run-evaluation-host.patch",
                    resource_sha256="1e82fac6d1d18bc061cee4c3e55c4505905b4cce813251810c0a1e3135c8f18f",
                    targets=("engine/eval_loop.py", "engine/model_client.py"),
                ),
            ),
        ),
    ),
    dependency_lock_resource="nemo_evaluator.lock.json",
    config_schema_version=1,
    config_schema=_NEMO_SCHEMA,
    task_manifest_resource="nemo_evaluator_tasks.json",
    compatibility_proxy=None,
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
            base_source_tree_sha256="bdfcc39c2423619696d359970e75611dd0aadee6c87a383961b78ab705acf1d5",
            source_tree_sha256="e859b2a55a3365525b8187f178870ac0b94aeec1f799c8449fbab3758f34b878",
            overlays=(
                SourceOverlayLock(
                    overlay_id="openbench-explicit-runtime",
                    resource="openbench/001-explicit-runtime.patch",
                    resource_sha256="6f583687b2ee339e35552233fae0534e8008ccfc653f1ea9c2a6092ae8e05b88",
                    targets=(
                        "__init__.py",
                        "model/_providers/aiperf_pipe.py",
                        "runtime/__init__.py",
                        "runtime/artifacts.py",
                        "runtime/host.py",
                        "runtime/inspect_session.py",
                    ),
                ),
            ),
        ),
        SourceComponentLock(
            distribution="inspect-ai",
            import_package="inspect_ai",
            version="0.3.141",
            commit="bb78d82dde311b68dbfd0b49f3186b9fc13a1465",
            base_source_tree_sha256="6bd6016a593ebc0e976285e6416025a0c8a123d8451b0fc180da9a6a17d9794b",
            source_tree_sha256="69e099965072804ba92e65c620e0f8d08c8973e728ac2a41a7d2777703d35022",
            restored_base_files=(
                RestoredBaseFileLock(
                    relative_path="tool/_tools/_computer/_resources/image_home_dir/.config/Code/User/settings.json",
                    content_sha256="9603332ca75385c1a519196a826245062b26686b5507b371d748600ec4b955a0",
                ),
                RestoredBaseFileLock(
                    relative_path="tool/_tools/_computer/_resources/tool/.vscode/settings.json",
                    content_sha256="71f7809c3470a607a7b083d8e4b78808d8793fea391f0f79d6c7661eda88fcb5",
                ),
                RestoredBaseFileLock(
                    relative_path="tool/_tools/_computer/_resources/tool/pyproject.toml",
                    content_sha256="19c1446057e9bd4d6c960c078bc970509dc0b5b57c93d602ce530bca0d069041",
                ),
            ),
            overlays=(
                SourceOverlayLock(
                    overlay_id="inspect-model-call-context",
                    resource="inspect_ai/001-public-model-call-context.patch",
                    resource_sha256="8b1e68251eddd82f3336f035e5dab912439e0076c821a86121720e5f1bba2d94",
                    targets=(
                        "_eval/task/run.py",
                        "model/__init__.py",
                        "model/_call_context.py",
                    ),
                ),
                SourceOverlayLock(
                    overlay_id="inspect-cache-veto",
                    resource="inspect_ai/002-cache-veto.patch",
                    resource_sha256="3ff98d606d66e2b3db220b106b8770a5cd829283e43736095b23a2c71a616a68",
                    targets=("model/_model.py",),
                ),
                SourceOverlayLock(
                    overlay_id="inspect-entry-point-policy",
                    resource="inspect_ai/003-entry-point-policy.patch",
                    resource_sha256="16d97eff5ce6946a967fdeaa685e6a48e17d28eb0095d1de5d42c1e2ce3186f1",
                    targets=("_util/entrypoints.py", "extensions.py"),
                ),
            ),
        ),
    ),
    dependency_lock_resource="openbench.lock.json",
    config_schema_version=1,
    config_schema=_OPENBENCH_SCHEMA,
    task_manifest_resource="openbench_tasks.json",
    compatibility_proxy=CompatibilityProxyDescriptor(
        dialects=("openai_chat_completions",),
        routes=(
            CompatibilityProxyRouteDescriptor(
                dialect="openai_chat_completions",
                selector="candidate",
                service_id="candidate",
                purpose="primary",
                semantic_operation_id="model.generate",
                restricted_payload=False,
            ),
        ),
        max_operations=40,
        max_concurrent_operations=40,
        max_request_bytes=40 * 8 * 1024 * 1024,
        max_response_bytes=40 * 8 * 1024 * 1024,
        max_stream_events=40,
        expires_after_ms=24 * 60 * 60 * 1000,
    ),
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
