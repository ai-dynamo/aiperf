# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Product discovery tests for deployment-owned stock evaluator roots."""

from __future__ import annotations

import base64
import csv
import hashlib
import shutil
from importlib import metadata
from pathlib import Path

import pytest

from aiperf.orchestrator import runner_installation


def _payload(root: Path) -> tuple[Path, ...]:
    files: dict[str, bytes] = {
        "runtime/bin/python3.12": b"pinned-python",
        "nemo/lib/python3.12/site-packages/nemo.dist-info/RECORD": b"nemo-record",
        "openbench/lib/python3.12/site-packages/openbench.dist-info/RECORD": b"openbench-record",
        "system/usr/lib/libc.so.6": b"pinned-system-library",
    }
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    roots = []
    entries = []
    for root_id, kind, relative in runner_installation._PROVIDER_ROOT_SPECS:
        members = {
            path.relative_to(root / relative).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (root / relative).rglob("*")
            if path.is_file()
        }
        entries.append(
            {
                "file_count": len(members),
                "id": root_id,
                "kind": kind,
                "path": relative,
                "tree_sha256": runner_installation._provider_tree_sha256(members),
            }
        )
        roots.append((root / relative).resolve())
    registry = {
        "platform": "linux-x86_64",
        "roots": entries,
        "schema_version": runner_installation._PROVIDER_ROOTS_SCHEMA,
    }
    (root / runner_installation._PROVIDER_ROOTS_REGISTRY).write_bytes(
        runner_installation._canonical_provider_registry(registry)
    )
    return tuple(roots)


def _record_digest(content: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode()
    return f"sha256={encoded.rstrip('=')}"


def _companion_distribution(
    prefix: Path,
    payload: Path,
    *,
    omit_payload: str | None = None,
    duplicate_payload: str | None = None,
) -> tuple[metadata.Distribution, Path, Path]:
    site_packages = prefix / "lib/python3.12/site-packages"
    dist_info = site_packages / "aiperf_runner-0.11.0.dist-info"
    installed_payload = site_packages / runner_installation._PROVIDER_ROOTS_WHEEL_PREFIX
    dist_info.mkdir(parents=True)
    shutil.copytree(payload, installed_payload)
    binary = prefix / "bin/aiperf-runner"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native-runner")
    binary.chmod(0o755)
    metadata_bytes = b"Metadata-Version: 2.3\nName: aiperf-runner\nVersion: 0.11.0\n"
    (dist_info / "METADATA").write_bytes(metadata_bytes)

    rows = []
    binary_relative = "../../../bin/aiperf-runner"
    rows.append(
        (binary_relative, _record_digest(binary.read_bytes()), binary.stat().st_size)
    )
    rows.append(
        (
            "aiperf_runner-0.11.0.dist-info/METADATA",
            _record_digest(metadata_bytes),
            len(metadata_bytes),
        )
    )
    for path in sorted(installed_payload.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(site_packages).as_posix()
        logical = path.relative_to(installed_payload).as_posix()
        if logical == omit_payload:
            continue
        content = path.read_bytes()
        rows.append((relative, _record_digest(content), len(content)))
        if logical == duplicate_payload:
            rows.append((relative, _record_digest(content), len(content)))
    rows.append(("aiperf_runner-0.11.0.dist-info/RECORD", "", ""))
    with (dist_info / "RECORD").open("w", newline="", encoding="utf-8") as output:
        csv.writer(output, lineterminator="\n").writerows(rows)
    return metadata.Distribution.at(dist_info), binary.resolve(), installed_payload


def _select(
    monkeypatch: pytest.MonkeyPatch,
    distribution: metadata.Distribution,
    *,
    explicit: Path | None = None,
) -> tuple[runner_installation.RunnerInstallation, tuple[Path, ...]]:
    selected: list[tuple[Path, ...]] = []
    monkeypatch.delenv("AIPERF_RUNNER_BIN", raising=False)
    monkeypatch.setattr(
        runner_installation.metadata,
        "distribution",
        lambda _name: distribution,
    )
    monkeypatch.setattr(
        runner_installation,
        "_load_capabilities",
        lambda _binary, roots: selected.append(roots) or {"protocol_versions": [2]},
    )
    installation = runner_installation.RunnerInstallation.resolve(explicit)
    assert len(selected) == 1
    return installation, selected[0]


def test_installed_companion_default_uses_only_its_complete_record_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "staged"
    expected_staged = _payload(staged)
    distribution, binary, installed_payload = _companion_distribution(
        tmp_path / "prefix", staged
    )
    monkeypatch.setenv("AIPERF_EVALUATOR_PROVIDER_ROOTS", "/attacker/ambient")

    installation, selected = _select(monkeypatch, distribution)

    assert installation.binary == binary
    assert selected == tuple(
        (installed_payload / root.relative_to(staged)).resolve()
        for root in expected_staged
    )
    assert installation.provider_roots == selected


def test_installed_companion_binary_and_roots_share_one_distribution_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "staged"
    expected_staged = _payload(staged)
    distribution, binary, installed_payload = _companion_distribution(
        tmp_path / "prefix", staged
    )
    lookups = 0

    def selected_distribution(_name: str) -> metadata.Distribution:
        nonlocal lookups
        lookups += 1
        if lookups != 1:
            pytest.fail("runner deployment queried a second companion distribution")
        return distribution

    selected: list[tuple[Path, ...]] = []
    monkeypatch.delenv("AIPERF_RUNNER_BIN", raising=False)
    monkeypatch.setattr(
        runner_installation.metadata,
        "distribution",
        selected_distribution,
    )
    monkeypatch.setattr(
        runner_installation,
        "_load_capabilities",
        lambda _binary, roots: selected.append(roots) or {"protocol_versions": [2]},
    )

    installation = runner_installation.RunnerInstallation.resolve()

    assert lookups == 1
    assert installation.binary == binary
    assert selected == [
        tuple(
            (installed_payload / root.relative_to(staged)).resolve()
            for root in expected_staged
        )
    ]


def test_embedded_root_file_named_like_runner_is_not_a_second_native_script(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged"
    _payload(staged)
    nested = staged / "nemo/bin/aiperf-runner"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"evaluator-environment-console-script")
    distribution, binary, _ = _companion_distribution(tmp_path / "prefix", staged)

    assert (
        runner_installation._companion_binary_from_distribution(distribution) == binary
    )


@pytest.mark.parametrize(
    "fault", ["unrecorded", "missing_record", "duplicate_record", "tampered"]
)
def test_installed_companion_invalid_payload_fails_to_empty_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    staged = tmp_path / "staged"
    _payload(staged)
    omitted = "nemo/lib/python3.12/site-packages/nemo.dist-info/RECORD"
    distribution, _, installed_payload = _companion_distribution(
        tmp_path / "prefix",
        staged,
        omit_payload=omitted if fault == "missing_record" else None,
        duplicate_payload=omitted if fault == "duplicate_record" else None,
    )
    if fault == "unrecorded":
        (installed_payload / "nemo/unrecorded.py").write_text("pass\n")
    elif fault == "tampered":
        assert len(b"tamper-byte") == len((installed_payload / omitted).read_bytes())
        (installed_payload / omitted).write_bytes(b"tamper-byte")

    _, selected = _select(monkeypatch, distribution)

    assert selected == ()


def test_installed_companion_rejects_a_symlinked_payload_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "staged"
    _payload(staged)
    distribution, _, installed_payload = _companion_distribution(
        tmp_path / "prefix", staged
    )
    relocated = tmp_path / "relocated-payload"
    installed_payload.rename(relocated)
    installed_payload.symlink_to(relocated, target_is_directory=True)

    _, selected = _select(monkeypatch, distribution)

    assert selected == ()


def test_adjacent_sidecar_rejects_a_symlinked_provider_root(tmp_path: Path) -> None:
    sidecar = tmp_path / "aiperf-runner.evaluator-roots"
    _payload(sidecar)
    relocated = tmp_path / "relocated-nemo"
    (sidecar / "nemo").rename(relocated)
    (sidecar / "nemo").symlink_to(relocated, target_is_directory=True)

    assert runner_installation._provider_roots_from_registry(sidecar) == ()


def test_payload_hashing_is_streamed_and_exact(tmp_path: Path) -> None:
    content = b"a" * (2 * 1024 * 1024 + 17)
    payload = tmp_path / "large-payload"
    payload.write_bytes(content)

    assert (
        runner_installation._file_sha256(payload) == hashlib.sha256(content).hexdigest()
    )


def test_explicit_runner_uses_only_its_adjacent_generated_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_staged = tmp_path / "installed-staged"
    _payload(installed_staged)
    distribution, _, _ = _companion_distribution(
        tmp_path / "installed-prefix", installed_staged
    )
    explicit = tmp_path / "development/aiperf-runner"
    explicit.parent.mkdir()
    explicit.write_bytes(b"development-runner")
    explicit.chmod(0o755)
    sidecar = explicit.with_name(
        f"{explicit.name}{runner_installation._PROVIDER_ROOTS_SIDECAR_SUFFIX}"
    )
    expected = _payload(sidecar)

    installation, selected = _select(
        monkeypatch,
        distribution,
        explicit=explicit,
    )

    assert installation.binary == explicit.resolve()
    assert selected == expected


def test_environment_runner_uses_only_its_adjacent_generated_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured/aiperf-runner"
    configured.parent.mkdir()
    configured.write_bytes(b"configured-runner")
    configured.chmod(0o755)
    sidecar = configured.with_name(
        f"{configured.name}{runner_installation._PROVIDER_ROOTS_SIDECAR_SUFFIX}"
    )
    expected = _payload(sidecar)
    selected: list[tuple[Path, ...]] = []
    monkeypatch.setenv("AIPERF_RUNNER_BIN", str(configured))
    monkeypatch.setattr(
        runner_installation.metadata,
        "distribution",
        lambda _name: pytest.fail("environment selection queried companion metadata"),
    )
    monkeypatch.setattr(
        runner_installation,
        "_load_capabilities",
        lambda _binary, roots: selected.append(roots) or {"protocol_versions": [2]},
    )

    installation = runner_installation.RunnerInstallation.resolve()

    assert installation.binary == configured.resolve()
    assert selected == [expected]


def test_path_runner_uses_only_its_adjacent_generated_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovered = tmp_path / "path/aiperf-runner"
    discovered.parent.mkdir()
    discovered.write_bytes(b"path-runner")
    discovered.chmod(0o755)
    sidecar = discovered.with_name(
        f"{discovered.name}{runner_installation._PROVIDER_ROOTS_SIDECAR_SUFFIX}"
    )
    expected = _payload(sidecar)
    selected: list[tuple[Path, ...]] = []
    monkeypatch.delenv("AIPERF_RUNNER_BIN", raising=False)
    monkeypatch.setattr(
        runner_installation.metadata,
        "distribution",
        lambda _name: (_ for _ in ()).throw(metadata.PackageNotFoundError),
    )
    monkeypatch.setattr(runner_installation.shutil, "which", lambda _name: discovered)
    monkeypatch.setattr(
        runner_installation,
        "_load_capabilities",
        lambda _binary, roots: selected.append(roots) or {"protocol_versions": [2]},
    )

    installation = runner_installation.RunnerInstallation.resolve()

    assert installation.binary == discovered.resolve()
    assert selected == [expected]


def test_missing_sidecar_clears_ambient_provider_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "installed-staged"
    _payload(staged)
    distribution, _, _ = _companion_distribution(tmp_path / "installed-prefix", staged)
    explicit = tmp_path / "development-runner"
    explicit.write_bytes(b"development-runner")
    explicit.chmod(0o755)
    monkeypatch.setenv("AIPERF_EVALUATOR_PROVIDER_ROOTS", "/attacker/ambient")

    installation, selected = _select(
        monkeypatch,
        distribution,
        explicit=explicit,
    )

    assert selected == ()
    child_environment = runner_installation._runner_subprocess_environment(
        installation.provider_roots
    )
    assert "AIPERF_EVALUATOR_PROVIDER_ROOTS" not in child_environment
