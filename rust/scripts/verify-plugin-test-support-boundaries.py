#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify test-support exclusion through each executable shipping boundary."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class BoundaryError(RuntimeError):
    """A shipping boundary includes or can select plugin test support."""


@dataclass(frozen=True)
class ContainerBoundary:
    """Exact default-stage instruction identity projected as a test image boundary."""

    stage: str
    instruction_sha256: str

    @property
    def image_digest(self) -> str:
        """Return the digest-qualified boundary token used by typed K8s projection."""
        return f"sha256:{self.instruction_sha256}"


EXPECTED_REACHABLE_DOCKER_PROJECTION_SHA256 = (
    "85179fec37130992b2fe6037ea1d3063a641ebc9f053566dadd054570f4efbf5"
)


def run(command: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    """Run one repository-owned boundary command and retain diagnostics."""
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise BoundaryError(
            f"command failed ({' '.join(command)}): {completed.stderr.strip()}"
        )
    return completed


def verify_native_install(root: Path) -> None:
    """Execute Make's native-install graph and pin its only installed payload."""
    lines = run(
        ["make", "--no-print-directory", "-n", "install-native"], root
    ).stdout.splitlines()
    commands = [shlex.split(line) for line in lines if line.strip()]
    expected = [
        [
            "cargo",
            "build",
            "--manifest-path",
            "rust/Cargo.toml",
            "--release",
            "-p",
            "aiperf-cli",
            "--features",
            "full",
        ],
        ["mkdir", "-p", "dist/native-bin"],
        ["cp", "rust/target/release/aiperf", "dist/native-bin/aiperf"],
        ["echo", "Pure-Rust aiperf installed to dist/native-bin/. Add it to PATH:"],
        ["echo", '  export PATH="$(pwd)/dist/native-bin:$PATH"'],
        [
            "echo",
            "Then: aiperf profile --model M --url 127.0.0.1:8000 "
            "--endpoint-type chat --concurrency 1,2,4 --request-count 100",
        ],
    ]
    if commands != expected:
        raise BoundaryError(
            f"native install command/payload projection drift: {commands!r}"
        )


def verify_wheel(root: Path, executable: Path) -> None:
    """Build and repack the real wheel, then validate its complete RECORD."""
    with tempfile.TemporaryDirectory(prefix="aiperf-task2-wheel-") as directory:
        output = Path(directory)
        run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                os.fspath(output),
            ],
            root,
        )
        run(
            [
                sys.executable,
                "tools/wheel_repack.py",
                "--wheel-dir",
                os.fspath(output),
                "--binary",
                os.fspath(executable),
            ],
            root,
        )
        wheels = list(output.glob("aiperf-*.whl"))
        if len(wheels) != 1:
            raise BoundaryError(f"wheel boundary produced {len(wheels)} artifacts")
        with zipfile.ZipFile(wheels[0]) as archive:
            names = archive.namelist()
            records = [name for name in names if name.endswith(".dist-info/RECORD")]
            if len(records) != 1:
                raise BoundaryError("wheel must contain exactly one RECORD")
            rows = list(csv.reader(archive.read(records[0]).decode().splitlines()))
            recorded = [row[0] for row in rows]
            if len(recorded) != len(set(recorded)) or set(recorded) != set(names):
                raise BoundaryError("wheel RECORD is not an exact artifact inventory")
            forbidden = [
                name
                for name in names
                if "rust/plugin-test-support" in name
                or "aiperf-plugin-test-support" in name
            ]
            if forbidden:
                raise BoundaryError(f"wheel ships plugin test support: {forbidden!r}")
            scripts = [name for name in names if name.endswith(".data/scripts/aiperf")]
            if len(scripts) != 1 or archive.read(scripts[0]) != executable.read_bytes():
                raise BoundaryError(
                    "wheel does not carry exactly the selected native executable"
                )


def dockerfile_instructions(path: Path) -> list[str]:
    """Return logical non-comment Dockerfile instructions."""
    instructions: list[str] = []
    pending = ""
    for physical in path.read_text().splitlines():
        stripped = physical.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pending = f"{pending} {stripped}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        instructions.append(pending)
        pending = ""
    if pending:
        raise BoundaryError("Dockerfile ends with an incomplete instruction")
    return instructions


def normalized_container_path(workdir: PurePosixPath, value: str) -> PurePosixPath:
    """Resolve one Docker path without consulting the host filesystem."""
    path = PurePosixPath(value)
    return path if path.is_absolute() else workdir / path


def parse_dockerfile(
    path: Path,
) -> tuple[
    dict[str, str | None],
    dict[str, list[tuple[int, str | None, PurePosixPath, PurePosixPath]]],
    dict[str, list[tuple[int, str]]],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, list[str]],
    list[str],
]:
    """Parse stage inheritance and all payload-mutating instructions."""
    parents: dict[str, str | None] = {}
    copies: dict[str, list[tuple[int, str | None, PurePosixPath, PurePosixPath]]] = {}
    runs: dict[str, list[tuple[int, str]]] = {}
    adds: dict[str, list[str]] = {}
    onbuilds: dict[str, list[str]] = {}
    stage_instructions: dict[str, list[str]] = {}
    stage_order: list[str] = []
    workdirs: dict[str, PurePosixPath] = {}
    stage: str | None = None
    for ordinal, instruction in enumerate(dockerfile_instructions(path)):
        operation, _, arguments = instruction.partition(" ")
        operation = operation.upper()
        if operation == "FROM":
            words = shlex.split(arguments)
            if len(words) < 3 or words[-2].upper() != "AS":
                raise BoundaryError(f"every Docker stage must be named: {instruction}")
            stage = words[-1]
            if stage in parents:
                raise BoundaryError(f"duplicate Docker stage name: {stage}")
            parent_image = words[-3]
            parent = parent_image if parent_image in parents else None
            parents[stage] = parent
            copies[stage] = []
            runs[stage] = []
            adds[stage] = []
            onbuilds[stage] = []
            stage_instructions[stage] = [instruction]
            stage_order.append(stage)
            workdirs[stage] = workdirs.get(parent, PurePosixPath("/"))
            continue
        if stage is None:
            raise BoundaryError("Dockerfile instruction precedes the first stage")
        stage_instructions[stage].append(instruction)
        if operation == "WORKDIR":
            workdirs[stage] = normalized_container_path(workdirs[stage], arguments)
            continue
        if operation == "RUN":
            runs[stage].append((ordinal, arguments))
            continue
        if operation == "ADD":
            adds[stage].append(arguments)
            continue
        if operation == "ONBUILD":
            onbuilds[stage].append(arguments)
            continue
        if operation != "COPY":
            continue
        copy_arguments = arguments.lstrip()
        source_stage = None
        while copy_arguments.startswith("--"):
            option, separator, copy_arguments = copy_arguments.partition(" ")
            if not separator or "=" not in option:
                raise BoundaryError(f"unsupported Docker COPY option: {instruction}")
            name, value = option[2:].split("=", 1)
            if name == "from":
                source_stage = value
            elif name not in {"chown", "chmod"}:
                raise BoundaryError(f"unsupported Docker COPY option --{name}")
            copy_arguments = copy_arguments.lstrip()
        if copy_arguments.startswith("["):
            decoded = json.loads(copy_arguments)
            if not isinstance(decoded, list) or not all(
                isinstance(item, str) for item in decoded
            ):
                raise BoundaryError(f"invalid JSON-form Docker COPY: {instruction}")
            operands = decoded
        else:
            operands = shlex.split(copy_arguments)
        if len(operands) < 2:
            raise BoundaryError(f"unsupported Docker COPY: {instruction}")
        destination = normalized_container_path(workdirs[stage], operands[-1])
        sources = operands[:-1]
        for source in sources:
            source_path = PurePosixPath(source)
            mapped_destination = destination
            if len(sources) > 1 or operands[-1].endswith("/"):
                mapped_destination /= source_path.name
            copies[stage].append(
                (ordinal, source_stage, source_path, mapped_destination)
            )
    return parents, copies, runs, adds, onbuilds, stage_instructions, stage_order


def is_within(path: PurePosixPath, ancestor: PurePosixPath) -> bool:
    """Return whether path equals or descends from ancestor."""
    return path == ancestor or ancestor in path.parents


def verify_final_container(root: Path) -> ContainerBoundary:
    """Resolve repository-source COPY ancestry into the actual runtime stage."""
    (
        parents,
        copies,
        runs,
        adds,
        onbuilds,
        stage_instructions,
        stage_order,
    ) = parse_dockerfile(root / "Dockerfile")
    forbidden = PurePosixPath("rust/plugin-test-support")

    def context_contains(source: PurePosixPath) -> bool:
        source = PurePosixPath(os.fspath(source).removeprefix("/"))
        return is_within(forbidden, source) or is_within(source, forbidden)

    def stage_contains(
        stage: str, query: PurePosixPath, seen: set[tuple[str, str]]
    ) -> bool:
        identity = (stage, os.fspath(query))
        if identity in seen:
            raise BoundaryError(f"cyclic Docker stage provenance at {identity!r}")
        seen = {*seen, identity}
        parent = parents[stage]
        if parent is not None and stage_contains(parent, query, seen):
            return True
        for _, source_stage, source, destination in copies[stage]:
            if is_within(query, destination):
                relative = query.relative_to(destination)
                upstream = source / relative
            elif is_within(destination, query):
                upstream = source
            else:
                continue
            if source_stage is None:
                if context_contains(upstream):
                    return True
            elif source_stage in parents and stage_contains(
                source_stage, upstream, seen
            ):
                return True
        return False

    if "runtime" not in parents or not stage_order:
        raise BoundaryError("Dockerfile omits the final runtime stage")
    final_stage = stage_order[-1]
    inherited = final_stage
    while inherited != "runtime":
        parent = parents[inherited]
        if parent is None:
            break
        inherited = parent
    if inherited != "runtime":
        raise BoundaryError("default Docker target does not inherit the runtime stage")

    reachable: set[str] = set()

    def visit(stage: str) -> None:
        if stage in reachable:
            return
        reachable.add(stage)
        parent = parents[stage]
        if parent is not None:
            visit(parent)
        for _, source_stage, _, _ in copies[stage]:
            if source_stage in parents:
                visit(source_stage)

    visit(final_stage)
    instruction_projection = [
        {"stage": stage, "instructions": stage_instructions[stage]}
        for stage in stage_order
        if stage in reachable
    ]
    projection_bytes = json.dumps(
        instruction_projection,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    projection_sha256 = hashlib.sha256(projection_bytes).hexdigest()
    if projection_sha256 != EXPECTED_REACHABLE_DOCKER_PROJECTION_SHA256:
        raise BoundaryError(
            f"reachable Docker instruction projection drift: sha256:{projection_sha256}"
        )
    unsupported_adds = {
        stage: adds[stage] for stage in sorted(reachable) if adds[stage]
    }
    if unsupported_adds:
        raise BoundaryError(
            f"reachable Docker stages use unsupported ADD: {unsupported_adds!r}"
        )
    unsupported_onbuilds = {
        stage: onbuilds[stage] for stage in sorted(reachable) if onbuilds[stage]
    }
    if unsupported_onbuilds:
        raise BoundaryError(
            f"reachable Docker stages use unsupported ONBUILD: {unsupported_onbuilds!r}"
        )
    mounted_runs = {
        stage: [command for _, command in runs[stage] if "--mount" in command]
        for stage in sorted(reachable)
        if any("--mount" in command for _, command in runs[stage])
    }
    if mounted_runs:
        raise BoundaryError(
            f"reachable Docker stages use unsupported RUN mounts: {mounted_runs!r}"
        )
    if runs[final_stage]:
        raise BoundaryError("the final runtime stage may not mutate its copied payload")

    tainted_copies = [
        (ordinal, stage)
        for stage in reachable
        for ordinal, source_stage, source, _ in copies[stage]
        if source_stage is None and context_contains(source)
    ]
    for first_tainted, stage in tainted_copies:
        following_runs = [
            command for ordinal, command in runs[stage] if ordinal > first_tainted
        ]
        if stage != "wheel-builder" or len(following_runs) != 1:
            raise BoundaryError(f"{stage} can mutate copied plugin test-support source")

    if stage_contains(final_stage, PurePosixPath("/"), set()):
        raise BoundaryError("final runtime container ships rust/plugin-test-support")
    return ContainerBoundary(stage="runtime", instruction_sha256=projection_sha256)


def verify_kubernetes_image_projection(
    root: Path, container_boundary: ContainerBoundary
) -> None:
    """Execute the operator's typed envelope-to-JobSet image projection."""
    operator_source = root / "aiperf-k8s-operator/src"
    sys.path.insert(0, os.fspath(operator_source))
    from aiperf_k8s_operator.contract import validate_envelope
    from aiperf_k8s_operator.reconciliation import build_jobset
    from jsonschema import Draft202012Validator

    fixture = json.loads(
        (
            root / "contracts/native-k8s/v1/fixtures/valid-one-cell-envelope.json"
        ).read_text()
    )
    expected_image_reference = (
        "registry.example.com/aiperf/runner@" + container_boundary.image_digest
    )
    fixture["imageDigest"] = container_boundary.image_digest
    fixture["imageReference"] = expected_image_reference
    envelope = validate_envelope(fixture)
    if container_boundary.stage != "runtime":
        raise BoundaryError(
            "Kubernetes image is not bound to the default runtime stage"
        )
    if (
        envelope.image_digest != container_boundary.image_digest
        or envelope.image_reference != expected_image_reference
    ):
        raise BoundaryError(
            "Kubernetes envelope is not bound to the exact container projection"
        )
    jobset = build_jobset(envelope, "http://operator.system.svc:8080", "task2-boundary")
    pod_specs = [
        job["template"]["spec"]["template"]["spec"]
        for job in jobset["spec"]["replicatedJobs"]
    ]
    containers = [container for pod in pod_specs for container in pod["containers"]]
    if not containers or any(
        container["image"] != envelope.image_reference for container in containers
    ):
        raise BoundaryError(
            "Kubernetes roles do not share the exact digest-qualified image"
        )
    if any(
        set(pod).intersection({"initContainers", "ephemeralContainers"})
        for pod in pod_specs
    ):
        raise BoundaryError("Kubernetes workload adds an untracked image payload")
    allowed_volumes = {"configMap", "emptyDir", "projected", "secret"}
    if any(
        set(volume).difference({"name"}).difference(allowed_volumes)
        for pod in pod_specs
        for volume in pod.get("volumes", [])
    ):
        raise BoundaryError("Kubernetes workload adds an untracked filesystem payload")
    capabilities = {
        "contractVersion": "native-k8s/v1",
        "imageDigest": envelope.image_digest,
        "cellular": True,
        "resultsSidecar": True,
        "hierarchicalAggregation": False,
    }
    schema = json.loads(
        (root / "contracts/native-k8s/v1/image-capabilities.schema.json").read_text()
    )
    Draft202012Validator(schema).validate(capabilities)
    if set(capabilities) != set(schema["required"]):
        raise BoundaryError(
            "Kubernetes image capabilities admit an untracked payload selector"
        )


def main() -> int:
    """Validate every current shipped boundary."""
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} REPOSITORY POLICY_TEST_EXECUTABLE", file=sys.stderr
        )
        return 64
    root = Path(sys.argv[1]).resolve()
    executable = Path(sys.argv[2]).resolve()
    try:
        verify_native_install(root)
        container_boundary = verify_final_container(root)
        verify_wheel(root, executable)
        verify_kubernetes_image_projection(root, container_boundary)
    except (BoundaryError, OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
