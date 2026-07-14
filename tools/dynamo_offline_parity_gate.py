# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-exact offline parity gate: `aiperf profile` vs `python -m dynamo.replay`.

This drives the real AIPerf product path — `aiperf profile --config <dynosim
YAML>` spawning the dynosim-featured `aiperf-runner` — against the official
Dynamo offline replay frontend, for each requested topology, and asserts the
two backend Dynamo reports are **byte-identical**.

The two frontends emit the same report values but serialize floats differently
(the Rust runner writes full f64 precision; `dynamo.replay` rounds to 6 decimals
in `write_report_json`). Both reports are therefore canonicalized to Dynamo's
own on-disk form — every float rounded to 6 decimals, then
`json.dumps(obj, indent=2, sort_keys=True) + "\\n"` — before the SHA-256 compare.
That canonical form is exactly the bytes `dynamo.replay` already writes, so a
match is byte-for-byte parity against the reference report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

# aiperf `transport.topology` -> the matching `dynamo.replay` worker wiring.
# Both frontends read the same trace and default engine, so a topology maps to
# one replay invocation; single and aggregated both provision one decode worker.
TOPOLOGIES: dict[str, dict[str, object]] = {
    "single": {"workers": 1, "router_mode": "round_robin"},
    "aggregated": {"workers": 1, "router_mode": "round_robin"},
}

# Engine args authored identically on both sides. Empty-object defaults also
# work; these mirror the canonical dynosim template so the gate exercises the
# prefix-cache/batching path rather than bare defaults.
ENGINE_ARGS: dict[str, object] = {
    "block_size": 16,
    "max_num_batched_tokens": 8192,
    "enable_prefix_caching": True,
}


@dataclass(frozen=True)
class TopologyResult:
    topology: str
    aiperf_report: str
    replay_report: str
    sha256: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aiperf",
        type=Path,
        required=True,
        help="AIPerf console entrypoint (e.g. .venv/bin/aiperf).",
    )
    parser.add_argument(
        "--runner-bin",
        type=Path,
        required=True,
        help="dynosim-featured aiperf-runner binary (exported as AIPERF_RUNNER_BIN).",
    )
    parser.add_argument(
        "--python",
        type=Path,
        required=True,
        help="Python interpreter with the dynamo bindings importable.",
    )
    parser.add_argument(
        "--official-pythonpath",
        required=True,
        help="PYTHONPATH for `python -m dynamo.replay` (bindings + components).",
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-concurrency", type=int, default=64)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument(
        "--topologies",
        default="single,aggregated",
        help="Comma-separated topologies to gate (default: single,aggregated).",
    )
    return parser


def _aiperf_config(topology: str, workers: int, router_mode: str, requests: int,
                   concurrency: int, trace: Path) -> str:
    """Author a dynosim_offline `aiperf profile` config for one topology."""
    config = {
        "schemaVersion": "2.0",
        "randomSeed": 42,
        "benchmark": {
            "model": "mock-model",
            "tokenizer": {"name": "builtin"},
            "endpoint": {"type": "dynosim"},
            "transport": {
                "type": "dynosim_offline",
                "topology": topology,
                "routerMode": router_mode,
                "workers": workers,
                "engine": ENGINE_ARGS,
                "artifacts": {
                    "reportJson": "dynamo_report.json",
                    "perRequestJsonl": "dynamo_per_request.jsonl",
                },
            },
            "dataset": {
                "type": "file",
                "path": str(trace),
                "format": "mooncake_trace",
                "sampling": "sequential",
            },
            "phases": {
                "type": "concurrency",
                "requests": requests,
                "concurrency": concurrency,
            },
        },
    }
    return json.dumps(config, indent=2)


def _run_aiperf(args: argparse.Namespace, topology: str, spec: dict[str, object]) -> Path:
    """Run `aiperf profile` for one topology; return its backend report path."""
    workdir = args.output_dir / f"aiperf-{topology}"
    workdir.mkdir(parents=True, exist_ok=True)
    config_path = workdir / "config.yaml"
    config_path.write_text(
        _aiperf_config(
            topology,
            int(spec["workers"]),
            str(spec["router_mode"]),
            args.requests,
            args.replay_concurrency,
            args.trace,
        ),
        encoding="utf-8",
    )
    artifact_dir = workdir / "artifacts"
    env = os.environ | {
        "AIPERF_RUNNER_BIN": str(args.runner_bin),
        "HF_HUB_OFFLINE": "1",
    }
    with (workdir / "stdout").open("wb") as out, (workdir / "stderr").open("wb") as err:
        subprocess.run(
            [
                str(args.aiperf),
                "profile",
                "--config",
                str(config_path),
                "--artifact-dir",
                str(artifact_dir),
                "--export-level",
                "summary",
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            env=env,
        )
    report = artifact_dir / "dynamo_report.json"
    if not report.is_file():
        raise RuntimeError(f"aiperf {topology}: missing backend report {report}")
    return report


def _run_replay(args: argparse.Namespace, topology: str, spec: dict[str, object]) -> Path:
    """Run `python -m dynamo.replay` mirroring one topology; return report path."""
    workdir = args.output_dir / f"replay-{topology}"
    workdir.mkdir(parents=True, exist_ok=True)
    report = workdir / "replay_report.json"
    env = os.environ | {
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": args.official_pythonpath,
    }
    with (workdir / "stdout").open("wb") as out, (workdir / "stderr").open("wb") as err:
        subprocess.run(
            [
                str(args.python),
                "-m",
                "dynamo.replay",
                str(args.trace),
                "--trace-format",
                "mooncake",
                "--replay-concurrency",
                str(args.replay_concurrency),
                "--replay-mode",
                "offline",
                "--num-workers",
                str(int(spec["workers"])),
                "--router-mode",
                str(spec["router_mode"]),
                "--extra-engine-args",
                json.dumps(ENGINE_ARGS),
                "--report-json",
                str(report),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            env=env,
        )
    if not report.is_file():
        raise RuntimeError(f"dynamo.replay {topology}: missing report {report}")
    return report


def _canonical_bytes(path: Path) -> bytes:
    """Reduce a report to Dynamo's on-disk form: 6-dp floats, sorted, indent 2.

    This is exactly the serialization `dynamo.replay.write_report_json` emits,
    so comparing canonical bytes is a byte-for-byte match against the reference
    report — it only strips the Rust runner's excess float precision, never a
    value difference (a genuine divergence survives the round()).
    """
    obj = json.loads(path.read_text(encoding="utf-8"))

    def reduce(value: object) -> object:
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, dict):
            return {key: reduce(item) for key, item in value.items()}
        if isinstance(value, list):
            return [reduce(item) for item in value]
        return value

    return (json.dumps(reduce(obj), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _assert_byte_exact(topology: str, aiperf_report: Path, replay_report: Path) -> str:
    aiperf_bytes = _canonical_bytes(aiperf_report)
    replay_bytes = _canonical_bytes(replay_report)
    if aiperf_bytes != replay_bytes:
        aiperf_lines = aiperf_bytes.decode().splitlines()
        replay_lines = replay_bytes.decode().splitlines()
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                aiperf_lines,
                replay_lines,
                fromfile=f"aiperf/{topology}",
                tofile=f"replay/{topology}",
                lineterm="",
            )
        )
        raise AssertionError(
            f"[{topology}] backend reports are NOT byte-exact:\n{diff}"
        )
    return hashlib.sha256(aiperf_bytes).hexdigest()


def main() -> int:
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topologies = [name.strip() for name in args.topologies.split(",") if name.strip()]
    unknown = [name for name in topologies if name not in TOPOLOGIES]
    if unknown:
        raise ValueError(f"unknown topologies {unknown}; known: {sorted(TOPOLOGIES)}")

    results: list[TopologyResult] = []
    for topology in topologies:
        spec = TOPOLOGIES[topology]
        aiperf_report = _run_aiperf(args, topology, spec)
        replay_report = _run_replay(args, topology, spec)
        sha = _assert_byte_exact(topology, aiperf_report, replay_report)
        results.append(
            TopologyResult(topology, str(aiperf_report), str(replay_report), sha)
        )
        print(f"[{topology}] BYTE-EXACT ok  sha256={sha}")

    summary = {
        "byte_exact": True,
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "topologies": {
            result.topology: {
                "aiperf_report": result.aiperf_report,
                "replay_report": result.replay_report,
                "canonical_sha256": result.sha256,
            }
            for result in results
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
