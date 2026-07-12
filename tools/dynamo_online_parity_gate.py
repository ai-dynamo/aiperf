# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate the AIPerf DynoSim facade against official online replay.

The two commands intentionally execute Dynamo's same canonical parser, binding,
live runtime, collector, and report writer. Separate processes are still used so
the gate measures frontend startup cost and real-clock run-to-run variance.
Deterministic report fields remain byte/ULP exact. Real-clock fields use exact,
tolerance-shifted conditional rank tests with Holm control across each case;
self-variance and paired blocks are diagnostics, never an acceptance escape.
Wall time and RSS are always gated. CPU and CPU-over-wall join the family only
when official median CPU is at least one second, where fixed process bootstrap
cost no longer dominates the measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import signal
import statistics
import struct
import subprocess
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path

EXACT_REPORT_FIELDS = frozenset(
    {
        "completed_requests",
        "decode_gpus_per_worker",
        "decode_worker_seconds",
        "gpu_hours",
        "num_requests",
        "prefill_gpus_per_worker",
        "prefill_worker_seconds",
        "processed_tokens",
        "total_input_tokens",
        "total_output_tokens",
    }
)

HOLM_FAMILY_ALPHA_NUMERATOR = 1
HOLM_FAMILY_ALPHA_DENOMINATOR = 20
PER_FIELD_TAIL_ALPHA_NUMERATOR = 1
PER_FIELD_TAIL_ALPHA_DENOMINATOR = 40

REQUIRED_POSITIVE_TAGS = frozenset(
    {
        "dispatch:authored",
        "dispatch:closed_loop",
        "engine:sglang",
        "engine:trtllm",
        "engine:vllm",
        "engine_feature:aic_timing",
        "engine_feature:kv_offload",
        "router:aic_prefill_load",
        "router:config",
        "router:kv",
        "router:model_profile",
        "router:policy_config",
        "router:round_robin",
        "source:applied_compute_agentic",
        "source:dynamo_multi_file",
        "source:mooncake",
        "source:synthetic_direct",
        "source:synthetic_multiturn",
        "workers:multi",
        "workers:single",
        "workload:inter_turn_delay",
        "workload:policy_class",
        "workload:shared_prefix",
    }
)

REQUIRED_REJECTION_TAGS = frozenset(
    {
        "reject:agentic_dynamo",
        "reject:agentic_mooncake",
        "reject:applied_compute_without_concurrency",
        "reject:disaggregated",
        "reject:max_sim_time",
        "reject:mooncake_delta",
        "reject:planner",
        "reject:report_jsonl",
        "reject:sla",
    }
)


@dataclass(frozen=True)
class Case:
    """One source-grounded online feature-basis case."""

    name: str
    arguments: tuple[str, ...]
    tags: frozenset[str]
    expected_requests: int | None = None
    expected_error: str | None = None
    required_stderr: tuple[str, ...] = ()


@dataclass(frozen=True)
class Sample:
    """One fresh child process and its kernel-accounted resources."""

    case: str
    frontend: str
    index: int
    returncode: int
    wall_s: float
    user_s: float
    system_s: float
    rss_kib: int
    report_path: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--official-pythonpath", required=True)
    parser.add_argument("--aiperf-source", type=Path, required=True)
    parser.add_argument("--dynamo-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-delta-percent", type=float, default=5.0)
    parser.add_argument("--max-deterministic-ulps", type=int, default=1)
    parser.add_argument(
        "--cases",
        help="comma-separated positive case names; default runs the complete matrix",
    )
    parser.add_argument(
        "--skip-rejections",
        action="store_true",
        help="skip the official online fail-closed matrix",
    )
    parser.add_argument(
        "--rejections-only",
        action="store_true",
        help="run only the official online fail-closed matrix",
    )
    parser.add_argument("--list-cases", action="store_true")
    return parser


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_compact_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_fixtures(
    output_dir: Path, dynamo_source: Path
) -> dict[str, Path | tuple[Path, ...]]:
    fixtures = output_dir / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)

    mooncake = fixtures / "mooncake.jsonl"
    _write_jsonl(
        mooncake,
        [
            {
                "timestamp": float(index * 4),
                "input_length": 128,
                "output_length": 32,
                "hash_ids": [index % 16],
                "policy_class": "latency" if index % 2 else "batch",
            }
            for index in range(512)
        ],
    )

    mooncake_multiturn = fixtures / "mooncake-multiturn.jsonl"
    multiturn_rows: list[dict[str, object]] = []
    for session in range(64):
        multiturn_rows.extend(
            [
                {
                    "session_id": f"session-{session}",
                    "timestamp": float(session),
                    "input_length": 128,
                    "output_length": 32,
                    "hash_ids": [session % 8],
                },
                {
                    "session_id": f"session-{session}",
                    "delay": 1.0,
                    "input_length": 128,
                    "output_length": 32,
                    "hash_ids": [session % 8],
                },
            ]
        )
    _write_jsonl(mooncake_multiturn, multiturn_rows)

    applied_compute = fixtures / "applied-compute-agentic.jsonl"
    _write_jsonl(
        applied_compute,
        [
            {
                "num_turns": 2,
                "input_prompt_length": 128,
                "assistant_response_length": [32, 32],
                "tool_call_output_length": [8, 8],
                "tool_call_latency": [0.001, 0.001],
                "final_assistant_response_length": 32,
            }
            for _ in range(128)
        ],
    )

    dynamo_paths = (fixtures / "dynamo-1.jsonl", fixtures / "dynamo-2.jsonl")
    dynamo_rows = [
        {
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": 2_000 + index,
            "request": {
                "request_id": f"request-{index}",
                "request_received_ms": 1_000 + index,
                "output_tokens": 32,
                "replay": {
                    "trace_block_size": 128,
                    "input_length": 128,
                    "input_sequence_hashes": [index % 16],
                },
            },
        }
        for index in range(512)
    ]
    _write_jsonl(dynamo_paths[0], dynamo_rows[:256])
    _write_jsonl(dynamo_paths[1], dynamo_rows[256:])

    agentic_mooncake = fixtures / "agentic-mooncake.jsonl"
    _write_jsonl(agentic_mooncake, [{}])

    agentic_dynamo = dynamo_source / "lib/bench/testdata/pi_request_trace.jsonl.gz"
    if not agentic_dynamo.is_file():
        raise FileNotFoundError(
            f"missing official agentic Dynamo trace: {agentic_dynamo}"
        )

    router_policy = fixtures / "router-policy.yaml"
    router_policy.write_text(
        """\
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: latency_all
    policy_family: latency
    cache_bucket: all
    queue_policy: fcfs
    quantum: 1
  - name: batch_all
    policy_family: batch
    cache_bucket: all
    queue_policy: wspt
    quantum: 4
models:
  parity/model:
    default_policy_family: batch
    uncached_isl_buckets:
      - min_tokens: 0
        bucket: all
    policy_classes:
      - name: latency_all
        policy_family: latency
        cache_bucket: all
        queue_policy: fcfs
        quantum: 1
      - name: batch_all
        policy_family: batch
        cache_bucket: all
        queue_policy: wspt
        quantum: 4
""",
        encoding="utf-8",
    )

    return {
        "agentic_dynamo": agentic_dynamo,
        "agentic_mooncake": agentic_mooncake,
        "applied_compute": applied_compute,
        "dynamo": dynamo_paths,
        "mooncake": mooncake,
        "mooncake_multiturn": mooncake_multiturn,
        "router_policy": router_policy,
    }


def _engine_args(engine: str) -> str:
    common: dict[str, object] = {
        "block_size": 128,
        "num_gpu_blocks": 4096,
        "speedup_ratio": 1.0,
    }
    if engine == "sglang":
        common |= {"engine_type": "sglang", "sglang": {"page_size": 128}}
    elif engine == "trtllm":
        common |= {
            "engine_type": "trtllm",
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 32,
        }
    elif engine != "vllm":
        raise ValueError(f"unknown engine {engine}")
    return _compact_json(common)


def _router_config() -> str:
    return _compact_json(
        {
            "router_event_threads": 1,
            "router_temperature": 0.0,
            "use_kv_events": True,
        }
    )


def _aic_router_config() -> str:
    return _compact_json(
        {
            "router_event_threads": 1,
            "router_prefill_load_model": "aic",
            "router_temperature": 0.0,
            "use_kv_events": True,
        }
    )


def _positive_cases(fixtures: dict[str, Path | tuple[Path, ...]]) -> list[Case]:
    mooncake = fixtures["mooncake"]
    router_policy = fixtures["router_policy"]
    applied_compute = fixtures["applied_compute"]
    dynamo_paths = fixtures["dynamo"]
    assert isinstance(mooncake, Path)
    assert isinstance(router_policy, Path)
    assert isinstance(applied_compute, Path)
    assert isinstance(dynamo_paths, tuple)

    offload_args = _compact_json(
        {
            "block_size": 64,
            "enable_g4_storage": True,
            "kv_bytes_per_token": 1,
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 8,
            "num_g2_blocks": 64,
            "num_g3_blocks": 64,
            "num_gpu_blocks": 8,
            "offload_batch_size": 4,
            "speedup_ratio": 1.0,
        }
    )
    aic_engine_args = _compact_json(
        {
            "aic_backend": "vllm",
            "aic_backend_version": "0.19.0",
            "aic_model_path": "Qwen/Qwen3-32B-FP8",
            "aic_system": "h200_sxm",
            "aic_tp_size": 1,
            "block_size": 64,
            "engine_type": "vllm",
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 16,
            "num_gpu_blocks": 4096,
            "speedup_ratio": 1.0,
        }
    )

    return [
        Case(
            name="synthetic-vllm-authored-round-robin",
            arguments=(
                "--input-tokens",
                "128",
                "--output-tokens",
                "32",
                "--request-count",
                "512",
                "--arrival-interval-ms",
                "1",
                "--arrival-speedup-ratio",
                "2",
                "--replay-mode",
                "online",
                "--num-workers",
                "1",
                "--router-mode",
                "round_robin",
                "--extra-engine-args",
                _engine_args("vllm"),
            ),
            tags=frozenset(
                {
                    "dispatch:authored",
                    "engine:vllm",
                    "router:round_robin",
                    "source:synthetic_direct",
                    "workers:single",
                }
            ),
            expected_requests=512,
        ),
        Case(
            name="synthetic-sglang-closed-loop-kv",
            arguments=(
                "--input-tokens",
                "128",
                "--output-tokens",
                "32",
                "--request-count",
                "512",
                "--replay-concurrency",
                "16",
                "--replay-mode",
                "online",
                "--num-workers",
                "2",
                "--router-mode",
                "kv_router",
                "--router-config",
                _router_config(),
                "--extra-engine-args",
                _engine_args("sglang"),
            ),
            tags=frozenset(
                {
                    "dispatch:closed_loop",
                    "engine:sglang",
                    "router:config",
                    "router:kv",
                    "source:synthetic_direct",
                    "workers:multi",
                }
            ),
            expected_requests=512,
        ),
        Case(
            name="synthetic-trtllm-multiturn-prefix",
            arguments=(
                "--input-tokens",
                "128",
                "--output-tokens",
                "32",
                "--request-count",
                "128",
                "--turns-per-session",
                "4",
                "--shared-prefix-ratio",
                "0.5",
                "--num-prefix-groups",
                "8",
                "--inter-turn-delay-ms",
                "1",
                "--replay-concurrency",
                "16",
                "--replay-mode",
                "online",
                "--num-workers",
                "2",
                "--router-mode",
                "kv_router",
                "--extra-engine-args",
                _engine_args("trtllm"),
            ),
            tags=frozenset(
                {
                    "dispatch:closed_loop",
                    "engine:trtllm",
                    "router:kv",
                    "source:synthetic_multiturn",
                    "workers:multi",
                    "workload:inter_turn_delay",
                    "workload:shared_prefix",
                }
            ),
            expected_requests=512,
        ),
        Case(
            name="mooncake-authored-kv",
            arguments=(
                str(mooncake),
                "--trace-format",
                "mooncake",
                "--trace-block-size",
                "128",
                "--arrival-speedup-ratio",
                "2",
                "--replay-mode",
                "online",
                "--num-workers",
                "2",
                "--router-mode",
                "kv_router",
                "--router-policy-config",
                str(router_policy),
                "--model-name",
                "parity/model",
                "--extra-engine-args",
                _engine_args("vllm"),
            ),
            tags=frozenset(
                {
                    "dispatch:authored",
                    "engine:vllm",
                    "router:kv",
                    "router:model_profile",
                    "router:policy_config",
                    "source:mooncake",
                    "workers:multi",
                    "workload:policy_class",
                }
            ),
            expected_requests=512,
        ),
        Case(
            name="applied-compute-agentic-closed-loop",
            arguments=(
                str(applied_compute),
                "--trace-format",
                "applied_compute_agentic",
                "--trace-block-size",
                "128",
                "--trace-shared-prefix-ratio",
                "0.5",
                "--trace-num-prefix-groups",
                "8",
                "--replay-concurrency",
                "16",
                "--replay-mode",
                "online",
                "--num-workers",
                "2",
                "--router-mode",
                "round_robin",
                "--extra-engine-args",
                _engine_args("vllm"),
            ),
            tags=frozenset(
                {
                    "dispatch:closed_loop",
                    "engine:vllm",
                    "router:round_robin",
                    "source:applied_compute_agentic",
                    "workers:multi",
                    "workload:shared_prefix",
                }
            ),
            expected_requests=384,
        ),
        Case(
            name="dynamo-multi-file-closed-loop",
            arguments=(
                *(str(path) for path in dynamo_paths),
                "--trace-format",
                "dynamo",
                "--replay-concurrency",
                "16",
                "--replay-mode",
                "online",
                "--num-workers",
                "2",
                "--router-mode",
                "round_robin",
                "--extra-engine-args",
                _engine_args("vllm"),
            ),
            tags=frozenset(
                {
                    "dispatch:closed_loop",
                    "engine:vllm",
                    "router:round_robin",
                    "source:dynamo_multi_file",
                    "workers:multi",
                }
            ),
            expected_requests=512,
        ),
        Case(
            name="synthetic-vllm-kv-offload",
            arguments=(
                "--input-tokens",
                "128",
                "--output-tokens",
                "16",
                "--request-count",
                "32",
                "--replay-concurrency",
                "4",
                "--replay-mode",
                "online",
                "--num-workers",
                "1",
                "--router-mode",
                "round_robin",
                "--extra-engine-args",
                offload_args,
            ),
            tags=frozenset(
                {
                    "dispatch:closed_loop",
                    "engine:vllm",
                    "engine_feature:kv_offload",
                    "router:round_robin",
                    "source:synthetic_direct",
                    "workers:single",
                }
            ),
            expected_requests=32,
        ),
        Case(
            name="synthetic-vllm-aic-engine-router",
            arguments=(
                "--input-tokens",
                "128",
                "--output-tokens",
                "16",
                "--request-count",
                "64",
                "--replay-concurrency",
                "8",
                "--replay-mode",
                "online",
                "--num-workers",
                "1",
                "--router-mode",
                "kv_router",
                "--router-config",
                _aic_router_config(),
                "--aic-backend",
                "vllm",
                "--aic-system",
                "h200_sxm",
                "--aic-backend-version",
                "0.19.0",
                "--aic-model-path",
                "Qwen/Qwen3-32B-FP8",
                "--aic-tp-size",
                "1",
                "--extra-engine-args",
                aic_engine_args,
            ),
            tags=frozenset(
                {
                    "dispatch:closed_loop",
                    "engine:vllm",
                    "engine_feature:aic_timing",
                    "router:aic_prefill_load",
                    "router:config",
                    "router:kv",
                    "source:synthetic_direct",
                    "workers:single",
                }
            ),
            expected_requests=64,
            required_stderr=(
                "AIC: using pure-Rust RustAicCallback (no GIL on the predict hot path)",
            ),
        ),
    ]


def _rejection_cases(
    fixtures: dict[str, Path | tuple[Path, ...]], output_dir: Path
) -> list[Case]:
    mooncake = fixtures["mooncake"]
    mooncake_multiturn = fixtures["mooncake_multiturn"]
    agentic_mooncake = fixtures["agentic_mooncake"]
    agentic_dynamo = fixtures["agentic_dynamo"]
    applied_compute = fixtures["applied_compute"]
    for value in (
        mooncake,
        mooncake_multiturn,
        agentic_mooncake,
        agentic_dynamo,
        applied_compute,
    ):
        assert isinstance(value, Path)
    base_synthetic = (
        "--input-tokens",
        "8",
        "--output-tokens",
        "2",
        "--request-count",
        "2",
        "--replay-mode",
        "online",
    )
    return [
        Case(
            "reject-disaggregated",
            (
                *base_synthetic,
                "--prefill-engine-args",
                _compact_json({"block_size": 64, "worker_type": "prefill"}),
                "--decode-engine-args",
                _compact_json({"block_size": 64, "worker_type": "decode"}),
            ),
            frozenset({"reject:disaggregated"}),
            expected_error="disagg replay only supports replay_mode='offline'",
        ),
        Case(
            "reject-mooncake-delta",
            (
                str(mooncake_multiturn),
                "--trace-format",
                "mooncake-delta",
                "--trace-block-size",
                "128",
                "--replay-mode",
                "online",
                "--extra-engine-args",
                _engine_args("vllm"),
            ),
            frozenset({"reject:mooncake_delta"}),
            expected_error="mooncake-delta trace format is not supported for online replay",
        ),
        Case(
            "reject-agentic-mooncake",
            (
                str(agentic_mooncake),
                "--trace-format",
                "agentic_mooncake",
                "--trace-block-size",
                "128",
                "--replay-mode",
                "online",
                "--extra-engine-args",
                _engine_args("vllm"),
            ),
            frozenset({"reject:agentic_mooncake"}),
            expected_error="agentic_mooncake trace format is not supported for online replay",
        ),
        Case(
            "reject-agentic-dynamo",
            (
                str(agentic_dynamo),
                "--trace-format",
                "dynamo",
                "--replay-mode",
                "online",
                "--extra-engine-args",
                _engine_args("vllm"),
            ),
            frozenset({"reject:agentic_dynamo"}),
            expected_error="agentic Dynamo request traces are not supported for online replay",
        ),
        Case(
            "reject-applied-compute-without-concurrency",
            (
                str(applied_compute),
                "--trace-format",
                "applied_compute_agentic",
                "--replay-mode",
                "online",
                "--extra-engine-args",
                _engine_args("vllm"),
            ),
            frozenset({"reject:applied_compute_without_concurrency"}),
            expected_error="requires --replay-concurrency",
        ),
        Case(
            "reject-report-jsonl",
            (*base_synthetic, "--report-jsonl", str(output_dir / "forbidden.jsonl")),
            frozenset({"reject:report_jsonl"}),
            expected_error="--report-jsonl only supports --replay-mode=offline",
        ),
        Case(
            "reject-max-sim-time",
            (
                str(mooncake),
                "--replay-mode",
                "online",
                "--max-sim-time-seconds",
                "1",
                "--extra-engine-args",
                _engine_args("vllm"),
            ),
            frozenset({"reject:max_sim_time"}),
            expected_error="max_sim_time_ms only supports replay_mode='offline'",
        ),
        Case(
            "reject-planner",
            (*base_synthetic, "--planner-config", "{}"),
            frozenset({"reject:planner"}),
            expected_error="--planner-config only supports --replay-mode=offline",
        ),
        Case(
            "reject-sla",
            (*base_synthetic, "--sla-e2e-ms", "10"),
            frozenset({"reject:sla"}),
            expected_error="goodput SLA",
        ),
    ]


def _command(python: Path, frontend: str, arguments: tuple[str, ...]) -> list[str]:
    module = ["-m", "dynamo.replay"]
    if frontend == "aiperf":
        module = ["-m", "aiperf", "dynosim", "run"]
    return [str(python), *module, *arguments]


def _surface_identity(
    python: Path,
    env: dict[str, str],
    timeout_seconds: float,
) -> dict[str, object]:
    probes = {
        "mocker_help": {
            "aiperf": [
                str(python),
                "-m",
                "aiperf",
                "dynosim",
                "mocker",
                "--help",
            ],
            "official": [str(python), "-m", "dynamo.mocker", "--help"],
        },
        "replay_help": {
            "aiperf": [
                str(python),
                "-m",
                "aiperf",
                "dynosim",
                "run",
                "--help",
            ],
            "official": [str(python), "-m", "dynamo.replay", "--help"],
        },
        "replay_unknown_option": {
            "aiperf": [
                str(python),
                "-m",
                "aiperf",
                "dynosim",
                "run",
                "--parity-unknown-option",
            ],
            "official": [
                str(python),
                "-m",
                "dynamo.replay",
                "--parity-unknown-option",
            ],
        },
    }
    results = {}
    for name, commands in probes.items():
        completed = {
            frontend: subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                env=env,
                timeout=timeout_seconds,
                check=False,
            )
            for frontend, command in commands.items()
        }
        aiperf = completed["aiperf"]
        official = completed["official"]
        if aiperf.returncode != official.returncode:
            raise AssertionError(f"{name}: CLI return codes differ")
        if aiperf.stdout != official.stdout or aiperf.stderr != official.stderr:
            raise AssertionError(f"{name}: CLI bytes differ")
        results[name] = {
            "long_options": sorted(
                option.decode("ascii")
                for option in set(re.findall(rb"--[a-z][a-z0-9-]*", official.stdout))
                if not option.endswith(b"-")
            ),
            "returncode": official.returncode,
            "stderr_bytes": len(official.stderr),
            "stderr_sha256": hashlib.sha256(official.stderr).hexdigest(),
            "stdout_bytes": len(official.stdout),
            "stdout_sha256": hashlib.sha256(official.stdout).hexdigest(),
        }
    return {
        "byte_exact": True,
        "canonical_modules": {
            "mocker": "dynamo.mocker",
            "run": "dynamo.replay",
        },
        "probes": results,
    }


def _run_child(
    case: Case,
    frontend: str,
    index: int,
    output_dir: Path,
    python: Path,
    env: dict[str, str],
    timeout_seconds: float,
    *,
    warmup: bool = False,
) -> Sample:
    prefix = "warmup-" if warmup else ""
    stem = f"{prefix}{frontend}-{index}"
    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    report = case_dir / f"{stem}.report.json"
    stdout_path = case_dir / f"{stem}.stdout"
    stderr_path = case_dir / f"{stem}.stderr"
    arguments = (*case.arguments, "--report-json", str(report))
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        started = time.perf_counter()
        process = subprocess.Popen(
            _command(python, frontend, arguments),
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
            start_new_session=True,
        )
        deadline = started + timeout_seconds
        while True:
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid:
                break
            if time.perf_counter() >= deadline:
                os.killpg(process.pid, signal.SIGKILL)
                _, status, usage = os.wait4(process.pid, 0)
                raise TimeoutError(
                    f"{case.name}/{frontend} exceeded {timeout_seconds:.1f}s"
                )
            time.sleep(0.01)
        wall_s = time.perf_counter() - started
        returncode = os.waitstatus_to_exitcode(status)
        process.returncode = returncode

    sample = Sample(
        case=case.name,
        frontend=frontend,
        index=index,
        returncode=returncode,
        wall_s=wall_s,
        user_s=usage.ru_utime,
        system_s=usage.ru_stime,
        rss_kib=usage.ru_maxrss,
        report_path=str(report),
    )
    (case_dir / f"{stem}.time.json").write_text(
        json.dumps(asdict(sample), allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sample


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _relative_delta_percent(aiperf: float, official: float) -> float:
    if aiperf == official:
        return 0.0
    if official == 0.0:
        return math.inf
    return (aiperf / official - 1.0) * 100.0


def _median_absolute(values: list[float]) -> float:
    return _median([abs(value) for value in values])


def _symmetric_delta_percent(left: float, right: float) -> float:
    """Return a direction-independent relative distance, bounded at 200%."""
    return abs(_signed_symmetric_delta_percent(left, right))


def _signed_symmetric_delta_percent(left: float, right: float) -> float:
    """Return a signed relative distance without privileging either frontend."""
    if left == right:
        return 0.0
    scale = (abs(left) + abs(right)) / 2.0
    if scale == 0.0:
        return math.inf
    return (left - right) / scale * 100.0


def _self_variance_percent(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return max(
        _symmetric_delta_percent(left, right)
        for left, right in itertools.combinations(values, 2)
    )


def _midranks_twice(values: list[Decimal]) -> list[int]:
    """Return exact doubled midranks, preserving ties without binary64 drift."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        doubled_midrank = start + 1 + end
        for position in range(start, end):
            ranks[order[position]] = doubled_midrank
        start = end
    return ranks


def _exact_rank_sum_tail(
    left: list[Decimal], right: list[Decimal], *, upper: bool
) -> tuple[int, int]:
    """Return an inclusive conditional-permutation tail for a rank-sum test."""
    if len(left) != len(right):
        raise ValueError("rank-sum samples must have the same size")
    ranks = _midranks_twice([*left, *right])
    sample_size = len(left)
    observed = sum(ranks[:sample_size])

    counts: list[dict[int, int]] = [{} for _ in range(sample_size + 1)]
    counts[0][0] = 1
    for seen, rank in enumerate(ranks, start=1):
        for selected in range(min(sample_size, seen), 0, -1):
            for rank_sum, ways in counts[selected - 1].items():
                candidate = rank_sum + rank
                counts[selected][candidate] = counts[selected].get(candidate, 0) + ways

    distribution = counts[sample_size]
    if upper:
        extreme = sum(ways for value, ways in distribution.items() if value >= observed)
    else:
        extreme = sum(ways for value, ways in distribution.items() if value <= observed)
    total = math.comb(len(ranks), sample_size)
    if sum(distribution.values()) != total:
        raise AssertionError("rank-sum permutation distribution is incomplete")
    return extreme, total


def _distribution_gate(
    left: list[float], right: list[float], limit: float
) -> dict[str, float | int | str | bool | list[float]]:
    """Test for evidence that either distribution lies outside the tolerance."""
    if len(left) != len(right):
        raise ValueError("paired distributions must have the same sample count")
    if len(left) != 9:
        raise ValueError("online parity distributions require exactly nine samples")
    if not math.isfinite(limit) or not 0.0 < limit < 100.0:
        raise ValueError("distribution tolerance must be between zero and 100 percent")
    if not all(math.isfinite(value) and value >= 0.0 for value in [*left, *right]):
        raise ValueError("online parity distributions must be finite and non-negative")

    left_median = _median(left)
    right_median = _median(right)
    median_delta = abs(_relative_delta_percent(left_median, right_median))
    left_self_variance = _self_variance_percent(left)
    right_self_variance = _self_variance_percent(right)
    volatile = max(left_self_variance, right_self_variance) > limit
    paired_deltas = [
        _relative_delta_percent(left_value, right_value)
        for left_value, right_value in zip(left, right, strict=True)
    ]
    block_size = len(paired_deltas) // 3
    block_medians = [
        _median(paired_deltas[offset : offset + block_size])
        for offset in range(0, len(paired_deltas), block_size)
    ]
    signed_block_median = _median(block_medians)
    median_absolute_block_median = _median([abs(value) for value in block_medians])
    diagnostic_block_effect = max(
        abs(signed_block_median), median_absolute_block_median
    )

    tolerance = Decimal(str(limit)) / Decimal(100)
    left_decimal = [Decimal(str(value)) for value in left]
    right_decimal = [Decimal(str(value)) for value in right]
    high_numerator, permutation_count = _exact_rank_sum_tail(
        [value / (Decimal(1) + tolerance) for value in left_decimal],
        right_decimal,
        upper=True,
    )
    low_numerator, low_permutation_count = _exact_rank_sum_tail(
        [value / (Decimal(1) - tolerance) for value in left_decimal],
        right_decimal,
        upper=False,
    )
    if low_permutation_count != permutation_count:
        raise AssertionError("rank-sum tails used different permutation spaces")
    unadjusted_per_field_passed = (
        high_numerator * PER_FIELD_TAIL_ALPHA_DENOMINATOR
        > permutation_count * PER_FIELD_TAIL_ALPHA_NUMERATOR
        and low_numerator * PER_FIELD_TAIL_ALPHA_DENOMINATOR
        > permutation_count * PER_FIELD_TAIL_ALPHA_NUMERATOR
    )
    return {
        "block_medians_percent": block_medians,
        "diagnostic_three_block_effect_percent": diagnostic_block_effect,
        "gate_method": "tolerance_shifted_exact_conditional_rank_sum",
        "high_regression_p_numerator": high_numerator,
        "high_regression_p_value": high_numerator / permutation_count,
        "left_median": left_median,
        "left_self_variance_percent": left_self_variance,
        "low_regression_p_numerator": low_numerator,
        "low_regression_p_value": low_numerator / permutation_count,
        "median_absolute_block_median_percent": median_absolute_block_median,
        "median_delta_percent": median_delta,
        "paired_signed_official_relative_deltas_percent": paired_deltas,
        "unadjusted_per_field_passed": unadjusted_per_field_passed,
        "unadjusted_per_field_tail_alpha": (
            PER_FIELD_TAIL_ALPHA_NUMERATOR / PER_FIELD_TAIL_ALPHA_DENOMINATOR
        ),
        "permutation_count": permutation_count,
        "right_median": right_median,
        "right_self_variance_percent": right_self_variance,
        "signed_block_median_percent": signed_block_median,
        "tolerance_percent": limit,
        "volatile": volatile,
    }


def _holm_rejections(
    hypotheses: list[dict[str, int | str]],
) -> list[dict[str, float | int | str]]:
    """Apply exact Holm step-down control across one case's tested directions."""
    ordered = sorted(
        hypotheses,
        key=lambda item: (
            int(item["p_numerator"]) / int(item["p_denominator"]),
            str(item["name"]),
        ),
    )
    rejections: list[dict[str, float | int | str]] = []
    family_size = len(ordered)
    for index, hypothesis in enumerate(ordered):
        numerator = int(hypothesis["p_numerator"])
        denominator = int(hypothesis["p_denominator"])
        remaining = family_size - index
        threshold_denominator = HOLM_FAMILY_ALPHA_DENOMINATOR * remaining
        reject = (
            numerator * threshold_denominator
            <= denominator * HOLM_FAMILY_ALPHA_NUMERATOR
        )
        if not reject:
            break
        rejections.append(
            {
                "holm_threshold": (HOLM_FAMILY_ALPHA_NUMERATOR / threshold_denominator),
                "name": str(hypothesis["name"]),
                "p_denominator": denominator,
                "p_numerator": numerator,
                "p_value": numerator / denominator,
            }
        )
    return rejections


def _ordered_float_bits(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    sign = 1 << 63
    mask = (1 << 64) - 1
    return (~bits & mask) if bits & sign else bits | sign


def _ulp_distance(left: float, right: float) -> int | float:
    if left == right:
        return 0
    if not math.isfinite(left) or not math.isfinite(right):
        return math.inf
    return abs(_ordered_float_bits(left) - _ordered_float_bits(right))


def _paired_resource_deltas(
    aiperf: list[Sample], official: list[Sample], attribute: str
) -> list[float]:
    official_by_index = {sample.index: sample for sample in official}
    result = []
    for sample in sorted(aiperf, key=lambda item: item.index):
        peer = official_by_index[sample.index]
        if attribute == "process_cpu_s":
            left = sample.user_s + sample.system_s
            right = peer.user_s + peer.system_s
        else:
            left = float(getattr(sample, attribute))
            right = float(getattr(peer, attribute))
        result.append(_relative_delta_percent(left, right))
    return result


def _paired_cpu_overhead_percent_of_wall(
    aiperf: list[Sample], official: list[Sample]
) -> list[float]:
    official_by_index = {sample.index: sample for sample in official}
    return [
        (
            (sample.user_s + sample.system_s)
            - (
                official_by_index[sample.index].user_s
                + official_by_index[sample.index].system_s
            )
        )
        / official_by_index[sample.index].wall_s
        * 100.0
        for sample in sorted(aiperf, key=lambda item: item.index)
    ]


def _resource_value(sample: Sample, attribute: str) -> float:
    if attribute == "process_cpu":
        return sample.user_s + sample.system_s
    if attribute == "process_cpu_overhead_of_wall":
        return (sample.user_s + sample.system_s) / sample.wall_s
    return float(getattr(sample, attribute))


def _load_report(sample: Sample) -> dict[str, int | float]:
    value = json.loads(Path(sample.report_path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{sample.report_path} is not a JSON object")
    if not all(
        isinstance(item, int | float) and not isinstance(item, bool)
        for item in value.values()
    ):
        raise AssertionError(f"{sample.report_path} contains a non-numeric field")
    if not all(math.isfinite(float(item)) for item in value.values()):
        raise AssertionError(f"{sample.report_path} contains a non-finite field")
    return value


def _report_comparison(
    case: Case,
    aiperf: list[Sample],
    official: list[Sample],
    limit: float,
    max_deterministic_ulps: int,
) -> dict[str, object]:
    reports = {
        "aiperf": {sample.index: _load_report(sample) for sample in aiperf},
        "official": {sample.index: _load_report(sample) for sample in official},
    }
    keys = [
        set(report) for frontend in reports.values() for report in frontend.values()
    ]
    if any(key_set != keys[0] for key_set in keys[1:]):
        raise AssertionError(f"{case.name}: report schemas differ")
    unknown_exact = EXACT_REPORT_FIELDS - keys[0]
    if unknown_exact:
        raise AssertionError(
            f"{case.name}: missing exact fields {sorted(unknown_exact)}"
        )

    exact_hashes: dict[str, set[str]] = {}
    exact_reports: list[tuple[str, int, dict[str, int | float]]] = []
    for frontend, frontend_reports in reports.items():
        hashes = set()
        for index, report in frontend_reports.items():
            projection = {field: report[field] for field in sorted(EXACT_REPORT_FIELDS)}
            exact_reports.append((frontend, index, projection))
            hashes.add(
                hashlib.sha256(
                    json.dumps(
                        projection,
                        allow_nan=False,
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
            )
            if case.expected_requests is not None:
                if report["num_requests"] != case.expected_requests:
                    raise AssertionError(
                        f"{case.name}: expected {case.expected_requests} requests, "
                        f"got {report['num_requests']}"
                    )
                if report["completed_requests"] != case.expected_requests:
                    raise AssertionError(
                        f"{case.name}: only {report['completed_requests']} of "
                        f"{case.expected_requests} requests completed"
                    )
        exact_hashes[frontend] = hashes
    all_exact_hashes = exact_hashes["aiperf"] | exact_hashes["official"]
    _, _, exact_reference = exact_reports[0]
    max_ulp_distance = 0
    for frontend, index, projection in exact_reports[1:]:
        for field, reference in exact_reference.items():
            candidate = projection[field]
            if type(reference) is not type(candidate):
                raise AssertionError(
                    f"{case.name}: invariant field {field} changed JSON type in "
                    f"{frontend} sample {index}"
                )
            if isinstance(reference, int):
                if reference != candidate:
                    raise AssertionError(
                        f"{case.name}: invariant integer field {field} differs"
                    )
                continue
            distance = _ulp_distance(float(reference), float(candidate))
            if not math.isfinite(distance) or distance > max_deterministic_ulps:
                raise AssertionError(
                    f"{case.name}: invariant float field {field} differs by "
                    f"{distance} ULPs (limit {max_deterministic_ulps})"
                )
            max_ulp_distance = max(max_ulp_distance, int(distance))

    variable_fields = sorted(keys[0] - EXACT_REPORT_FIELDS)
    field_deltas: dict[str, list[float]] = {}
    field_median_abs: dict[str, float] = {}
    field_gate: dict[str, dict[str, object]] = {}
    for field in variable_fields:
        indices = sorted(reports["aiperf"])
        aiperf_values = [float(reports["aiperf"][index][field]) for index in indices]
        official_values = [
            float(reports["official"][index][field]) for index in indices
        ]
        deltas = [
            _relative_delta_percent(aiperf_value, official_value)
            for aiperf_value, official_value in zip(
                aiperf_values, official_values, strict=True
            )
        ]
        aiperf_median = _median(aiperf_values)
        official_median = _median(official_values)
        gate = _distribution_gate(aiperf_values, official_values, limit)
        median_delta = float(gate["median_delta_percent"])
        field_deltas[field] = deltas
        field_median_abs[field] = median_delta
        field_gate[field] = {
            "aiperf_median": aiperf_median,
            "aiperf_range": [min(aiperf_values), max(aiperf_values)],
            "official_median": official_median,
            "official_range": [min(official_values), max(official_values)],
            **gate,
        }

    return {
        "deterministic_projection_byte_exact": len(all_exact_hashes) == 1,
        "deterministic_projection_sha256": sorted(all_exact_hashes),
        "deterministic_projection_max_ulp_distance": max_ulp_distance,
        "field_deltas_percent": field_deltas,
        "field_gate": field_gate,
        "field_median_delta_percent": field_median_abs,
        "field_count": len(keys[0]),
        "max_field_median_delta_percent": max(field_median_abs.values()),
    }


def _run_positive_case(
    case: Case,
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, object]:
    def validate_diagnostics(sample: Sample, *, warmup: bool) -> None:
        prefix = "warmup-" if warmup else ""
        stderr = (
            args.output_dir
            / case.name
            / f"{prefix}{sample.frontend}-{sample.index}.stderr"
        ).read_text(encoding="utf-8", errors="replace")
        for required in case.required_stderr:
            if required not in stderr:
                raise AssertionError(
                    f"{case.name}/{sample.frontend}: missing required diagnostic "
                    f"{required!r}"
                )

    samples: dict[str, list[Sample]] = {"aiperf": [], "official": []}
    for index in range(1, args.warmups + 1):
        order = ("official", "aiperf") if index % 2 else ("aiperf", "official")
        for frontend in order:
            sample = _run_child(
                case,
                frontend,
                index,
                args.output_dir,
                args.python,
                env,
                args.timeout_seconds,
                warmup=True,
            )
            if sample.returncode != 0:
                raise RuntimeError(
                    f"{case.name}/{frontend} warmup exited {sample.returncode}"
                )
            validate_diagnostics(sample, warmup=True)

    for index in range(1, args.samples + 1):
        order = ("aiperf", "official") if index % 2 else ("official", "aiperf")
        for frontend in order:
            sample = _run_child(
                case,
                frontend,
                index,
                args.output_dir,
                args.python,
                env,
                args.timeout_seconds,
            )
            if sample.returncode != 0:
                raise RuntimeError(
                    f"{case.name}/{frontend} sample {index} exited {sample.returncode}"
                )
            validate_diagnostics(sample, warmup=False)
            samples[frontend].append(sample)

    resource_deltas = {
        "process_cpu": _paired_resource_deltas(
            samples["aiperf"], samples["official"], "process_cpu_s"
        ),
        "process_cpu_overhead_of_wall": _paired_cpu_overhead_percent_of_wall(
            samples["aiperf"], samples["official"]
        ),
        "rss": _paired_resource_deltas(
            samples["aiperf"], samples["official"], "rss_kib"
        ),
        "wall": _paired_resource_deltas(
            samples["aiperf"], samples["official"], "wall_s"
        ),
    }
    resource_median_abs = {
        field: _median_absolute(deltas) for field, deltas in resource_deltas.items()
    }
    resource_attributes = {
        "process_cpu": "process_cpu",
        "process_cpu_overhead_of_wall": "process_cpu_overhead_of_wall",
        "rss": "rss_kib",
        "wall": "wall_s",
    }
    resource_gate = {
        field: _distribution_gate(
            [_resource_value(sample, attribute) for sample in samples["aiperf"]],
            [_resource_value(sample, attribute) for sample in samples["official"]],
            args.max_delta_percent,
        )
        for field, attribute in resource_attributes.items()
    }
    resource_diagnostic_three_block_effect = {
        field: float(gate["diagnostic_three_block_effect_percent"])
        for field, gate in resource_gate.items()
    }
    gated_resources = ("rss", "wall")
    official_cpu_median = _median(
        [sample.user_s + sample.system_s for sample in samples["official"]]
    )
    if official_cpu_median >= 1.0:
        gated_resources = (
            *gated_resources,
            "process_cpu",
            "process_cpu_overhead_of_wall",
        )
    report = _report_comparison(
        case,
        samples["aiperf"],
        samples["official"],
        args.max_delta_percent,
        args.max_deterministic_ulps,
    )
    hypotheses: list[dict[str, int | str]] = []
    report_field_gate = report["field_gate"]
    assert isinstance(report_field_gate, dict)
    tested_gates = [
        *((f"report/{field}", gate) for field, gate in report_field_gate.items()),
        *((f"resource/{field}", resource_gate[field]) for field in gated_resources),
    ]
    for name, gate in tested_gates:
        assert isinstance(gate, dict)
        denominator = int(gate["permutation_count"])
        hypotheses.extend(
            [
                {
                    "name": f"{name}/high",
                    "p_denominator": denominator,
                    "p_numerator": int(gate["high_regression_p_numerator"]),
                },
                {
                    "name": f"{name}/low",
                    "p_denominator": denominator,
                    "p_numerator": int(gate["low_regression_p_numerator"]),
                },
            ]
        )
    rejections = _holm_rejections(hypotheses)
    failures = [
        f"{rejection['name']}: exact p={rejection['p_value']:.8f} <= "
        f"Holm threshold {rejection['holm_threshold']:.8f}"
        for rejection in rejections
    ]
    report_rejections = [
        rejection
        for rejection in rejections
        if str(rejection["name"]).startswith("report/")
    ]
    resource_rejections = [
        rejection
        for rejection in rejections
        if str(rejection["name"]).startswith("resource/")
    ]
    report["holm_rejections"] = report_rejections
    report["passed"] = not report_rejections
    return {
        "case_family_gate": {
            "family_alpha": (
                HOLM_FAMILY_ALPHA_NUMERATOR / HOLM_FAMILY_ALPHA_DENOMINATOR
            ),
            "hypothesis_count": len(hypotheses),
            "method": "exact_holm_step_down",
            "passed": not rejections,
            "rejections": rejections,
        },
        "failures": failures,
        "official_process_cpu_median_s": official_cpu_median,
        "passed": not rejections,
        "report": report,
        "resource_deltas_percent": resource_deltas,
        "resource_diagnostic_three_block_effect_percent": (
            resource_diagnostic_three_block_effect
        ),
        "resource_gate": resource_gate,
        "resource_holm_rejections": resource_rejections,
        "resource_median_abs_delta_percent": resource_median_abs,
        "required_stderr": list(case.required_stderr),
        "samples": {
            frontend: [asdict(sample) for sample in frontend_samples]
            for frontend, frontend_samples in samples.items()
        },
        "tags": sorted(case.tags),
    }


def _run_rejection_case(
    case: Case, args: argparse.Namespace, env: dict[str, str]
) -> dict[str, object]:
    assert case.expected_error is not None
    samples = {}
    outputs = {}
    for frontend in ("official", "aiperf"):
        sample = _run_child(
            case,
            frontend,
            1,
            args.output_dir,
            args.python,
            env,
            args.timeout_seconds,
        )
        if sample.returncode == 0:
            raise AssertionError(f"{case.name}/{frontend} unexpectedly succeeded")
        stderr = (args.output_dir / case.name / f"{frontend}-1.stderr").read_text(
            encoding="utf-8", errors="replace"
        )
        if case.expected_error not in stderr:
            raise AssertionError(
                f"{case.name}/{frontend} omitted expected error {case.expected_error!r}"
            )
        stdout_bytes = (
            args.output_dir / case.name / f"{frontend}-1.stdout"
        ).read_bytes()
        stderr_bytes = (
            args.output_dir / case.name / f"{frontend}-1.stderr"
        ).read_bytes()
        outputs[frontend] = (stdout_bytes, stderr_bytes)
        samples[frontend] = asdict(sample)
    if samples["official"]["returncode"] != samples["aiperf"]["returncode"]:
        raise AssertionError(f"{case.name}: rejection return codes differ")
    if outputs["official"] != outputs["aiperf"]:
        raise AssertionError(f"{case.name}: rejection output bytes differ")
    return {
        "byte_exact": True,
        "expected_error": case.expected_error,
        "stderr_sha256": hashlib.sha256(outputs["official"][1]).hexdigest(),
        "stdout_sha256": hashlib.sha256(outputs["official"][0]).hexdigest(),
        "samples": samples,
        "tags": sorted(case.tags),
    }


def _validate_matrix(positive: list[Case], rejection: list[Case]) -> None:
    positive_tags = frozenset().union(*(case.tags for case in positive))
    rejection_tags = frozenset().union(*(case.tags for case in rejection))
    missing_positive = REQUIRED_POSITIVE_TAGS - positive_tags
    missing_rejection = REQUIRED_REJECTION_TAGS - rejection_tags
    if missing_positive or missing_rejection:
        raise AssertionError(
            "online parity matrix is incomplete: "
            f"positive={sorted(missing_positive)}, rejection={sorted(missing_rejection)}"
        )


def _module_available(
    python: Path,
    env: dict[str, str],
    module_name: str,
    timeout_seconds: float,
) -> bool:
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.util,sys; "
            "raise SystemExit(importlib.util.find_spec(sys.argv[1]) is None)",
            module_name,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        timeout=timeout_seconds,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    args = _parser().parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if not args.rejections_only and args.samples != 9:
        raise ValueError("positive online parity requires exactly nine samples")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be finite and positive")
    if (
        not math.isfinite(args.max_delta_percent)
        or not 0.0 < args.max_delta_percent < 100.0
    ):
        raise ValueError("--max-delta-percent must be finite and between zero and 100")
    if args.max_deterministic_ulps < 0:
        raise ValueError("--max-deterministic-ulps must be non-negative")
    if args.skip_rejections and args.rejections_only:
        raise ValueError(
            "--skip-rejections and --rejections-only are mutually exclusive"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = _write_fixtures(args.output_dir, args.dynamo_source)
    positive = _positive_cases(fixtures)
    rejection = _rejection_cases(fixtures, args.output_dir)
    _validate_matrix(positive, rejection)

    if args.list_cases:
        for case in [*positive, *rejection]:
            print(case.name)
        return 0

    if args.cases:
        selected_names = {
            name.strip() for name in args.cases.split(",") if name.strip()
        }
        known = {case.name for case in positive}
        unknown = selected_names - known
        if unknown:
            raise ValueError(f"unknown positive cases: {sorted(unknown)}")
        positive = [case for case in positive if case.name in selected_names]
    if args.rejections_only:
        positive = []

    pythonpath = os.pathsep.join([str(args.aiperf_source), args.official_pythonpath])
    env = os.environ.copy() | {
        "MPLCONFIGDIR": str(args.output_dir / "matplotlib"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": pythonpath,
    }

    summary: dict[str, object] = {
        "available_optional_features": {
            "aiconfigurator": _module_available(
                args.python,
                env,
                "aiconfigurator",
                args.timeout_seconds,
            ),
        },
        "coverage": {
            "positive_tags": sorted(REQUIRED_POSITIVE_TAGS),
            "rejection_tags": sorted(REQUIRED_REJECTION_TAGS),
        },
        "max_delta_percent": args.max_delta_percent,
        "max_deterministic_ulps": args.max_deterministic_ulps,
        "positive_cases": {},
        "rejection_cases": {},
        "samples": args.samples,
        "warmups": args.warmups,
    }
    print("checking byte-exact canonical CLI surface", flush=True)
    summary["cli_surface"] = _surface_identity(
        args.python,
        env,
        args.timeout_seconds,
    )
    positive_results = summary["positive_cases"]
    assert isinstance(positive_results, dict)
    for case in positive:
        print(f"running positive case: {case.name}", flush=True)
        positive_results[case.name] = _run_positive_case(case, args, env)
        (args.output_dir / "summary.partial.json").write_text(
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if not args.skip_rejections:
        rejection_results = summary["rejection_cases"]
        assert isinstance(rejection_results, dict)
        for case in rejection:
            print(f"running rejection case: {case.name}", flush=True)
            rejection_results[case.name] = _run_rejection_case(case, args, env)

    summary["passed"] = all(
        result.get("passed") is True for result in positive_results.values()
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
