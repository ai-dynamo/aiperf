# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline parity gate: `aiperf profile` vs `python -m dynamo.replay`.

Drives the real AIPerf product path — `aiperf profile --config <dynosim YAML>`
spawning the dynosim `aiperf` — against the official Dynamo offline replay
frontend, across the full supported dynosim feature matrix (topology × router ×
worker counts), and checks the two backend Dynamo reports agree.

Two comparison modes (per `Case.tolerance`):

* **Byte-exact** — the deterministic scenarios (single, aggregated any-router,
  disaggregated round-robin, disaggregated KV with a single decode worker). Both
  reports are canonicalized to Dynamo's on-disk form — every float rounded to 6
  decimals, then `json.dumps(obj, indent=2, sort_keys=True) + "\\n"` (exactly
  what `dynamo.replay` writes; only the Rust runner's excess f64 precision is
  stripped) — and SHA-256 compared. These are stable byte-for-byte run-to-run.

* **Tolerance** — disaggregated KV with a worker *fleet*. These are inherently
  non-deterministic in `dynamo.replay` itself: the KV router's greedy tie-break
  uses entropy `rand::rng()`, so tied requests route to different workers
  run-to-run, reshaping per-worker batch composition. `dynamo.replay` does not
  reproduce itself here, so byte-exactness is impossible. Instead: counts /
  token accounting / worker parallelism must match **exactly**; central metrics
  (means, medians, throughput, duration) must agree within `Case.tolerance`; and
  the inherently-unbounded distribution tails (`min_*`/`max_*`/`std_*`/high
  percentiles) are not gated.

Runs the whole matrix and exits non-zero if any case fails its check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Engine args authored identically on both sides (shared by prefill+decode in
# the disaggregated cases). Mirrors the canonical dynosim template so the gate
# exercises the prefix-cache/batching path rather than bare defaults.
ENGINE_ARGS: dict[str, object] = {
    "block_size": 16,
    "max_num_batched_tokens": 8192,
    "enable_prefix_caching": True,
}

# The kitchen-sink engine: every deterministic knob both frontends pass straight
# through to MockEngineArgs, loaded up to prove parity holds under a maximal
# configuration (data_parallel_size stays 1 — the offline path requires it).
RIDICULOUS_ENGINE: dict[str, object] = {
    "engine_type": "vllm",
    "block_size": 16,
    "num_gpu_blocks": 8192,
    "max_num_seqs": 512,
    "max_num_batched_tokens": 16384,
    "max_model_len": 32768,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    "speedup_ratio": 1.5,
    "decode_speedup_ratio": 2.0,
    "startup_time": 0.5,
    "gpu_memory_utilization": 0.9,
    "enable_local_indexer": True,
    "handoff_session_timeout_ms": 300000,
    "kv_transfer_timing_mode": "full_prompt",
    "preemption_mode": "lifo",
}


@dataclass(frozen=True)
class Case:
    """One point in the dynosim feature matrix.

    `router` is the aiperf spelling (`round_robin`/`kv`); it maps to
    `dynamo.replay`'s `round_robin`/`kv_router`. Aggregated/single use `workers`;
    disaggregated uses `prefill_workers`/`decode_workers`.
    """

    name: str
    topology: str  # single | aggregated | disaggregated
    router: str  # round_robin | kv
    workers: int = 1
    prefill_workers: int = 1
    decode_workers: int = 1
    engine: dict[str, object] | None = None  # None -> ENGINE_ARGS
    # None => byte-exact (deterministic scenario). A float => tolerance compare
    # for scenarios inherently non-deterministic in dynamo.replay itself (KV
    # router entropy tie-break under a worker fleet). Counts/parallelism stay
    # exact; central metrics must agree within this relative tolerance;
    # distribution tails are not gated. See the module docstring and
    # `_compare_tolerant`.
    tolerance: float | None = None

    @property
    def engine_args(self) -> dict[str, object]:
        return self.engine if self.engine is not None else ENGINE_ARGS

    @property
    def is_disagg(self) -> bool:
        return self.topology == "disaggregated"

    @property
    def replay_router_mode(self) -> str:
        return "kv_router" if self.router == "kv" else "round_robin"


# The full supported offline matrix. Online (wall-clock) is intentionally
# excluded: it is non-deterministic and cannot be byte-exact.
CASES: list[Case] = [
    Case("single-rr", "single", "round_robin", workers=1),
    Case("aggregated-rr-w1", "aggregated", "round_robin", workers=1),
    Case("aggregated-rr-w2", "aggregated", "round_robin", workers=2),
    # KV routing is only meaningful with >1 worker (offline replay rejects
    # kv_router at num_workers=1 — nothing to route).
    Case("aggregated-kv-w2", "aggregated", "kv", workers=2),
    Case("aggregated-kv-w4", "aggregated", "kv", workers=4),
    Case("disagg-rr-p1d1", "disaggregated", "round_robin", prefill_workers=1, decode_workers=1),
    Case("disagg-rr-p1d2", "disaggregated", "round_robin", prefill_workers=1, decode_workers=2),
    Case("disagg-rr-p2d1", "disaggregated", "round_robin", prefill_workers=2, decode_workers=1),
    Case("disagg-kv-p1d1", "disaggregated", "kv", prefill_workers=1, decode_workers=1),
    # Disaggregated KV with a worker fleet is inherently non-deterministic in
    # dynamo.replay itself (entropy tie-break; see Case.tolerance). Compared with
    # a relative tolerance rather than byte-exact — counts stay exact, timing is
    # within the reference's own run-to-run spread.
    Case("disagg-kv-p2d1", "disaggregated", "kv", prefill_workers=2, decode_workers=1,
         tolerance=0.15),
    Case("disagg-kv-p1d2", "disaggregated", "kv", prefill_workers=1, decode_workers=2,
         tolerance=0.15),
    # The most ridiculous setup: disaggregated, KV-routed across a prefill AND
    # decode fleet, with every deterministic engine knob loaded. Tolerance-checked
    # (KV-fleet non-determinism), counts exact.
    Case("kitchen-sink", "disaggregated", "kv", prefill_workers=3, decode_workers=4,
         engine=RIDICULOUS_ENGINE, tolerance=0.15),
]


@dataclass
class Outcome:
    case: str
    ok: bool
    sha256: str = ""
    detail: str = ""
    mode: str = "byte-exact"
    max_rel: float = 0.0  # worst relative delta seen (tolerance mode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiperf", type=Path, required=True,
                        help="AIPerf console entrypoint (e.g. .venv/bin/aiperf).")
    parser.add_argument("--runner-bin", type=Path, required=True,
                        help="dynosim aiperf runner (exported as AIPERF_RUNNER_BIN).")
    parser.add_argument("--python", type=Path, required=True,
                        help="Python with the dynamo bindings importable.")
    parser.add_argument("--official-pythonpath", required=True,
                        help="PYTHONPATH for `python -m dynamo.replay` (bindings + components).")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-concurrency", type=int, default=64)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--cases", default="",
                        help="Comma-separated case names to run (default: the full matrix).")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop at the first non-byte-exact case (default: run all).")
    return parser


def _aiperf_config(case: Case, requests: int, concurrency: int, trace: Path) -> str:
    transport: dict[str, object] = {
        "type": "dynosim_offline",
        "topology": case.topology,
        "routerMode": case.router,
        "engine": case.engine_args,
        "artifacts": {
            "reportJson": "dynamo_report.json",
            "perRequestJsonl": "dynamo_per_request.jsonl",
        },
    }
    if case.is_disagg:
        transport["prefillWorkers"] = case.prefill_workers
        transport["decodeWorkers"] = case.decode_workers
    else:
        transport["workers"] = case.workers
    config = {
        "schemaVersion": "2.0",
        "randomSeed": 42,
        "benchmark": {
            "model": "mock-model",
            "tokenizer": {"name": "builtin"},
            "endpoint": {"type": "dynosim"},
            "transport": transport,
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


def _run_aiperf(args: argparse.Namespace, case: Case) -> Path:
    workdir = args.output_dir / f"aiperf-{case.name}"
    workdir.mkdir(parents=True, exist_ok=True)
    config_path = workdir / "config.yaml"
    config_path.write_text(
        _aiperf_config(case, args.requests, args.replay_concurrency, args.trace),
        encoding="utf-8",
    )
    artifact_dir = workdir / "artifacts"
    env = os.environ | {"AIPERF_RUNNER_BIN": str(args.runner_bin), "HF_HUB_OFFLINE": "1"}
    with (workdir / "stdout").open("wb") as out, (workdir / "stderr").open("wb") as err:
        subprocess.run(
            [str(args.aiperf), "profile", "--config", str(config_path),
             "--artifact-dir", str(artifact_dir), "--export-level", "summary"],
            check=True, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env,
        )
    report = artifact_dir / "dynamo_report.json"
    if not report.is_file():
        raise RuntimeError(f"aiperf {case.name}: missing backend report {report}")
    return report


def _run_replay(args: argparse.Namespace, case: Case) -> Path:
    workdir = args.output_dir / f"replay-{case.name}"
    workdir.mkdir(parents=True, exist_ok=True)
    report = workdir / "replay_report.json"
    command = [
        str(args.python), "-m", "dynamo.replay", str(args.trace),
        "--trace-format", "mooncake",
        "--replay-concurrency", str(args.replay_concurrency),
        "--replay-mode", "offline",
        "--router-mode", case.replay_router_mode,
    ]
    if case.is_disagg:
        # Presence of prefill+decode engine args selects the disaggregated
        # topology; each side must carry its `worker_type` (aiperf sets these
        # internally from the topology; the replay CLI requires them explicit).
        command += [
            "--num-prefill-workers", str(case.prefill_workers),
            "--num-decode-workers", str(case.decode_workers),
            "--prefill-engine-args", json.dumps({**case.engine_args, "worker_type": "prefill"}),
            "--decode-engine-args", json.dumps({**case.engine_args, "worker_type": "decode"}),
        ]
    else:
        command += ["--num-workers", str(case.workers),
                    "--extra-engine-args", json.dumps(case.engine_args)]
    command += ["--report-json", str(report)]
    env = os.environ | {"PYTHONNOUSERSITE": "1", "PYTHONPATH": args.official_pythonpath}
    with (workdir / "stdout").open("wb") as out, (workdir / "stderr").open("wb") as err:
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL,
                       stdout=out, stderr=err, env=env)
    if not report.is_file():
        raise RuntimeError(f"dynamo.replay {case.name}: missing report {report}")
    return report


def _canonical_bytes(path: Path) -> bytes:
    """Reduce a report to Dynamo's on-disk form: 6-dp floats, sorted, indent 2."""
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


def _compare(case: Case, aiperf_report: Path, replay_report: Path) -> Outcome:
    if case.tolerance is not None:
        return _compare_tolerant(case, aiperf_report, replay_report)
    aiperf_bytes = _canonical_bytes(aiperf_report)
    replay_bytes = _canonical_bytes(replay_report)
    if aiperf_bytes == replay_bytes:
        return Outcome(case.name, True, hashlib.sha256(aiperf_bytes).hexdigest())
    import difflib

    diff = "\n".join(
        difflib.unified_diff(
            aiperf_bytes.decode().splitlines(),
            replay_bytes.decode().splitlines(),
            fromfile=f"aiperf/{case.name}", tofile=f"replay/{case.name}", lineterm="",
        )
    )
    return Outcome(case.name, False, detail=diff)


def _compare_tolerant(case: Case, aiperf_report: Path, replay_report: Path) -> Outcome:
    """Relative-tolerance compare for the inherently non-deterministic KV-fleet
    scenarios: every non-float field (counts, modes, worker parallelism) must
    match exactly; float fields must agree within `case.tolerance` relative
    (with a small absolute floor so genuine zeros / sub-milli values pass)."""
    tol = case.tolerance
    assert tol is not None

    def _is_distribution_tail(name: str) -> bool:
        # Distribution tails / spread are dominated by single random routings and
        # are genuinely unbounded run-to-run under the KV-fleet tie-break (min/max
        # can swing 30%+, std 20%+). Not meaningful to gate against one reference
        # run; the central tendencies below carry the parity signal.
        return (
            name.startswith(("std_", "max_", "min_"))
            or any(p in name for p in ("p75", "p90", "p95", "p99"))
            or name == "wall_time_ms"  # real elapsed wall-clock, not a sim metric
        )

    aiperf = json.loads(aiperf_report.read_text())
    replay = json.loads(replay_report.read_text())
    exact_violations: list[str] = []
    tol_violations: list[str] = []
    worst_rel = 0.0
    for key in sorted(set(aiperf) & set(replay)):
        av, bv = aiperf[key], replay[key]
        if isinstance(av, bool) or isinstance(bv, bool) or isinstance(av, str):
            if av != bv:
                exact_violations.append(f"  {key}: {av!r} != {bv!r} (exact)")
        elif isinstance(av, int) and isinstance(bv, int):
            # Counts / worker parallelism are deterministic invariants — exact.
            if av != bv:
                exact_violations.append(f"  {key}: {av} != {bv} (int, must be exact)")
        elif isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            if _is_distribution_tail(key):
                continue  # inherently non-deterministic; not gated
            rel = abs(av - bv) / max(abs(bv), 1.0)  # absolute floor of 1.0 unit
            if rel > tol:
                tol_violations.append(f"  {key}: {av} vs {bv} rel={rel:.5f} > {tol}")
            worst_rel = max(worst_rel, rel)
    only_a = sorted(set(aiperf) - set(replay))
    only_b = sorted(set(replay) - set(aiperf))
    if only_a or only_b:
        exact_violations.append(f"  key mismatch: only-aiperf={only_a} only-replay={only_b}")
    ok = not exact_violations and not tol_violations
    detail = ""
    if not ok:
        lines = [f"[{case.name}] central-metric tol={tol:.2f} FAILED (tails/spread not gated):"]
        lines += exact_violations
        lines += tol_violations
        detail = "\n".join(lines)
    return Outcome(
        case.name, ok, detail=detail, mode=f"tolerance≤{tol:.2f}", max_rel=worst_rel
    )


def main() -> int:
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = {name.strip() for name in args.cases.split(",") if name.strip()}
    cases = [case for case in CASES if not selected or case.name in selected]
    unknown = selected - {case.name for case in CASES}
    if unknown:
        raise ValueError(f"unknown cases {sorted(unknown)}; known: {[c.name for c in CASES]}")

    outcomes: list[Outcome] = []
    for case in cases:
        try:
            aiperf_report = _run_aiperf(args, case)
            replay_report = _run_replay(args, case)
            outcome = _compare(case, aiperf_report, replay_report)
        except Exception as error:  # noqa: BLE001 - report, don't abort the matrix
            outcome = Outcome(case.name, False, detail=f"{type(error).__name__}: {error}")
        outcomes.append(outcome)
        if outcome.ok and outcome.mode == "byte-exact":
            print(f"[{outcome.case:<18}] BYTE-EXACT   sha256={outcome.sha256}")
        elif outcome.ok:
            print(f"[{outcome.case:<18}] PARITY-OK    {outcome.mode} (worst rel={outcome.max_rel:.5f})")
        else:
            print(f"[{outcome.case:<18}] DIVERGED     ({outcome.mode})")
            print(outcome.detail)
            if args.fail_fast:
                break

    passed = [o for o in outcomes if o.ok]
    failed = [o for o in outcomes if not o.ok]
    summary = {
        "all_pass": not failed,
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "passed": [o.case for o in passed],
        "failed": [o.case for o in failed],
        "cases": {
            o.case: {
                "pass": o.ok,
                "mode": o.mode,
                **({"canonical_sha256": o.sha256} if o.mode == "byte-exact" else {"worst_rel": o.max_rel}),
            }
            for o in outcomes
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    exact = sum(1 for o in passed if o.mode == "byte-exact")
    tol = len(passed) - exact
    print(f"\n{len(passed)}/{len(outcomes)} pass ({exact} byte-exact, {tol} within-tolerance)"
          + (f"; FAILED: {[o.case for o in failed]}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
