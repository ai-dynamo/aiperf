#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-ramp AIPerfJob concurrency: when a job's send phase completes, kill it
and submit the next one at `--step` more concurrency (requests stay 3× the
concurrency target). Stops when concurrency would exceed `--max-concurrency`,
when a submitted job ends in Failed, or on Ctrl-C.

Used by the persistent Monitor ramp loop during cluster-durability ramps.
Intentionally stdout-heavy — every state change prints one line so the outer
Monitor surfaces it as a notification.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = PROJECT_ROOT / "dev/deploy/mock-500k-streaming-300pod.yaml"


def log(msg: str) -> None:
    print(f"[ramp] {msg}", flush=True)


def run(
    cmd: list[str], check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def kubectl_json(args: list[str]) -> dict | None:
    res = subprocess.run(
        ["kubectl", *args, "-o", "json"], capture_output=True, text=True
    )
    if res.returncode != 0:
        return None
    return json.loads(res.stdout) if res.stdout.strip() else None


def render_yaml(
    template: Path,
    name: str,
    concurrency: int,
    requests: int,
    workers: int | None = None,
) -> Path:
    spec = yaml.safe_load(template.read_text())
    spec["metadata"]["name"] = name
    prof = spec["spec"]["benchmark"]["phases"]["profiling"]
    prof["concurrency"] = concurrency
    prof["requests"] = requests
    if workers is not None:
        spec["spec"]["benchmark"]["runtime"]["workers"] = workers
    out_dir = Path("/tmp/aiperf-ramp")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{name}.yaml"
    out.write_text(yaml.safe_dump(spec, sort_keys=False))
    return out


def ns_arg(ns: str) -> list[str]:
    return ["-n", ns]


def get_job_state(ns: str, name: str) -> dict | None:
    d = kubectl_json(["get", "aiperfjob", name, *ns_arg(ns)])
    if not d:
        return None
    status = d.get("status", {}) or {}
    phase = status.get("phase")
    current_phase = status.get("currentPhase")
    profiling = (status.get("phases") or {}).get("profiling") or {}
    return {
        "phase": phase,
        "currentPhase": current_phase,
        "sendingComplete": profiling.get("sendingComplete", False),
        "requestsCompleted": profiling.get("requestsCompleted", 0),
        "requestsSent": profiling.get("requestsSent", 0),
        "requestsTotal": profiling.get("requestsTotal", 0),
        "requestsPerSecond": profiling.get("requestsPerSecond", 0),
        "recordsSuccess": profiling.get("recordsSuccess", 0),
        "wasCancelled": profiling.get("wasCancelled", False),
        "timeoutTriggered": profiling.get("timeoutTriggered", False),
    }


def wait_for_send_complete(ns: str, name: str, poll_s: int = 10) -> str:
    """Block until send phase completes or job reaches a terminal state.

    Returns one of: 'send_complete', 'failed', 'completed', 'missing'.
    """
    last = None
    while True:
        s = get_job_state(ns, name)
        if s is None:
            log(f"{name}: CR missing; treating as terminal")
            return "missing"
        sig = (
            s["phase"],
            s["currentPhase"],
            s["sendingComplete"],
            s["requestsSent"],
            s["requestsTotal"],
        )
        if sig != last:
            log(
                f"{name}: phase={s['phase']} currentPhase={s['currentPhase']} "
                f"sent={s['requestsSent']}/{s['requestsTotal']} "
                f"completed={s['requestsCompleted']} rps={s['requestsPerSecond']} "
                f"records={s['recordsSuccess']} sendingComplete={s['sendingComplete']}"
            )
            last = sig
        if s["phase"] == "Failed":
            return "failed"
        if s["phase"] == "Completed":
            return "completed"
        if s["sendingComplete"]:
            return "send_complete"
        if s["requestsTotal"] and s["requestsSent"] >= s["requestsTotal"]:
            return "send_complete"
        time.sleep(poll_s)


def delete_job(ns: str, name: str) -> None:
    log(f"deleting AIPerfJob {name}")
    run(
        ["kubectl", *ns_arg(ns), "delete", "aiperfjob", name, "--wait=false"],
        check=False,
    )


def apply_yaml(path: Path) -> None:
    log(f"applying {path.name}")
    run(["kubectl", "apply", "-f", str(path)])


def name_for(concurrency: int, suffix: str) -> str:
    k = concurrency // 1000
    return f"mock-{k}k-{suffix}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--namespace", default="acasagrande-aiperf-bench")
    p.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Base AIPerfJob yaml to clone per step.",
    )
    p.add_argument(
        "--current-job",
        required=True,
        help="Already-submitted job to drain before first ramp step.",
    )
    p.add_argument("--current-concurrency", type=int, required=True)
    p.add_argument("--step", type=int, default=50000)
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=1500000,
        help="Stop once next step would exceed this.",
    )
    p.add_argument("--requests-multiplier", type=float, default=3.0)
    p.add_argument(
        "--workers-per-concurrency",
        type=float,
        default=0.0,
        help="If >0, set runtime.workers = ceil(concurrency * this ratio), rounded "
        "up to a multiple of workers_per_pod. Default 0 keeps the template's "
        "runtime.workers value. 0.003 reproduces the 500k/1500 baseline.",
    )
    p.add_argument(
        "--suffix",
        default="ramp",
        help="Suffix for generated job names (mock-<N>k-<suffix>).",
    )
    p.add_argument(
        "--stop-on-failed",
        action="store_true",
        default=False,
        help="Halt immediately if any run ends Failed. Default False: skip past "
        "Failed/missing runs and keep ramping (bounded by --max-consecutive-failures).",
    )
    p.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=5,
        help="Halt if this many consecutive runs end in Failed/missing.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    ns = args.namespace
    cur_name = args.current_job
    cur_conc = args.current_concurrency
    template = args.template
    if not template.exists():
        log(f"template not found: {template}")
        return 2

    log(
        f"starting ramp: watching {cur_name} (concurrency={cur_conc}), step={args.step}, "
        f"max={args.max_concurrency}, template={template.name}, "
        f"stop_on_failed={args.stop_on_failed}, max_consec_fail={args.max_consecutive_failures}"
    )

    consecutive_failures = 0

    while True:
        outcome = wait_for_send_complete(ns, cur_name)
        log(f"{cur_name}: outcome={outcome}")
        if outcome in ("failed", "missing"):
            consecutive_failures += 1
            log(
                f"{cur_name}: counted as failure ({consecutive_failures}/"
                f"{args.max_consecutive_failures})"
            )
            if args.stop_on_failed:
                log(f"halting: {cur_name} failed (stop-on-failed)")
                return 1
            if consecutive_failures >= args.max_consecutive_failures:
                log(
                    f"halting: {consecutive_failures} consecutive failures "
                    f"hit --max-consecutive-failures"
                )
                return 1
        else:
            consecutive_failures = 0

        next_conc = cur_conc + args.step
        if next_conc > args.max_concurrency:
            log(f"halting: next step {next_conc} exceeds max {args.max_concurrency}")
            # Still tear down the current job so records-drain doesn't squat on resources.
            delete_job(ns, cur_name)
            return 0

        delete_job(ns, cur_name)

        next_name = name_for(next_conc, args.suffix)
        next_reqs = int(next_conc * args.requests_multiplier)
        next_workers: int | None = None
        if args.workers_per_concurrency > 0:
            # Round up to a multiple of the template's workers_per_pod so every
            # pod is fully occupied (avoids leaving a partially-filled pod).
            tpl = yaml.safe_load(template.read_text())
            per_pod = int(
                tpl["spec"]["benchmark"]["runtime"].get("workers_per_pod", 10)
            )
            raw = int(-(-next_conc * args.workers_per_concurrency // 1))
            next_workers = ((raw + per_pod - 1) // per_pod) * per_pod
        yaml_path = render_yaml(
            template, next_name, next_conc, next_reqs, workers=next_workers
        )
        apply_yaml(yaml_path)

        cur_name = next_name
        cur_conc = next_conc
        extra = f", workers={next_workers}" if next_workers is not None else ""
        log(
            f"ramped to {cur_name} (concurrency={cur_conc}, requests={next_reqs}{extra})"
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted")
        sys.exit(130)
