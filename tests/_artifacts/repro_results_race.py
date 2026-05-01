"""Reproduction script for the `aiperf kube results` post-completion race.

Submits a fast AIPerfJob, polls until phase=Completed, then immediately shells
to `aiperf kube results`. Loops N times.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

CONTEXT = "kind-aiperf-60a95def"
NAMESPACE = "debug-test"
ATTEMPTS = 8


def _kubectl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", "--context", CONTEXT, *args],
        capture_output=True,
        text=True,
        check=check,
    )


JOB_TEMPLATE = """apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: {name}
  namespace: {namespace}
spec:
  benchmark:
    benchmark:
      requestCount: 16
      warmupRequestCount: 0
    loadgen:
      concurrency: 4
    endpoint:
      url: http://aiperf-mock-server.default.svc.cluster.local:8000/v1
      type: chat
      streaming: true
    input:
      conversation:
        num: 4
      tokens:
        mean: 16
        stddev: 0
      output_tokens:
        mean: 16
        stddev: 0
    tokenizer:
      name: gpt2
  image: aiperf:local
  imagePullPolicy: Never
"""


def _create_job(name: str) -> None:
    manifest = JOB_TEMPLATE.format(name=name, namespace=NAMESPACE)
    p = subprocess.run(
        ["kubectl", "--context", CONTEXT, "apply", "-f", "-"],
        input=manifest,
        text=True,
        capture_output=True,
    )
    if p.returncode != 0:
        print(f"create failed: {p.stdout}\n{p.stderr}")
        sys.exit(2)


def _delete_job(name: str) -> None:
    subprocess.run(
        [
            "kubectl",
            "--context",
            CONTEXT,
            "-n",
            NAMESPACE,
            "delete",
            "aiperfjob",
            name,
            "--wait=false",
        ],
        capture_output=True,
    )


def _phase(name: str) -> str:
    p = _kubectl(
        "-n",
        NAMESPACE,
        "get",
        "aiperfjob",
        name,
        "-o",
        "jsonpath={.status.phase}",
        check=False,
    )
    return p.stdout.strip() or "<empty>"


def _poll_until_complete(name: str, timeout: float = 300.0) -> tuple[str, float]:
    """Tight 100ms poll. Returns (final_phase, observed_at_monotonic)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        phase = _phase(name)
        if phase in ("Completed", "Failed", "Cancelled"):
            return phase, time.monotonic()
        time.sleep(0.1)
    return "<timeout>", time.monotonic()


def _run_results(name: str, output_dir: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        [
            ".venv/bin/aiperf",
            "kube",
            "results",
            name,
            "--namespace",
            NAMESPACE,
            "--kube-context",
            CONTEXT,
            "--output",
            str(output_dir),
            "--all",
        ],
        cwd="/home/anthony/nvidia/projects/aiperf/ajc/new-config-kube",
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    return proc.returncode, proc.stdout, proc.stderr


def _capture_state(name: str) -> str:
    """Snapshot CR state at the moment of CLI failure."""
    parts = []
    p = _kubectl(
        "-n",
        NAMESPACE,
        "get",
        "aiperfjob",
        name,
        "-o",
        "jsonpath={.status.phase}|{.status.jobId}|{.metadata.resourceVersion}",
        check=False,
    )
    parts.append(f"  cr_state: {p.stdout!r} (rc={p.returncode}, err={p.stderr!r})")
    return "\n".join(parts)


def main() -> None:
    fails = 0
    for i in range(1, ATTEMPTS + 1):
        suffix = uuid.uuid4().hex[:6]
        name = f"repro-{i}-{suffix}"
        out = Path(f"/tmp/repro-results-{i}-{suffix}")
        print(f"\n=== Attempt {i}: {name} ===", flush=True)

        _create_job(name)
        try:
            phase, observed_at = _poll_until_complete(name)
            cli_started_at = time.monotonic()
            rc, stdout, stderr = _run_results(name, out)
            cli_dur = time.monotonic() - cli_started_at
            gap_ms = (cli_started_at - observed_at) * 1000.0

            files = (
                [p.name for p in out.rglob("*") if p.is_file()] if out.exists() else []
            )
            ok_lookup = "No AIPerfJob or AIPerfSweep found" not in stdout
            ok_files = bool(files)

            print(
                f"  phase={phase} gap_ms_observed_to_cli={gap_ms:.1f} "
                f"cli_rc={rc} cli_dur={cli_dur:.2f}s files={len(files)} "
                f"ok_lookup={ok_lookup}",
                flush=True,
            )
            if not ok_lookup or not ok_files:
                fails += 1
                print(
                    f"  FAIL stdout=\n{stdout}\n  stderr=\n{stderr}",
                    flush=True,
                )
                print(_capture_state(name), flush=True)
        finally:
            _delete_job(name)
    print(f"\nTotal failures: {fails}/{ATTEMPTS}", flush=True)


if __name__ == "__main__":
    main()
