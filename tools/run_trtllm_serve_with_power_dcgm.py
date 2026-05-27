#!/usr/bin/env python3
"""Run aiperf against trtllm-serve to validate DCGM-sourced power against pynvml ground truth.

Identical orchestration to run_trtllm_serve_with_power.py, but aiperf is configured to
read GPU telemetry from a DCGM exporter URL (default: http://localhost:9400/metrics)
via --gpu-telemetry. The pynvml REST shim still runs alongside and provides the
ground-truth energy/power numbers. summary.json gains a parallel `aiperf_reported`
block, parsed from <artifact_dir>/profile_export_aiperf.json, exposing per-GPU
gpu_power_usage and energy_consumption stats. The script does not compute diffs
between the two sources — that's a downstream step.

Flow:
  1. Launch tools/debug_pynvml_server.py in a subprocess and wait for /status.
  2. Launch trtllm-serve in a subprocess and poll {url}/v1/models until any HTTP
     response comes back (--serve-ready-timeout / --serve-ready-poll).
  3. Run a throwaway warmup aiperf profile (skip with --skip-warmup).
  4. POST /start to begin sampling and capture an energy baseline (steady state).
  5. POST /sample to record the explicit post-warmup baseline snapshot.
  6. Run the official aiperf profile (telemetry sourced from --dcgm-url).
  7. POST /sample to record the post-official snapshot.
  8. POST /stop to release NVML.
  9. Parse aiperf's profile_export_aiperf.json; merge `aiperf_reported` into summary.json.
 10. Tear down trtllm-serve cleanly.

In synthetic mode (no --input-file), OSL is the source of truth: --osl
drives --output-tokens-mean AND the min_tokens / max_tokens extra-inputs.
In --input-file mode, per-row max_tokens in the raw_payload JSONL dictates
generation length; --isl/--osl are rejected as mutually exclusive.

All snapshots, logs, and aiperf artifact dirs are written under --output-dir.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_POWER_HOST = "127.0.0.1"
DEFAULT_POWER_PORT = 8765
DEFAULT_POWER_SERVER = Path(__file__).resolve().parent / "debug_pynvml_server.py"
DEFAULT_SERVE_HOST = "0.0.0.0"
DEFAULT_SERVE_PORT = 8000


def http_get(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_post(url: str, timeout: float = 30.0) -> dict[str, Any]:
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_for_url(url: str, *, label: str, timeout_s: float, poll_s: float) -> None:
    """Poll url until *any* HTTP response comes back (or a timeout).

    HTTPError (4xx/5xx) counts as "HTTP server is up" - we only retry on
    connection-level errors.
    """
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0):
                return
        except urllib.error.HTTPError:
            return
        except (urllib.error.URLError, ConnectionError, OSError) as err:
            last_err = err
            time.sleep(poll_s)
    raise RuntimeError(
        f"{label} at {url} did not become reachable within {timeout_s:.0f}s: {last_err}"
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_serve_cmd(args: argparse.Namespace) -> list[str]:
    return [
        "mpirun",
        "-n",
        "1",
        "--oversubscribe",
        "--allow-run-as-root",
        "trtllm-serve",
        args.model,
        "--tokenizer",
        args.tokenizer or args.model,
        "--tp_size",
        str(args.tp_size),
        "--host",
        args.serve_host,
        "--port",
        str(args.serve_port),
        "--extra_llm_api_options",
        args.serve_config,
    ]


def _build_aiperf_cmd(
    args: argparse.Namespace,
    *,
    request_count: int,
    concurrency: int,
    artifact_dir: Path,
    aiperf_warmup_count: int = 0,
) -> list[str]:
    cmd = [
        "aiperf",
        "profile",
        "-m",
        args.model,
        "--endpoint-type",
        args.endpoint_type,
        "--url",
        args.aiperf_url,
        "--streaming",
        "--random-seed",
        str(args.random_seed),
    ]
    # Dataset source: synthetic generation (default) vs raw_payload from file.
    if args.input_file is None:
        cmd += [
            "--synthetic-input-tokens-mean",
            str(args.isl),
            "--synthetic-input-tokens-stddev",
            "0",
            "--output-tokens-mean",
            str(args.osl),
            "--output-tokens-stddev",
            "0",
        ]
    else:
        cmd += [
            "--custom-dataset-type",
            "raw_payload",
            "--input-file",
            str(args.input_file),
        ]
    cmd += [
        "--request-count",
        str(request_count),
        "--concurrency",
        str(concurrency),
        "--extra-inputs",
        "ignore_eos:true",
        "--no-server-metrics",
        "--gpu-telemetry",
        args.dcgm_url,
        "--ui",
        "simple",
        "--artifact_dir",
        str(artifact_dir),
    ]
    # min/max_tokens clamp only applies when the dataset doesn't supply
    # per-row max_tokens (synthetic mode).
    if args.input_file is None:
        cmd += [
            "--extra-inputs",
            f"min_tokens:{args.osl}",
            "--extra-inputs",
            f"max_tokens:{args.osl}",
        ]
    # aiperf's --warmup-request-count requires > 0; the driver does its own
    # wrapper-level warmup via a separate aiperf invocation, so this flag is
    # only emitted when a caller explicitly opts into aiperf's internal warmup.
    if aiperf_warmup_count > 0:
        cmd += ["--warmup-request-count", str(aiperf_warmup_count)]
    return cmd


def build_aiperf_warmup_cmd(args: argparse.Namespace, artifact_dir: Path) -> list[str]:
    return _build_aiperf_cmd(
        args,
        request_count=args.warmup_request_count,
        concurrency=args.warmup_concurrency or args.concurrency,
        artifact_dir=artifact_dir,
    )


def build_aiperf_official_cmd(
    args: argparse.Namespace, artifact_dir: Path
) -> list[str]:
    return _build_aiperf_cmd(
        args,
        request_count=args.request_count,
        concurrency=args.concurrency,
        artifact_dir=artifact_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model / serve.
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Model name or path passed to both trtllm-serve and aiperf -m",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="trtllm-serve --tokenizer (default: same as --model)",
    )
    parser.add_argument(
        "--serve-config",
        required=True,
        help="Path passed to trtllm-serve --extra_llm_api_options (must exist)",
    )
    parser.add_argument("--tp-size", type=int, default=4, help="trtllm-serve --tp_size")
    parser.add_argument(
        "--serve-host",
        default=DEFAULT_SERVE_HOST,
        help="trtllm-serve --host (default 0.0.0.0)",
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=DEFAULT_SERVE_PORT,
        help="trtllm-serve --port (default 8000)",
    )
    parser.add_argument(
        "--serve-ready-timeout",
        type=float,
        default=1800.0,
        help="Seconds to wait for trtllm-serve to expose /v1/models (default 1800)",
    )
    parser.add_argument(
        "--serve-ready-poll",
        type=float,
        default=5.0,
        help="Poll interval (s) for trtllm-serve readiness (default 5)",
    )

    # aiperf.
    parser.add_argument(
        "--aiperf-url",
        default=None,
        help="aiperf --url (default: http://localhost:<serve-port>)",
    )
    parser.add_argument(
        "--endpoint-type",
        default="completions",
        help="aiperf --endpoint-type (default completions)",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=None,
        help="Input sequence length (--synthetic-input-tokens-mean, stddev=0). "
        "Mutually exclusive with --input-file. Backfilled to 1000 when "
        "--input-file is unset.",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=None,
        help="Output sequence length (--output-tokens-mean and min/max_tokens, stddev=0). "
        "Mutually exclusive with --input-file. Backfilled to 1000 when "
        "--input-file is unset.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Path to a raw_payload JSONL file. When set, the driver passes "
        "--custom-dataset-type raw_payload --input-file <path> to aiperf and "
        "disables synthetic dataset generation. Mutually exclusive with --isl "
        "and --osl. Note: the aiperf raw_payload loader requires `messages`-"
        "shaped JSONL lines, so the operator must also pass --endpoint-type "
        "chat (this driver does not auto-flip).",
    )
    parser.add_argument(
        "--random-seed", type=int, default=100, help="aiperf --random-seed"
    )
    parser.add_argument(
        "--request-count",
        type=int,
        default=10000,
        help="aiperf --request-count for the official run",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        required=True,
        help="aiperf --concurrency for the official run",
    )
    parser.add_argument(
        "--warmup-request-count",
        type=int,
        default=100,
        help="aiperf --request-count for the warmup run (default 100)",
    )
    parser.add_argument(
        "--warmup-concurrency",
        type=int,
        default=None,
        help="Override --concurrency during warmup (default: same as --concurrency)",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the warmup aiperf call (e.g. if the server is already warm)",
    )

    # pynvml debug server.
    parser.add_argument(
        "--power-host",
        default=DEFAULT_POWER_HOST,
        help="pynvml debug server host",
    )
    parser.add_argument(
        "--power-port",
        type=int,
        default=DEFAULT_POWER_PORT,
        help="pynvml debug server port",
    )
    parser.add_argument(
        "--power-server",
        default=str(DEFAULT_POWER_SERVER),
        help=f"Path to debug_pynvml_server.py (default: {DEFAULT_POWER_SERVER})",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch the pynvml server "
        "(default: current sys.executable)",
    )

    # DCGM exporter URL passed to aiperf as --gpu-telemetry.
    parser.add_argument(
        "--dcgm-url",
        default="http://localhost:9400/metrics",
        help="DCGM exporter URL passed to aiperf as --gpu-telemetry for both "
        "warmup and official runs. dcgm-exporter is presumed already running "
        "at this URL; the driver does not launch or manage it. "
        "(default: %(default)s)",
    )

    # Output / lifecycle.
    parser.add_argument(
        "--output-dir",
        default="./power-run-serve",
        help="Directory for JSON snapshots, logs, and aiperf artifact dirs",
    )
    parser.add_argument(
        "--keep-running-on-error",
        action="store_true",
        help="Don't kill trtllm-serve or pynvml server if something fails "
        "(useful for post-mortem inspection)",
    )

    return parser.parse_args()


def _terminate_group(proc: subprocess.Popen, label: str, timeout_s: float) -> None:
    """Send SIGTERM to a process group, then SIGKILL after timeout."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    print(f"[run] stopping {label} (pgid={pgid})", flush=True)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        print(
            f"[run] {label} did not exit in {timeout_s:.0f}s; sending SIGKILL",
            flush=True,
        )
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


def main() -> int:
    args = parse_args()

    serve_config = Path(args.serve_config)
    if not serve_config.is_file():
        print(f"error: --serve-config not found at {serve_config}", file=sys.stderr)
        return 2
    if args.concurrency <= 0:
        print(
            f"error: --concurrency must be > 0 (got {args.concurrency})",
            file=sys.stderr,
        )
        return 2
    if args.warmup_concurrency is not None and args.warmup_concurrency <= 0:
        print(
            f"error: --warmup-concurrency must be > 0 (got {args.warmup_concurrency})",
            file=sys.stderr,
        )
        return 2
    if args.warmup_request_count <= 0 and not args.skip_warmup:
        print(
            f"error: --warmup-request-count must be > 0 (got {args.warmup_request_count}) "
            "or pass --skip-warmup",
            file=sys.stderr,
        )
        return 2
    if args.request_count <= 0:
        print(
            f"error: --request-count must be > 0 (got {args.request_count})",
            file=sys.stderr,
        )
        return 2
    if args.input_file is not None:
        input_file_path = Path(args.input_file).resolve()
        if not input_file_path.is_file():
            print(
                f"error: --input-file not found at {input_file_path}",
                file=sys.stderr,
            )
            return 2
        args.input_file = input_file_path
        if args.isl is not None:
            print(
                "error: --isl and --input-file are mutually exclusive "
                "(--input-file supplies the prompts directly)",
                file=sys.stderr,
            )
            return 2
        if args.osl is not None:
            print(
                "error: --osl and --input-file are mutually exclusive "
                "(per-row max_tokens in the dataset dictates generation length)",
                file=sys.stderr,
            )
            return 2
    else:
        if args.isl is None:
            args.isl = 1000  # backfill original default
        if args.osl is None:
            args.osl = 1000  # backfill original default
        if args.isl <= 0 or args.osl <= 0:
            print(
                f"error: --isl and --osl must be > 0 (got isl={args.isl} osl={args.osl})",
                file=sys.stderr,
            )
            return 2

    power_server_path = Path(args.power_server).resolve()
    if not power_server_path.is_file():
        print(
            f"error: --power-server not found at {power_server_path}", file=sys.stderr
        )
        return 2

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aiperf_url is None:
        args.aiperf_url = f"http://localhost:{args.serve_port}"

    power_base = f"http://{args.power_host}:{args.power_port}"
    serve_ready_url = f"http://localhost:{args.serve_port}/v1/models"

    serve_proc: subprocess.Popen | None = None
    power_proc: subprocess.Popen | None = None

    bench_rc = 1
    warmup_rc: int | None = None
    warmup_wall_s: float | None = None
    bench_wall_s: float = 0.0

    try:
        # 1. Launch pynvml debug server.
        print(
            f"[run] launching pynvml server: {args.python} {power_server_path}",
            flush=True,
        )
        power_proc = subprocess.Popen(
            [
                args.python,
                str(power_server_path),
                "--host",
                args.power_host,
                "--port",
                str(args.power_port),
            ],
            stdout=(output_dir / "pynvml-server.log").open("wb"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_for_url(
            f"{power_base}/status",
            label="pynvml server",
            timeout_s=30.0,
            poll_s=0.5,
        )
        print(f"[run] pynvml server ready at {power_base}", flush=True)

        # 2. Launch trtllm-serve.
        serve_cmd = build_serve_cmd(args)
        serve_log = output_dir / "trtllm-serve.log"
        print(f"[run] launching trtllm-serve: {' '.join(serve_cmd)}", flush=True)
        print(f"[run] trtllm-serve log: {serve_log}", flush=True)
        serve_proc = subprocess.Popen(
            serve_cmd,
            stdout=serve_log.open("wb"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(
            f"[run] waiting up to {args.serve_ready_timeout:.0f}s for {serve_ready_url}",
            flush=True,
        )
        wait_for_url(
            serve_ready_url,
            label="trtllm-serve",
            timeout_s=args.serve_ready_timeout,
            poll_s=args.serve_ready_poll,
        )
        print(f"[run] trtllm-serve ready at {serve_ready_url}", flush=True)

        # 3. Warmup aiperf run.
        if args.skip_warmup:
            print("[run] skipping warmup (per --skip-warmup)", flush=True)
        else:
            warmup_artifact = output_dir / "aiperf-warmup"
            warmup_cmd = build_aiperf_warmup_cmd(args, warmup_artifact)
            warmup_log = output_dir / "aiperf-warmup.log"
            print(f"[run] warmup: {' '.join(warmup_cmd)}", flush=True)
            print(f"[run] warmup log: {warmup_log}", flush=True)
            warmup_start = time.monotonic()
            with warmup_log.open("wb") as fh:
                warmup_rc = subprocess.call(
                    warmup_cmd, stdout=fh, stderr=subprocess.STDOUT
                )
            warmup_wall_s = time.monotonic() - warmup_start
            print(
                f"[run] warmup exited rc={warmup_rc} wall={warmup_wall_s:.1f}s",
                flush=True,
            )
            if warmup_rc != 0:
                print(
                    f"[run] warmup failed (rc={warmup_rc}); aborting before official run",
                    file=sys.stderr,
                )
                return warmup_rc

        # 4. Capture baseline at steady state.
        start_status = http_post(f"{power_base}/start")
        write_json(output_dir / "start.json", start_status)
        print(
            f"[run] /start ok (gpu_count={start_status.get('gpu_count')}, "
            f"started_at_ns={start_status.get('started_at')})",
            flush=True,
        )

        baseline = http_post(f"{power_base}/sample")
        write_json(output_dir / "baseline.json", baseline)
        baseline_energy = baseline.get("total_energy_delta_j") or 0.0
        baseline_power = baseline.get("total_power_w") or 0.0
        print(
            f"[run] baseline (post-warmup, pre-official): "
            f"total_power_w={baseline_power:.2f} "
            f"total_energy_delta_j={baseline_energy:.2f}",
            flush=True,
        )

        # 5. Official aiperf run.
        official_artifact = output_dir / "aiperf-official"
        cmd = build_aiperf_official_cmd(args, official_artifact)
        bench_log = output_dir / "aiperf-official.log"
        print(f"[run] official benchmark: {' '.join(cmd)}", flush=True)
        print(f"[run] bench log: {bench_log}", flush=True)
        bench_start = time.monotonic()
        with bench_log.open("wb") as fh:
            bench_rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        bench_wall_s = time.monotonic() - bench_start
        print(
            f"[run] official benchmark exited rc={bench_rc} wall={bench_wall_s:.1f}s",
            flush=True,
        )

        # 6. Final snapshot.
        final = http_post(f"{power_base}/sample")
        write_json(output_dir / "final.json", final)
        final_energy = final.get("total_energy_delta_j") or 0.0
        final_power = final.get("total_power_w") or 0.0
        print(
            f"[run] final: total_power_w={final_power:.2f} "
            f"total_energy_delta_j={final_energy:.2f}",
            flush=True,
        )

        stop_status = http_post(f"{power_base}/stop")
        write_json(output_dir / "stop.json", stop_status)

        energy_j = final_energy - baseline_energy
        avg_power_w = energy_j / bench_wall_s if bench_wall_s > 0 else 0.0
        aiperf_reported = _parse_aiperf_telemetry(official_artifact)
        summary = {
            "warmup_skipped": args.skip_warmup,
            "warmup_return_code": warmup_rc,
            "warmup_wall_seconds": warmup_wall_s,
            "bench_wall_seconds": bench_wall_s,
            "bench_return_code": bench_rc,
            "baseline_total_energy_delta_j": baseline_energy,
            "final_total_energy_delta_j": final_energy,
            "benchmark_energy_j": energy_j,
            "benchmark_energy_wh": energy_j / 3600.0,
            "average_power_w": avg_power_w,
            "per_gpu_energy_delta_j": _per_gpu_delta(baseline, final),
            "isl": args.isl,
            "osl": args.osl,
            "request_count": args.request_count,
            "concurrency": args.concurrency,
            "dcgm_url": args.dcgm_url,
            "aiperf_reported": aiperf_reported,
            "input_file": str(args.input_file) if args.input_file is not None else None,
        }
        write_json(output_dir / "summary.json", summary)

        print()
        print("=== Energy Summary ===")
        if args.skip_warmup:
            print("Warmup                  : skipped")
        else:
            print(f"Warmup wall time        : {warmup_wall_s:.1f} s (rc={warmup_rc})")
        print(f"Official wall time      : {bench_wall_s:.1f} s (rc={bench_rc})")
        print(f"Baseline energy (delta) : {baseline_energy:.2f} J  (post-warmup)")
        print(f"Final energy (delta)    : {final_energy:.2f} J  (post-official)")
        print(
            f"Energy used by official : {energy_j:.2f} J ({energy_j / 3600.0:.3f} Wh)"
        )
        print(f"Average power draw      : {avg_power_w:.2f} W")
        print("Per-GPU energy (J):")
        for entry in summary["per_gpu_energy_delta_j"]:
            print(
                f"  gpu={entry['gpu_index']:>2} "
                f"uuid={entry['gpu_uuid']} "
                f"delta_j={entry['energy_delta_j']:.2f}"
            )

        print()
        print("=== AIPerf-reported (DCGM) ===")
        print(f"Endpoint               : {args.dcgm_url}")
        if not aiperf_reported:
            print(
                "(no aiperf telemetry collected — "
                "check dcgm-exporter availability and aiperf logs)"
            )
        else:
            print("Per-GPU (gpu_power_usage W, energy_consumption MJ):")
            for gpu_uuid, gpu_data in aiperf_reported.items():
                power_stats = gpu_data.get("gpu_power_usage") or {}
                energy_stats = gpu_data.get("energy_consumption") or {}
                power_avg = _fmt_stat(power_stats.get("avg"))
                power_p99 = _fmt_stat(power_stats.get("p99"))
                energy_sum = _fmt_stat(energy_stats.get("sum", energy_stats.get("avg")))
                idx = gpu_data.get("gpu_index")
                idx_str = f"{idx:>2}" if isinstance(idx, int) else " ?"
                print(
                    f"  gpu={idx_str} "
                    f"uuid={gpu_uuid} "
                    f"power_avg={power_avg} "
                    f"power_p99={power_p99} "
                    f"energy={energy_sum}"
                )

        print(f"Artifacts: {output_dir}")
        return bench_rc

    except Exception as err:
        print(f"[run] error: {err}", file=sys.stderr)
        if args.keep_running_on_error:
            if power_proc is not None:
                print(
                    f"[run] leaving pynvml server running at {power_base} "
                    f"(pid={power_proc.pid})",
                    file=sys.stderr,
                )
                power_proc = None
            if serve_proc is not None:
                print(
                    f"[run] leaving trtllm-serve running at {serve_ready_url} "
                    f"(pid={serve_proc.pid})",
                    file=sys.stderr,
                )
                serve_proc = None
        raise
    finally:
        # Tear down serve first so it stops calling NVML before we kill pynvml server.
        if serve_proc is not None:
            _terminate_group(serve_proc, "trtllm-serve", timeout_s=60.0)
        if power_proc is not None:
            _terminate_group(power_proc, "pynvml server", timeout_s=10.0)


def _per_gpu_delta(
    baseline: dict[str, Any], final: dict[str, Any]
) -> list[dict[str, Any]]:
    base_samples = {s.get("gpu_index"): s for s in baseline.get("samples", [])}
    out: list[dict[str, Any]] = []
    for fsample in final.get("samples", []):
        idx = fsample.get("gpu_index")
        bsample = base_samples.get(idx, {})
        b_delta = bsample.get("energy_delta_j") or 0.0
        f_delta = fsample.get("energy_delta_j") or 0.0
        out.append(
            {
                "gpu_index": idx,
                "gpu_uuid": fsample.get("gpu_uuid"),
                "energy_delta_j": f_delta - b_delta,
            }
        )
    return out


def _fmt_stat(value: Any) -> str:
    """Format a stat value for console output; falls back to 'n/a' if non-numeric."""
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "n/a"


def _parse_aiperf_telemetry(artifact_dir: Path) -> dict[str, Any]:
    """Parse per-GPU power/energy stats from aiperf's profile_export_aiperf.json.

    Returns a dict keyed by GPU UUID. Each value carries the per-GPU
    identifiers plus the raw stat dicts aiperf computed for `gpu_power_usage`
    (watts) and `energy_consumption` (megajoules, counter-delta).

    Empty dict on missing file or empty/absent telemetry section (soft failure
    — a warning is printed to stderr and the caller still writes summary.json).
    Re-raises json.JSONDecodeError if the file exists but is malformed (hard
    failure — that indicates an aiperf bug worth surfacing).
    """
    json_path = artifact_dir / "profile_export_aiperf.json"
    if not json_path.is_file():
        print(
            f"[run] warning: no aiperf telemetry file at {json_path}",
            file=sys.stderr,
        )
        return {}

    data = json.loads(json_path.read_text())
    telemetry = data.get("telemetry_data")
    if not telemetry:
        print(
            f"[run] warning: aiperf JSON has no telemetry_data section at {json_path}",
            file=sys.stderr,
        )
        return {}

    endpoints = telemetry.get("endpoints", {})
    if not endpoints:
        print(
            f"[run] warning: aiperf telemetry has no endpoints at {json_path}",
            file=sys.stderr,
        )
        return {}

    out: dict[str, Any] = {}
    for endpoint_url, endpoint_data in endpoints.items():
        if not isinstance(endpoint_data, dict):
            continue
        gpus = endpoint_data.get("gpus", {})
        if not isinstance(gpus, dict):
            continue
        for gpu_uuid, gpu in gpus.items():
            if not isinstance(gpu, dict):
                continue
            metrics = gpu.get("metrics", {}) or {}
            out[gpu_uuid] = {
                "gpu_index": gpu.get("gpu_index"),
                "gpu_name": gpu.get("gpu_name"),
                "hostname": gpu.get("hostname"),
                "endpoint_url": endpoint_url,
                "gpu_power_usage": metrics.get("gpu_power_usage", {}) or {},
                "energy_consumption": metrics.get("energy_consumption", {}) or {},
            }
    return out


if __name__ == "__main__":
    sys.exit(main())
