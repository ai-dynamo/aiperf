#!/usr/bin/env python3
"""Run trtllm-bench while recording GPU energy via the debug pynvml REST API.

Flow:
  1. Launch tools/debug_pynvml_server.py in a subprocess and wait for /status.
  2. Run a separate warmup trtllm-bench invocation (skip with --skip-warmup).
     Dataset: --warmup-dataset if set, otherwise --dataset. Optionally sliced
     to the first N JSONL lines via --warmup-num-requests.
  3. POST /start to begin sampling and capture an energy baseline (steady state).
  4. POST /sample to record the explicit post-warmup baseline snapshot.
  5. Run the official trtllm-bench (always --warmup 0).
  6. POST /sample to record the post-official snapshot.
  7. POST /stop to release NVML.
  8. Print energy used = final.total_energy_delta_j - baseline.total_energy_delta_j.

All snapshots and logs are written under --output-dir for later inspection.
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

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_SERVER = Path(__file__).resolve().parent / "debug_pynvml_server.py"


def http_get(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_post(url: str, timeout: float = 30.0) -> dict[str, Any]:
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_for_server(base: str, attempts: int = 60, delay: float = 0.5) -> None:
    last_err: Exception | None = None
    for _ in range(attempts):
        try:
            http_get(f"{base}/status", timeout=2.0)
            return
        except (urllib.error.URLError, ConnectionError, OSError) as err:
            last_err = err
            time.sleep(delay)
    raise RuntimeError(f"pynvml server at {base} did not respond: {last_err}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _build_cmd(
    args: argparse.Namespace,
    *,
    warmup_iters: int,
    dataset: str,
    concurrency: int,
) -> list[str]:
    return [
        "mpirun",
        "-n",
        "1",
        "--oversubscribe",
        "--allow-run-as-root",
        "trtllm-bench",
        "--model",
        args.model,
        "throughput",
        "--tp",
        str(args.tp),
        "--extra_llm_api_options",
        args.config,
        "--warmup",
        str(warmup_iters),
        "--dataset",
        dataset,
        "--streaming",
        "--concurrency",
        str(concurrency),
    ]


def build_warmup_cmd(args: argparse.Namespace, warmup_dataset: Path) -> list[str]:
    """trtllm-bench command for the separate warmup invocation.

    Only called when --warmup-dataset is set. warmup_dataset is either
    args.warmup_dataset directly, or a sliced-down copy if --warmup-num-requests
    > 0.
    """
    return _build_cmd(
        args,
        warmup_iters=args.warmup,
        dataset=str(warmup_dataset),
        concurrency=args.warmup_concurrency or args.concurrency,
    )


def build_official_cmd(args: argparse.Namespace) -> list[str]:
    """trtllm-bench command for the measured official run; no in-process warmup."""
    return _build_cmd(
        args,
        warmup_iters=0,
        dataset=args.dataset,
        concurrency=args.concurrency,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="pynvml server host")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="pynvml server port"
    )
    parser.add_argument(
        "--server",
        default=str(DEFAULT_SERVER),
        help=f"Path to debug_pynvml_server.py (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch the server (default: current sys.executable)",
    )
    parser.add_argument(
        "--output-dir",
        default="./power-run",
        help="Directory for JSON snapshots and the bench log",
    )
    parser.add_argument(
        "--bench-log",
        default="trtllm-bench.log",
        help="Filename for trtllm-bench combined stdout+stderr (under --output-dir)",
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-70B-Instruct", help="--model"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path passed to trtllm-bench --extra_llm_api_options (must exist)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path passed to trtllm-bench --dataset (must exist)",
    )
    parser.add_argument("--tp", type=int, default=4, help="trtllm-bench --tp")
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations passed to the warmup trtllm-bench call only "
        "(the official call always runs with --warmup 0)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        required=True,
        help="trtllm-bench --concurrency for the official run",
    )
    parser.add_argument(
        "--warmup-concurrency",
        type=int,
        default=None,
        help="Optional override for --concurrency during the warmup run "
        "(default: same as --concurrency)",
    )
    parser.add_argument(
        "--warmup-dataset",
        default=None,
        help="Optional override for --dataset during the warmup run "
        "(default: same as --dataset).",
    )
    parser.add_argument(
        "--warmup-num-requests",
        type=int,
        default=10,
        help="If > 0, slice the first N JSONL lines from the warmup source "
        "dataset (--warmup-dataset if set, else --dataset) into "
        "<output-dir>/warmup-dataset.jsonl and use that for the warmup "
        "invocation. Pass 0 to use the warmup source dataset as-is. "
        "(default: 10)",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the warmup trtllm-bench call (e.g. if the system is already warm)",
    )
    parser.add_argument(
        "--keep-server-on-error",
        action="store_true",
        help="Don't kill the server if the benchmark fails (useful for post-mortem /sample calls)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = f"http://{args.host}:{args.port}"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_log = output_dir / args.bench_log

    server_path = Path(args.server).resolve()
    if not server_path.is_file():
        print(f"error: server not found at {server_path}", file=sys.stderr)
        return 2

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"error: --config not found at {config_path}", file=sys.stderr)
        return 2
    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        print(f"error: --dataset not found at {dataset_path}", file=sys.stderr)
        return 2
    if args.warmup_dataset is not None and not Path(args.warmup_dataset).is_file():
        print(
            f"error: --warmup-dataset not found at {args.warmup_dataset}",
            file=sys.stderr,
        )
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
    if args.warmup_num_requests < 0:
        print(
            f"error: --warmup-num-requests must be >= 0 "
            f"(got {args.warmup_num_requests})",
            file=sys.stderr,
        )
        return 2

    print(f"[run] launching pynvml server: {args.python} {server_path}", flush=True)
    server_proc = subprocess.Popen(
        [args.python, str(server_path), "--host", args.host, "--port", str(args.port)],
        stdout=(output_dir / "pynvml-server.log").open("wb"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    bench_rc = 1
    warmup_rc: int | None = None
    warmup_wall_s: float | None = None
    try:
        wait_for_server(base)
        print(f"[run] server ready at {base}", flush=True)

        if args.skip_warmup:
            print("[run] skipping warmup (per --skip-warmup)", flush=True)
        else:
            src_warmup_dataset = Path(args.warmup_dataset or args.dataset)
            if args.warmup_num_requests > 0:
                sliced = output_dir / "warmup-dataset.jsonl"
                n_written = _slice_jsonl(
                    src_warmup_dataset, sliced, args.warmup_num_requests
                )
                print(
                    f"[run] sliced {n_written} lines from {src_warmup_dataset} "
                    f"-> {sliced}",
                    flush=True,
                )
                warmup_dataset_for_call = sliced
            else:
                warmup_dataset_for_call = src_warmup_dataset

            warmup_cmd = build_warmup_cmd(args, warmup_dataset_for_call)
            warmup_log = output_dir / "trtllm-bench-warmup.log"
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

        start_status = http_post(f"{base}/start")
        write_json(output_dir / "start.json", start_status)
        print(
            f"[run] /start ok (gpu_count={start_status.get('gpu_count')}, "
            f"started_at_ns={start_status.get('started_at')})",
            flush=True,
        )

        baseline = http_post(f"{base}/sample")
        write_json(output_dir / "baseline.json", baseline)
        baseline_energy = baseline.get("total_energy_delta_j") or 0.0
        baseline_power = baseline.get("total_power_w") or 0.0
        print(
            f"[run] baseline (post-warmup, pre-official): "
            f"total_power_w={baseline_power:.2f} "
            f"total_energy_delta_j={baseline_energy:.2f}",
            flush=True,
        )

        cmd = build_official_cmd(args)
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

        final = http_post(f"{base}/sample")
        write_json(output_dir / "final.json", final)
        final_energy = final.get("total_energy_delta_j") or 0.0
        final_power = final.get("total_power_w") or 0.0
        print(
            f"[run] final: total_power_w={final_power:.2f} "
            f"total_energy_delta_j={final_energy:.2f}",
            flush=True,
        )

        stop_status = http_post(f"{base}/stop")
        write_json(output_dir / "stop.json", stop_status)

        energy_j = final_energy - baseline_energy
        avg_power_w = energy_j / bench_wall_s if bench_wall_s > 0 else 0.0
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
        print(f"Artifacts: {output_dir}")
        return bench_rc
    except Exception as err:
        print(f"[run] error: {err}", file=sys.stderr)
        if args.keep_server_on_error:
            print(
                f"[run] leaving server running at {base} (pid={server_proc.pid}); "
                "kill manually when done.",
                file=sys.stderr,
            )
            server_proc = None
        raise
    finally:
        if server_proc is not None:
            try:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
                server_proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)


def _slice_jsonl(src: Path, dst: Path, n: int) -> int:
    """Write the first n non-empty lines of src to dst. Returns lines written."""
    written = 0
    with src.open("r") as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            written += 1
            if written >= n:
                break
    return written


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


if __name__ == "__main__":
    sys.exit(main())
