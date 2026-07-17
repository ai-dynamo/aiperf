# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-tune the mock server's scheduler to match real profile_export trace(s).

Reads the target TTFT/ITL and OSL distributions from a profile_export.jsonl,
launches the Rust mock server locally, drives it with closed-loop load at the
target concurrency, and solves --scheduler-prefill-chunks-per-request so the
emitted TTFT median matches the trace.

With a SECOND trace at a different concurrency (--trace2/--concurrency2), it also
derives the sublinear prefill-throughput exponent from the TTFT-vs-concurrency
slope, so a SINGLE config reproduces BOTH concurrency points (real serving TTFT
grows sublinearly with load; a fixed-rate scheduler would grow it linearly):

    exponent = 1 - log(ttft2/ttft1) / log(c2/c1)        # TTFT ~ C^(1-exponent)

Key tricks that keep this cheap:
  * Scheduler TTFT is queue-driven and ISL-independent, so tiny prompts suffice;
    only the OSL distribution is replayed.
  * TTFT scales linearly in chunks-per-request and step_ms (and the exponent is
    dimensionless), so it tunes at a compressed step_ms for speed and maps the
    result back to the real step_ms.
  * max-batch is decoupled from the load (set to peak concurrency by default) so
    decode never bottlenecks and the prefill exponent alone shapes the curve.

Usage (single point):
  python tune_to_trace.py --trace c512.jsonl --concurrency 512 \
      --mock-bin rust/target/release/aiperf-mock-server

Usage (two points -> concurrency-transferable config):
  python tune_to_trace.py --trace c512.jsonl --concurrency 512 \
      --trace2 c1024.jsonl --concurrency2 1024 \
      --mock-bin rust/target/release/aiperf-mock-server
"""

from __future__ import annotations

import argparse
import asyncio
import math
import subprocess
import sys
import time

import aiohttp
import numpy as np
import orjson


def metric_value(metrics, key):
    metric = metrics.get(key)
    return metric.get("value") if isinstance(metric, dict) else None


def read_trace(path):
    ttft, itl, osl = [], [], []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            m, md = r.get("metrics", {}), r.get("metadata", {})
            if md.get("was_cancelled"):
                continue
            t = metric_value(m, "time_to_first_token")
            i = metric_value(m, "inter_token_latency")
            o = metric_value(m, "output_sequence_length")
            if t is not None:
                ttft.append(t)
            if i is not None:
                itl.append(i)
            if o is not None:
                osl.append(int(o))
    return np.array(ttft, float), np.array(itl, float), np.array(osl, int)


def lognormal_cv(a):
    a = a[a > 0]
    return float(math.sqrt(math.exp(np.log(a).std() ** 2) - 1))


async def drive(url, concurrency, duration, osl_samples, rng):
    """Closed-loop load; returns (ttft_ms list, itl_ms list)."""
    ttfts, itls = [], []
    osl_samples = np.minimum(osl_samples, 2000)  # cap mega-decodes so tuning stays fast

    async def worker(sess, deadline):
        while time.perf_counter() < deadline:
            osl = int(rng.choice(osl_samples))
            body = {
                "model": "m",
                "stream": True,
                "max_tokens": max(1, osl),
                "messages": [{"role": "user", "content": "tune"}],
            }
            t0 = time.perf_counter()
            first = last = None
            try:
                async with sess.post(url, json=body) as resp:
                    async for ln in resp.content:
                        if not ln.startswith(b"data:"):
                            continue
                        now = time.perf_counter()
                        if first is None:
                            first = now
                            ttfts.append((now - t0) * 1000)
                        elif last is not None:
                            itls.append((now - last) * 1000)
                        last = now
            except aiohttp.ClientError:
                pass

    deadline = time.perf_counter() + duration
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as sess:
        await asyncio.gather(*[worker(sess, deadline) for _ in range(concurrency)])
    return ttfts[len(ttfts) // 5 :], itls[len(itls) // 5 :]  # drop warmup


def launch_mock(
    args, *, step_ms, max_batch, pmax, chunks, work_cv, itl, itl_cv, exponent, ref
):
    cmd = [
        args.mock_bin,
        "--port",
        str(args.port),
        "--no-tokenizer",
        "--log-level",
        "ERROR",
        "--scheduler-enabled",
        "--scheduler-step-ms",
        str(step_ms),
        "--scheduler-max-batch-size",
        str(max_batch),
        "--scheduler-max-prefill-chunks-per-step",
        str(pmax),
        "--scheduler-prefill-chunks-per-request",
        str(chunks),
        "--scheduler-prefill-work-cv",
        f"{work_cv:.3f}",
        "--scheduler-prefill-throughput-exponent",
        f"{exponent:.4f}",
        "--scheduler-prefill-throughput-ref",
        str(ref),
        "--itl",
        f"{itl:.2f}",
        "--itl-jitter-cv",
        f"{itl_cv:.3f}",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace", required=True, help="primary trace (median is matched here)"
    )
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument(
        "--trace2",
        help="second trace at a different concurrency (derives the exponent)",
    )
    ap.add_argument("--concurrency2", type=int)
    ap.add_argument("--mock-bin", required=True)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument(
        "--duration", type=float, default=20.0, help="seconds per probe run"
    )
    ap.add_argument(
        "--tune-step-ms",
        type=float,
        default=0.0,
        help="compressed step_ms for fast tuning (0 = derive ~real/10)",
    )
    ap.add_argument(
        "--pmax", type=int, default=4, help="base max prefill chunks per step"
    )
    ap.add_argument(
        "--max-batch",
        type=int,
        default=0,
        help="decode batch capacity (0 = auto: peak concurrency so decode never bottlenecks)",
    )
    args = ap.parse_args()

    ttft, itl, osl = read_trace(args.trace)
    if len(ttft) == 0 or len(osl) == 0:
        sys.exit("trace has no TTFT/OSL data")
    tgt_p50 = float(np.percentile(ttft, 50))
    tgt_p90 = float(np.percentile(ttft, 90))
    tgt_p99 = float(np.percentile(ttft, 99))
    work_cv = lognormal_cv(ttft)
    real_step = (
        max(1.0, float(np.percentile(itl[np.isfinite(itl)], 50))) if len(itl) else 15.0
    )
    itl_cv = lognormal_cv(itl[np.isfinite(itl)]) if len(itl) else 0.0

    # Second point -> derive the sublinear prefill-throughput exponent.
    exponent, ref = 0.0, args.concurrency
    pts = [(args.concurrency, tgt_p50, osl)]
    if args.trace2 and args.concurrency2:
        ttft2, _, osl2 = read_trace(args.trace2)
        p50_2 = float(np.percentile(ttft2, 50))
        x = math.log(p50_2 / tgt_p50) / math.log(args.concurrency2 / args.concurrency)
        exponent = max(0.0, min(0.95, 1.0 - x))
        pts.append((args.concurrency2, p50_2, osl2))
        print(
            f"two-point: TTFT {tgt_p50:.0f}ms@C{args.concurrency} -> {p50_2:.0f}ms@C{args.concurrency2}"
            f"  (TTFT ~ C^{x:.2f})  =>  prefill-throughput-exponent={exponent:.3f}, ref={ref}"
        )

    peak_c = max(c for c, _, _ in pts)
    max_batch = (
        args.max_batch or peak_c
    )  # decode non-binding for a prefill/queue-dominated trace
    tune_step = args.tune_step_ms or max(0.5, real_step / 10.0)
    scale = tune_step / real_step
    rng = np.random.default_rng(0)

    print(
        f"trace: n={len(ttft)}  TTFT p50/90/99={tgt_p50:.0f}/{tgt_p90:.0f}/{tgt_p99:.0f}ms"
        f"  work_cv={work_cv:.2f}  ITL p50={real_step:.0f}ms cv={itl_cv:.2f}"
    )
    print(
        f"tuning at step_ms={tune_step:.2f} (={scale:.3f}x real {real_step:.0f}); "
        f"max_batch={max_batch}; target median (scaled)={tgt_p50 * scale:.0f}ms\n"
    )

    async def run(chunks, conc, osl_arr):
        proc = launch_mock(
            args,
            step_ms=tune_step,
            max_batch=max_batch,
            pmax=args.pmax,
            chunks=chunks,
            work_cv=work_cv,
            itl=real_step * scale,
            itl_cv=itl_cv,
            exponent=exponent,
            ref=ref,
        )
        try:
            await asyncio.sleep(1.5)
            url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
            t, _ = await drive(url, conc, args.duration, osl_arr, rng)
            return (float(np.median(t)) if t else 0.0, t)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    # Damped fixed-point solve of chunks-per-request at the PRIMARY concurrency.
    # The update is damped (^0.7) because each run carries ~15% sampling noise.
    target_scaled = tgt_p50 * scale
    chunks, best, best_err, t_best = 8, 8, 1e9, []
    for it in range(5):
        med, ti = asyncio.run(run(chunks, args.concurrency, osl))
        if med <= 0:
            sys.exit("no samples — is the mock binary correct / port free?")
        err = med / target_scaled - 1.0
        print(
            f"  iter {it}: chunks={chunks:>3} -> median {med / scale:7.0f}ms "
            f"(target {tgt_p50:.0f}, err {err * 100:+.0f}%)"
        )
        if abs(err) < abs(best_err):
            best, best_err, t_best = chunks, err, ti
        if abs(err) <= 0.12:
            break
        nxt = max(1, round(chunks * (target_scaled / med) ** 0.7))
        if nxt == chunks:
            break
        chunks = nxt
    chunks = best

    def unscale_percentile(samples, percentile):
        return np.percentile(samples, percentile) / scale

    print(f"\nvalidation (TTFT unscaled to real step_ms={real_step:.0f}):")
    print(
        f"  C{args.concurrency:<5} target p50/90/99 = {tgt_p50:.0f}/{tgt_p90:.0f}/{tgt_p99:.0f}ms"
    )
    print(
        f"  C{args.concurrency:<5} tuned  p50/90/99 = "
        f"{unscale_percentile(t_best, 50):.0f}/"
        f"{unscale_percentile(t_best, 90):.0f}/"
        f"{unscale_percentile(t_best, 99):.0f}ms"
    )
    # Transfer check: the SAME config at the second concurrency (not re-tuned).
    if len(pts) > 1:
        c2, p50_2, osl2 = pts[1]
        _, t2 = asyncio.run(run(chunks, c2, osl2))
        got = unscale_percentile(t2, 50) if t2 else 0.0
        print(
            f"  C{c2:<5} target p50         = {p50_2:.0f}ms   (transfer test, NOT re-tuned)"
        )
        print(
            f"  C{c2:<5} tuned  p50         = {got:.0f}ms   (err {got / p50_2 * 100 - 100:+.0f}%)"
        )
    print(
        f"  (ITL set directly from trace: step_ms={real_step:.0f}, itl-jitter-cv={itl_cv:.2f})"
    )

    print("\n>>> calibrated command:")
    extra = (
        (
            f" \\\n  --scheduler-prefill-throughput-exponent {exponent:.3f}"
            f" --scheduler-prefill-throughput-ref {ref}"
        )
        if exponent > 0
        else ""
    )
    print(
        f"aiperf-mock-server --scheduler-enabled \\\n"
        f"  --scheduler-step-ms {real_step:.0f} --scheduler-max-batch-size {max_batch} \\\n"
        f"  --scheduler-prefill-chunks-per-request {chunks} "
        f"--scheduler-max-prefill-chunks-per-step {args.pmax} \\\n"
        f"  --scheduler-prefill-work-cv {work_cv:.2f}{extra} \\\n"
        f"  --itl {real_step:.0f} --itl-jitter-cv {itl_cv:.2f}"
    )


if __name__ == "__main__":
    main()
