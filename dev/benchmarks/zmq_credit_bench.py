#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open-loop ZMQ credit microbenchmark: throughput + round-trip latency.

Blasts COUNT credits **open-loop** (fire all of them as fast as possible, no
closed-loop in-flight window) across three ZMQ patterns, comparing the
production zmq.asyncio await drive against the FdEdgeReader / sync-NOBLOCK
drive, with NOTHING else in the loop (no HTTP, no aiperf services, no mock):

  await : zmq.asyncio await-recv / await-send, yielding every YIELD_INTERVAL.
  fd    : FdEdgeReader (loop.add_reader + sync NOBLOCK batch-drain) for recv,
          sync NOBLOCK send.

Each credit carries a send timestamp (perf_counter_ns, CLOCK_MONOTONIC =
cross-process comparable on Linux); the worker echoes it back, so the producer
records the true per-credit round-trip latency on return. Reports throughput
(credits/sec over the blast) AND latency percentiles (p50/p99/p99.9).

Methodology:
  * Open-loop blast of N credits, not a closed in-flight loop.
  * MULTI-PROCESS rig (single-process inflates 20-40x via uncontested sockets).
  * GC disabled in every process.
  * Latency clock is perf_counter_ns (CLOCK_MONOTONIC, cross-process on Linux).

Usage:
  python dev/benchmarks/zmq_credit_bench.py --mode fd --pattern dealer_router --count 250000 --workers 8
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import multiprocessing as mp
import time
from collections import deque

import msgspec
import zmq
import zmq.asyncio

from aiperf.zmq.fd_reader import FdEdgeReader

YIELD_INTERVAL = 10
_now = time.perf_counter_ns


class Credit(msgspec.Struct, tag_field="t", tag="c"):
    id: int
    ts: int = 0  # producer send time (perf_counter_ns)
    phase: str = "profiling"


class CreditReturn(msgspec.Struct, tag_field="t", tag="cr"):
    id: int
    ts: int = 0  # echoed producer send time


_WireMsg = Credit | CreditReturn
_ENC = msgspec.msgpack.Encoder()
_DEC = msgspec.msgpack.Decoder(_WireMsg)
_READY = -2


def _pcts(samples_ns: list[int]) -> dict:
    if not samples_ns:
        return {}
    s = sorted(samples_ns)
    n = len(s)

    def p(q: float) -> float:
        # Nearest-rank on (n-1) so p50 etc. aren't biased toward higher indices.
        return s[min(n - 1, int(q * (n - 1)))] / 1e6  # -> ms

    return {
        "p50": p(0.50),
        "p90": p(0.90),
        "p99": p(0.99),
        "p999": p(0.999),
        "max": s[-1] / 1e6,
        "min": s[0] / 1e6,
    }


def _run(coro) -> None:
    try:
        import uvloop

        uvloop.run(coro)
    except ImportError:
        asyncio.run(coro)


def _recv_mp(sock):
    """Sync NOBLOCK multipart recv (recv_multipart delegates to the async recv)."""
    first = zmq.Socket.recv(sock, flags=zmq.NOBLOCK)
    last = first
    while sock.getsockopt(zmq.RCVMORE):
        last = zmq.Socket.recv(sock, flags=zmq.NOBLOCK)
    return first, last


# =============================================================================
# Worker: echo each credit straight back (id + ts), zero work.
# =============================================================================
async def _worker(mode, pattern, addrs, identity, count):
    ctx = zmq.asyncio.Context.instance()
    if pattern == "dealer_router":
        rx = ctx.socket(zmq.DEALER)
        rx.setsockopt(zmq.IDENTITY, identity.encode())
        rx.setsockopt(zmq.SNDHWM, 0)
        rx.setsockopt(zmq.RCVHWM, 0)
        rx.connect(addrs[0])
        tx = rx
        await rx.send(_ENC.encode(CreditReturn(id=_READY)))
    else:  # push_pull
        rx = ctx.socket(zmq.PULL)
        rx.setsockopt(zmq.RCVHWM, 0)
        rx.connect(addrs[0])
        tx = ctx.socket(zmq.PUSH)
        tx.setsockopt(zmq.SNDHWM, 0)
        tx.connect(addrs[1])
        zmq.Socket.send(tx, _ENC.encode(CreditReturn(id=_READY)), copy=False)

    done = asyncio.Event()
    seen = {"n": 0}

    def make_return(c: Credit) -> bytes:
        return _ENC.encode(CreditReturn(id=c.id, ts=c.ts))

    if mode == "fd":

        def recv_one():
            return _DEC.decode(zmq.Socket.recv(rx, flags=zmq.NOBLOCK))

        def dispatch(c):
            zmq.Socket.send(tx, make_return(c), flags=zmq.NOBLOCK, copy=False)
            seen["n"] += 1
            if seen["n"] >= count:
                done.set()

        reader = FdEdgeReader(
            socket=rx, recv_one=recv_one, dispatch=dispatch, batch_limit=YIELD_INTERVAL
        )
        reader.start()
        await done.wait()
        reader.stop()
    else:
        n = 0
        while seen["n"] < count:
            c = _DEC.decode(await rx.recv())
            await tx.send(make_return(c))
            seen["n"] += 1
            n += 1
            if n % YIELD_INTERVAL == 0:
                await asyncio.sleep(0)
    with contextlib.suppress(Exception):
        rx.close(0)
        if tx is not rx:
            tx.close(0)


def _worker_main(mode, pattern, addrs, identity, count):
    gc.disable()
    _run(_worker(mode, pattern, addrs, identity, count))


# =============================================================================
# Producer: blast COUNT credits open-loop, collect returns, record RTT latency.
# =============================================================================
async def _producer(mode, pattern, addrs, n_workers, count, result):
    ctx = zmq.asyncio.Context.instance()
    if pattern == "dealer_router":
        sock = ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.SNDHWM, 0)
        sock.setsockopt(zmq.RCVHWM, 0)
        sock.bind(addrs[0])
        rx = tx = sock
        idents: list[bytes] = []
        while len(idents) < n_workers:
            idents.append((await sock.recv_multipart())[0])
        targets = deque(idents)
    else:  # push_pull
        tx = ctx.socket(zmq.PUSH)
        tx.setsockopt(zmq.SNDHWM, 0)
        tx.bind(addrs[0])
        rx = ctx.socket(zmq.PULL)
        rx.setsockopt(zmq.RCVHWM, 0)
        rx.bind(addrs[1])
        ready = 0
        while ready < n_workers:
            await rx.recv()
            ready += 1
        targets = None

    lat: list[int] = []
    done = asyncio.Event()
    times = {"first_send": 0, "last_recv": 0}

    def on_return(cr: CreditReturn) -> None:
        now = _now()
        lat.append(now - cr.ts)
        times["last_recv"] = now
        if len(lat) >= count:
            done.set()

    # ---- recv side ----
    recv_task = None
    reader = None
    if mode == "fd":
        if pattern == "dealer_router":

            def recv_one():
                _ident, payload = _recv_mp(rx)
                return _DEC.decode(payload)
        else:

            def recv_one():
                return _DEC.decode(zmq.Socket.recv(rx, flags=zmq.NOBLOCK))

        reader = FdEdgeReader(
            socket=rx, recv_one=recv_one, dispatch=on_return, batch_limit=YIELD_INTERVAL
        )
        reader.start()
    else:

        async def recv_loop():
            n = 0
            while not done.is_set():
                if pattern == "dealer_router":
                    cr = _DEC.decode((await rx.recv_multipart())[-1])
                else:
                    cr = _DEC.decode(await rx.recv())
                on_return(cr)
                n += 1
                if n % YIELD_INTERVAL == 0:
                    await asyncio.sleep(0)

        recv_task = asyncio.create_task(recv_loop())

    # ---- open-loop blast ----
    def send_credit(i: int) -> None:
        payload = _ENC.encode(Credit(id=i, ts=_now()))
        if pattern == "dealer_router":
            ident = targets[0]
            targets.rotate(-1)
            if mode == "fd":
                zmq.Socket.send(tx, ident, flags=zmq.NOBLOCK | zmq.SNDMORE, copy=False)
                zmq.Socket.send(tx, payload, flags=zmq.NOBLOCK, copy=False)
            else:
                return ident, payload
        else:
            if mode == "fd":
                zmq.Socket.send(tx, payload, flags=zmq.NOBLOCK, copy=False)
            else:
                return None, payload
        return None

    times["first_send"] = _now()
    for i in range(count):
        if mode == "fd":
            send_credit(i)
        else:
            framed = send_credit(i)
            if pattern == "dealer_router":
                await tx.send_multipart([framed[0], framed[1]])
            else:
                await tx.send(framed[1])
        if i % YIELD_INTERVAL == 0:
            await asyncio.sleep(0)  # let returns drain concurrently (open-loop)

    await done.wait()
    if reader is not None:
        reader.stop()
    if recv_task is not None:
        recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await recv_task

    elapsed = (times["last_recv"] - times["first_send"]) / 1e9
    result["count"] = len(lat)
    result["elapsed"] = elapsed
    result["cps"] = len(lat) / elapsed if elapsed else 0.0
    result["lat"] = _pcts(lat)
    with contextlib.suppress(Exception):
        rx.close(0)
        if tx is not rx:
            tx.close(0)


def _producer_main(mode, pattern, addrs, n_workers, count, q):
    gc.disable()
    result: dict = {}
    _run(_producer(mode, pattern, addrs, n_workers, count, result))
    q.put(result)


def run_roundtrip(mode, pattern, workers, count) -> dict:
    ctx = mp.get_context("spawn")
    stamp = int(time.time() * 1000)
    if pattern == "dealer_router":
        addrs = (f"ipc:///tmp/cb_dr_{mode}_{stamp}.ipc",)
    else:
        addrs = (
            f"ipc:///tmp/cb_pp_d_{mode}_{stamp}.ipc",
            f"ipc:///tmp/cb_pp_u_{mode}_{stamp}.ipc",
        )
    q = ctx.Queue()
    # split the blast across workers so each echoes ~count/workers
    per = count // workers
    pp = ctx.Process(
        target=_producer_main, args=(mode, pattern, addrs, workers, per * workers, q)
    )
    pp.start()
    time.sleep(0.3)
    wps = [
        ctx.Process(
            target=_worker_main, args=(mode, pattern, addrs, f"w{i}", count)
        )  # worker tolerates >= its share
        for i in range(workers)
    ]
    for w in wps:
        w.start()
    result = q.get(timeout=300)
    pp.join(timeout=10)
    for w in wps:
        w.terminate()
    for w in wps:
        w.join(timeout=5)
    return result


# =============================================================================
# PUB/SUB: one PUB blasts COUNT timestamped msgs; SUBs measure one-way latency.
# =============================================================================
_TOPIC = b"x\xff"


async def _pub(mode, addr, count):
    ctx = zmq.asyncio.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 0)
    pub.bind(addr)
    await asyncio.sleep(1.0)  # slow-joiner
    for i in range(count):
        payload = _ENC.encode(Credit(id=i, ts=_now()))
        if mode == "fd":
            zmq.Socket.send(pub, _TOPIC, flags=zmq.NOBLOCK | zmq.SNDMORE, copy=False)
            zmq.Socket.send(pub, payload, flags=zmq.NOBLOCK, copy=False)
        else:
            await pub.send_multipart([_TOPIC, payload])
        if i % YIELD_INTERVAL == 0:
            await asyncio.sleep(0)
    await asyncio.sleep(2.0)  # let subs drain
    pub.close(0)


def _pub_main(mode, addr, count):
    gc.disable()
    _run(_pub(mode, addr, count))


async def _sub(mode, addr, count, result):
    ctx = zmq.asyncio.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 0)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(addr)
    lat: list[int] = []
    done = asyncio.Event()
    times = {"first": 0, "last": 0}

    def on_msg(c: Credit) -> None:
        now = _now()
        if not times["first"]:
            times["first"] = now
        lat.append(now - c.ts)
        times["last"] = now
        if len(lat) >= count:
            done.set()

    if mode == "fd":

        def recv_one():
            _t, payload = _recv_mp(sub)
            return _DEC.decode(payload)

        reader = FdEdgeReader(
            socket=sub, recv_one=recv_one, dispatch=on_msg, batch_limit=YIELD_INTERVAL
        )
        reader.start()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(done.wait(), timeout=60)
        reader.stop()
    else:
        n = 0
        try:
            while not done.is_set():
                c = _DEC.decode((await asyncio.wait_for(sub.recv_multipart(), 60))[-1])
                on_msg(c)
                n += 1
                if n % YIELD_INTERVAL == 0:
                    await asyncio.sleep(0)
        except TimeoutError:
            pass
    elapsed = (times["last"] - times["first"]) / 1e9
    result["count"] = len(lat)
    result["elapsed"] = elapsed
    result["cps"] = len(lat) / elapsed if elapsed else 0.0
    result["lat"] = _pcts(lat)
    sub.close(0)


def _sub_main(mode, addr, count, q):
    gc.disable()
    result: dict = {}
    _run(_sub(mode, addr, count, result))
    q.put(result)


def run_pubsub(mode, workers, count) -> dict:
    ctx = mp.get_context("spawn")
    addr = f"ipc:///tmp/cb_ps_{mode}_{int(time.time() * 1000)}.ipc"
    q = ctx.Queue()
    subs = [
        ctx.Process(target=_sub_main, args=(mode, addr, count, q))
        for _ in range(workers)
    ]
    pub = ctx.Process(target=_pub_main, args=(mode, addr, count))
    pub.start()
    for s in subs:
        s.start()
    results = [q.get(timeout=120) for _ in range(workers)]
    pub.join(timeout=10)
    for s in subs:
        s.terminate()
    for s in subs:
        s.join(timeout=5)
    agg = sum(r["cps"] for r in results)
    # report aggregate throughput, and the worst (max) per-sub latency percentiles
    best = max(results, key=lambda r: r["count"])
    return {
        "count": sum(r["count"] for r in results),
        "elapsed": best["elapsed"],
        "cps": agg,
        "lat": best["lat"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["await", "fd"], required=True)
    ap.add_argument(
        "--pattern",
        choices=["dealer_router", "push_pull", "pubsub"],
        default="dealer_router",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--count", type=int, default=250_000)
    args = ap.parse_args()
    if args.pattern == "pubsub":
        r = run_pubsub(args.mode, args.workers, args.count)
    else:
        r = run_roundtrip(args.mode, args.pattern, args.workers, args.count)
    lat = r.get("lat", {})
    unit = "msgs/s (agg recv, 1-way lat)" if args.pattern == "pubsub" else "credit RT/s"
    print(
        f"pattern={args.pattern} mode={args.mode} workers={args.workers} "
        f"count={args.count} => {r['cps']:,.0f} {unit} | "
        f"lat ms p50={lat.get('p50', 0):.3f} p99={lat.get('p99', 0):.3f} "
        f"p999={lat.get('p999', 0):.3f} max={lat.get('max', 0):.1f} "
        f"(got {r['count']:,} in {r['elapsed']:.2f}s)"
    )


if __name__ == "__main__":
    main()
