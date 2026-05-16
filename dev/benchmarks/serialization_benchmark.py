#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Serialization benchmark: JSON (orjson) vs MessagePack (msgspec).

Tests if serialization is a bottleneck in the receive path.
"""

import asyncio
import multiprocessing as mp
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import contextlib

from clients.msgpack_clients import ZMQPullClientMsgpack, ZMQPushClientMsgpack
from clients.yield_interval_clients import ZMQPullClientYieldTest

from aiperf.common.enums import LifecycleState, MessageType
from aiperf.common.messages import HeartbeatMessage
from aiperf.zmq.push_client import ZMQPushClient

# ============================================================================
# Sender Processes
# ============================================================================


async def sender_json(sender_id, ipc_addr, num_messages):
    """Sender using JSON serialization."""
    client = None
    try:
        await asyncio.sleep(2.0)
        client = ZMQPushClient(address=ipc_addr, bind=False)
        await client.initialize()

        for i in range(num_messages):
            msg = HeartbeatMessage(
                service_id=f"w{sender_id}",
                state=LifecycleState.RUNNING,
                service_type="bench",
                request_id=f"w{sender_id:03d}-{i:04d}",
            )
            await client.push(msg)

        await asyncio.sleep(0.1)
    finally:
        if client:
            with contextlib.suppress(BaseException):
                await client.stop()


async def sender_msgpack(sender_id, ipc_addr, num_messages):
    """Sender using msgpack serialization."""
    client = None
    try:
        await asyncio.sleep(2.0)
        client = ZMQPushClientMsgpack(address=ipc_addr, bind=False)
        await client.initialize()

        for i in range(num_messages):
            msg = HeartbeatMessage(
                service_id=f"w{sender_id}",
                state=LifecycleState.RUNNING,
                service_type="bench",
                request_id=f"w{sender_id:03d}-{i:04d}",
            )
            await client.push(msg)

        await asyncio.sleep(0.1)
    finally:
        if client:
            with contextlib.suppress(BaseException):
                await client.stop()


# ============================================================================
# Receiver Processes
# ============================================================================


async def receiver_json(ipc_addr, total_messages, yield_interval):
    """Receiver using JSON serialization."""
    client = None
    try:
        received = [0]
        blocked = [0]
        max_block = [0.0]
        monitoring = [True]

        async def monitor():
            while monitoring[0]:
                start = time.perf_counter()
                await asyncio.sleep(0.001)
                delay = (time.perf_counter() - start) * 1000
                if delay > 10.0:
                    blocked[0] += 1
                    max_block[0] = max(max_block[0], delay)

        async def handler(msg):
            received[0] += 1

        client = ZMQPullClientYieldTest(
            address=ipc_addr, bind=True, yield_interval=yield_interval
        )
        client.register_pull_callback(MessageType.HEARTBEAT, handler)
        await client.initialize()
        await client.start()

        monitor_task = asyncio.create_task(monitor())
        start = time.perf_counter()

        while received[0] < total_messages and (time.perf_counter() - start) < 60:
            await asyncio.sleep(0.5)

        duration = time.perf_counter() - start
        monitoring[0] = False
        monitor_task.cancel()
        with contextlib.suppress(BaseException):
            await monitor_task

        return {
            "count": received[0],
            "rate": received[0] / duration if duration > 0 else 0,
            "blocked": blocked[0],
            "max_block": max_block[0],
        }
    finally:
        if client:
            with contextlib.suppress(BaseException):
                await client.stop()


async def receiver_msgpack(ipc_addr, total_messages, yield_interval):
    """Receiver using msgpack serialization."""
    client = None
    try:
        received = [0]
        blocked = [0]
        max_block = [0.0]
        monitoring = [True]

        async def monitor():
            while monitoring[0]:
                start = time.perf_counter()
                await asyncio.sleep(0.001)
                delay = (time.perf_counter() - start) * 1000
                if delay > 10.0:
                    blocked[0] += 1
                    max_block[0] = max(max_block[0], delay)

        async def handler(msg):
            received[0] += 1

        client = ZMQPullClientMsgpack(
            address=ipc_addr, bind=True, yield_interval=yield_interval
        )
        client.register_pull_callback(MessageType.HEARTBEAT, handler)
        await client.initialize()
        await client.start()

        monitor_task = asyncio.create_task(monitor())
        start = time.perf_counter()

        while received[0] < total_messages and (time.perf_counter() - start) < 60:
            await asyncio.sleep(0.5)

        duration = time.perf_counter() - start
        monitoring[0] = False
        monitor_task.cancel()
        with contextlib.suppress(BaseException):
            await monitor_task

        return {
            "count": received[0],
            "rate": received[0] / duration if duration > 0 else 0,
            "blocked": blocked[0],
            "max_block": max_block[0],
        }
    finally:
        if client:
            with contextlib.suppress(BaseException):
                await client.stop()


# ============================================================================
# Process Entry Points
# ============================================================================


def run_sender_json(sid, ipc, num, shared_dict, idx):
    result = asyncio.run(sender_json(sid, ipc, num))
    shared_dict[f"sender_{idx}"] = result


def run_sender_msgpack(sid, ipc, num, shared_dict, idx):
    result = asyncio.run(sender_msgpack(sid, ipc, num))
    shared_dict[f"sender_{idx}"] = result


def run_receiver_json(ipc, total, yield_int, shared_dict):
    result = asyncio.run(receiver_json(ipc, total, yield_int))
    shared_dict["receiver"] = result


def run_receiver_msgpack(ipc, total, yield_int, shared_dict):
    result = asyncio.run(receiver_msgpack(ipc, total, yield_int))
    shared_dict["receiver"] = result


# ============================================================================
# Test Runner
# ============================================================================


def run_single_test(
    serialization, yield_interval, num_workers, msgs_per_worker, tmpdir, manager
):
    """Run test with specified serialization format."""
    ipc = f"ipc://{tmpdir}/test_{serialization}_{yield_interval}_{time.time()}.ipc"
    total = num_workers * msgs_per_worker

    shared_results = manager.dict()

    # Choose sender/receiver based on serialization
    if serialization == "json":
        rx_target = run_receiver_json
        tx_target = run_sender_json
    else:  # msgpack
        rx_target = run_receiver_msgpack
        tx_target = run_sender_msgpack

    # Start receiver
    rx_proc = mp.Process(
        target=rx_target,
        args=(ipc, total, yield_interval, shared_results),
    )
    rx_proc.start()
    time.sleep(2.0)

    # Start all workers
    tx_procs = []
    for i in range(num_workers):
        p = mp.Process(
            target=tx_target,
            args=(i, ipc, msgs_per_worker, shared_results, i),
        )
        tx_procs.append(p)
        p.start()

    # Wait for completion
    for p in tx_procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    rx_proc.join(timeout=60)
    if rx_proc.is_alive():
        rx_proc.terminate()
        rx_proc.join(timeout=5)

    # Get results
    result = None
    if "receiver" in shared_results:
        with contextlib.suppress(BaseException):
            result = dict(shared_results["receiver"])

    return result if result else {}


def main():
    """Compare JSON vs msgpack serialization."""
    mp.set_start_method("spawn", force=True)

    print("=" * 100)
    print("SERIALIZATION COMPARISON - JSON (orjson) vs MessagePack (msgspec)")
    print("=" * 100)
    print("Testing both serialization formats with yield intervals: 1, 15, 30")
    print("100 workers × 1,000 messages = 100,000 messages per run")
    print("5 runs per configuration for statistical significance")
    print("=" * 100)

    tmpdir = tempfile.mkdtemp(prefix="zmq_serial_bench_")
    print(f"\nTemp directory: {tmpdir}\n", flush=True)

    try:
        with mp.Manager() as manager:
            all_results = {}

            # Test key yield intervals with both serializations
            for yield_int in [1, 15, 30]:
                print(f"\n{'=' * 100}")
                print(f"Testing Yield Interval = {yield_int}")
                print(f"{'=' * 100}\n")

                for serialization in ["json", "msgpack"]:
                    print(f"{serialization.upper()}:", flush=True)
                    runs = []

                    for run in range(5):
                        print(f"  Run {run + 1}/5: Starting...", flush=True)

                        result = run_single_test(
                            serialization, yield_int, 100, 1000, tmpdir, manager
                        )

                        if result and result.get("count", 0) > 0:
                            runs.append(result)
                            print(
                                f"  Run {run + 1}: {result['rate']:8,.0f} msg/s, "
                                f"blocked={result['blocked']:3d}x, "
                                f"max_block={result['max_block']:6.1f}ms",
                                flush=True,
                            )
                        else:
                            print(f"  Run {run + 1}: FAILED", flush=True)

                        time.sleep(2)

                    if runs:
                        rates = [r["rate"] for r in runs]
                        stats = {
                            "serialization": serialization,
                            "yield": yield_int,
                            "avg_rate": statistics.mean(rates),
                            "min_rate": min(rates),
                            "max_rate": max(rates),
                            "stddev": statistics.stdev(rates) if len(rates) > 1 else 0,
                            "avg_blocked": statistics.mean(
                                [r["blocked"] for r in runs]
                            ),
                        }

                        all_results[f"{serialization}_yield_{yield_int}"] = stats

                        print(
                            f"  → Avg: {stats['avg_rate']:,.0f} msg/s (±{stats['stddev']:.0f}), "
                            f"{stats['avg_blocked']:.1f} blocks",
                            flush=True,
                        )

    finally:
        try:
            shutil.rmtree(tmpdir)
            print(f"\n✅ Cleaned up: {tmpdir}")
        except Exception:
            pass

    # Print comparison
    print("\n" + "=" * 100)
    print("COMPARISON: JSON vs MessagePack")
    print("=" * 100)

    print(
        f"\n{'Yield':>5} | {'Format':>8} | {'Avg Rate':>12} | {'Min':>12} | {'Max':>12} | {'StdDev':>10} | {'Blocked':>8}"
    )
    print("-" * 100)

    for yield_int in [1, 15, 30]:
        json_key = f"json_yield_{yield_int}"
        msgpack_key = f"msgpack_yield_{yield_int}"

        if json_key in all_results:
            s = all_results[json_key]
            print(
                f"{yield_int:5d} | {'JSON':>8} | {s['avg_rate']:10,.0f} msg/s | "
                f"{s['min_rate']:10,.0f} | {s['max_rate']:10,.0f} | {s['stddev']:10.0f} | "
                f"{s['avg_blocked']:8.1f}x"
            )

        if msgpack_key in all_results:
            s = all_results[msgpack_key]
            marker = ""

            # Calculate improvement if we have JSON to compare
            if json_key in all_results:
                improvement = (
                    (s["avg_rate"] - all_results[json_key]["avg_rate"])
                    / all_results[json_key]["avg_rate"]
                    * 100
                )
                marker = (
                    f" (+{improvement:.1f}%)"
                    if improvement > 0
                    else f" ({improvement:.1f}%)"
                )

            print(
                f"{yield_int:5d} | {'msgpack':>8} | {s['avg_rate']:10,.0f} msg/s | "
                f"{s['min_rate']:10,.0f} | {s['max_rate']:10,.0f} | {s['stddev']:10.0f} | "
                f"{s['avg_blocked']:8.1f}x{marker}"
            )

        print("")

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Find best of each
    json_results = {k: v for k, v in all_results.items() if k.startswith("json_")}
    msgpack_results = {k: v for k, v in all_results.items() if k.startswith("msgpack_")}

    if json_results and msgpack_results:
        best_json = max(json_results.values(), key=lambda s: s["avg_rate"])
        best_msgpack = max(msgpack_results.values(), key=lambda s: s["avg_rate"])

        print(f"\nBest JSON (yield={best_json['yield']}):")
        print(f"  {best_json['avg_rate']:,.0f} msg/s (±{best_json['stddev']:.0f})")

        print(f"\nBest MessagePack (yield={best_msgpack['yield']}):")
        print(
            f"  {best_msgpack['avg_rate']:,.0f} msg/s (±{best_msgpack['stddev']:.0f})"
        )

        improvement = (
            (best_msgpack["avg_rate"] - best_json["avg_rate"])
            / best_json["avg_rate"]
            * 100
        )

        print("\nMessagePack vs JSON:")
        print(f"  Improvement: {improvement:+.1f}%")

        if abs(improvement) < 5:
            print(
                "  Verdict: Minimal difference - serialization is NOT a major bottleneck"
            )
        elif improvement > 10:
            print("  Verdict: MessagePack provides significant improvement!")
        else:
            print("  Verdict: MessagePack provides moderate improvement")

    print("=" * 100)
    print("\n✅ Serialization benchmark complete!")


if __name__ == "__main__":
    main()
