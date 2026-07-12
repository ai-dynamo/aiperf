# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate AIPerf against the official Dynamo offline replay frontend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    """One child process measurement from ``wait4``."""

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
    parser.add_argument("--aiperf", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--official-pythonpath", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--max-delta-percent", type=float, default=5.0)
    return parser


def _command_prefix(cpu: int | None) -> list[str]:
    if cpu is None:
        return []
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("--cpu requires taskset")
    return [taskset, "-c", str(cpu)]


def _aiperf_command(args: argparse.Namespace, report: Path) -> list[str]:
    return [
        *_command_prefix(args.cpu),
        str(args.aiperf),
        "--offline",
        "--trace-file",
        str(args.trace),
        "--trace-format",
        "mooncake",
        "--trace-block-size",
        "128",
        "--replay-concurrency",
        "16",
        "--offline-topology",
        "aggregated",
        "--offline-workers",
        "1",
        "--offline-router",
        "round-robin",
        "--extra-engine-args",
        "{}",
        "--report-json",
        str(report),
    ]


def _official_command(args: argparse.Namespace, report: Path) -> list[str]:
    return [
        *_command_prefix(args.cpu),
        str(args.python),
        "-m",
        "dynamo.replay",
        str(args.trace),
        "--trace-format",
        "mooncake",
        "--trace-block-size",
        "128",
        "--replay-concurrency",
        "16",
        "--replay-mode",
        "offline",
        "--num-workers",
        "1",
        "--router-mode",
        "round_robin",
        "--extra-engine-args",
        "{}",
        "--report-json",
        str(report),
    ]


def _run_child(
    frontend: str,
    index: int,
    command: list[str],
    output_dir: Path,
    env: dict[str, str],
) -> Sample:
    stem = f"{frontend}-{index}"
    report = output_dir / f"{stem}.report.json"
    with (
        (output_dir / f"{stem}.stdout").open("wb") as stdout_file,
        (output_dir / f"{stem}.stderr").open("wb") as stderr_file,
    ):
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
        )
        _, status, usage = os.wait4(process.pid, 0)
        wall_s = time.perf_counter() - started
        returncode = os.waitstatus_to_exitcode(status)
        process.returncode = returncode

    sample = Sample(
        frontend=frontend,
        index=index,
        returncode=returncode,
        wall_s=wall_s,
        user_s=usage.ru_utime,
        system_s=usage.ru_stime,
        rss_kib=usage.ru_maxrss,
        report_path=str(report),
    )
    (output_dir / f"{stem}.time.json").write_text(
        json.dumps(asdict(sample), sort_keys=True) + "\n", encoding="utf-8"
    )
    if returncode != 0:
        raise RuntimeError(f"{frontend} sample {index} exited {returncode}")
    return sample


def _median(samples: list[Sample], attribute: str) -> float:
    return statistics.median(float(getattr(sample, attribute)) for sample in samples)


def _median_process_cpu(samples: list[Sample]) -> float:
    return statistics.median(sample.user_s + sample.system_s for sample in samples)


def _paired_deltas_percent(
    aiperf_samples: list[Sample],
    official_samples: list[Sample],
    value: Callable[[Sample], float],
) -> list[float]:
    aiperf_by_index = {sample.index: sample for sample in aiperf_samples}
    official_by_index = {sample.index: sample for sample in official_samples}
    if aiperf_by_index.keys() != official_by_index.keys():
        raise AssertionError("frontends do not have the same sample indexes")
    return [
        (value(aiperf_by_index[index]) / value(official_by_index[index]) - 1.0) * 100.0
        for index in sorted(aiperf_by_index)
    ]


def _median_absolute(values: list[float]) -> float:
    return statistics.median(abs(value) for value in values)


def main() -> int:
    args = _parser().parse_args()
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.samples < 1 or args.samples % 2 == 0:
        raise ValueError("--samples must be a positive odd number")
    if args.max_delta_percent < 0:
        raise ValueError("--max-delta-percent must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    official_env = base_env | {
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": args.official_pythonpath,
    }
    samples: dict[str, list[Sample]] = {"aiperf": [], "official": []}

    for index in range(1, args.warmups + 1):
        order = ("official", "aiperf") if index % 2 else ("aiperf", "official")
        for frontend in order:
            stem = f"warmup-{frontend}"
            report = args.output_dir / f"{stem}-{index}.report.json"
            if frontend == "aiperf":
                command = _aiperf_command(args, report)
                env = base_env
            else:
                command = _official_command(args, report)
                env = official_env
            _run_child(stem, index, command, args.output_dir, env)

    for index in range(1, args.samples + 1):
        order = ("aiperf", "official") if index % 2 else ("official", "aiperf")
        for frontend in order:
            report = args.output_dir / f"{frontend}-{index}.report.json"
            if frontend == "aiperf":
                command = _aiperf_command(args, report)
                env = base_env
            else:
                command = _official_command(args, report)
                env = official_env
            samples[frontend].append(
                _run_child(frontend, index, command, args.output_dir, env)
            )

    reports = [
        Path(sample.report_path).read_bytes()
        for frontend_samples in samples.values()
        for sample in frontend_samples
    ]
    report_hashes = {hashlib.sha256(report).hexdigest() for report in reports}
    if len(report_hashes) != 1:
        raise AssertionError(f"report bytes differ: {sorted(report_hashes)}")

    aiperf_wall = _median(samples["aiperf"], "wall_s")
    official_wall = _median(samples["official"], "wall_s")
    aiperf_process_cpu = _median_process_cpu(samples["aiperf"])
    official_process_cpu = _median_process_cpu(samples["official"])
    aiperf_rss = _median(samples["aiperf"], "rss_kib")
    official_rss = _median(samples["official"], "rss_kib")
    wall_delta_percent = (aiperf_wall / official_wall - 1.0) * 100.0
    process_cpu_delta_percent = (
        aiperf_process_cpu / official_process_cpu - 1.0
    ) * 100.0
    rss_delta_percent = (aiperf_rss / official_rss - 1.0) * 100.0
    wall_paired_deltas = _paired_deltas_percent(
        samples["aiperf"], samples["official"], lambda sample: sample.wall_s
    )
    process_cpu_paired_deltas = _paired_deltas_percent(
        samples["aiperf"],
        samples["official"],
        lambda sample: sample.user_s + sample.system_s,
    )
    rss_paired_deltas = _paired_deltas_percent(
        samples["aiperf"],
        samples["official"],
        lambda sample: float(sample.rss_kib),
    )
    summary = {
        "aiperf_process_cpu_median_s": aiperf_process_cpu,
        "aiperf_rss_median_kib": aiperf_rss,
        "aiperf_wall_median_s": aiperf_wall,
        "official_process_cpu_median_s": official_process_cpu,
        "official_rss_median_kib": official_rss,
        "official_wall_median_s": official_wall,
        "process_cpu_delta_percent": process_cpu_delta_percent,
        "process_cpu_paired_deltas_percent": process_cpu_paired_deltas,
        "process_cpu_paired_median_abs_delta_percent": _median_absolute(
            process_cpu_paired_deltas
        ),
        "report_bytes": len(reports[0]),
        "report_sha256": report_hashes.pop(),
        "rss_delta_percent": rss_delta_percent,
        "rss_paired_deltas_percent": rss_paired_deltas,
        "rss_paired_median_abs_delta_percent": _median_absolute(rss_paired_deltas),
        "samples": {
            frontend: [asdict(sample) for sample in frontend_samples]
            for frontend, frontend_samples in samples.items()
        },
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "wall_paired_deltas_percent": wall_paired_deltas,
        "wall_paired_median_abs_delta_percent": _median_absolute(wall_paired_deltas),
        "wall_delta_percent": wall_delta_percent,
        "warmups": args.warmups,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    limit = args.max_delta_percent
    wall_paired_delta = _median_absolute(wall_paired_deltas)
    if wall_paired_delta > limit:
        raise AssertionError(
            f"paired wall delta {wall_paired_delta:.3f}% exceeds {limit:.3f}%"
        )
    process_cpu_paired_delta = _median_absolute(process_cpu_paired_deltas)
    if process_cpu_paired_delta > limit:
        raise AssertionError(
            "paired process CPU delta "
            f"{process_cpu_paired_delta:.3f}% exceeds {limit:.3f}%"
        )
    rss_paired_delta = _median_absolute(rss_paired_deltas)
    if rss_paired_delta > limit:
        raise AssertionError(
            f"paired RSS delta {rss_paired_delta:.3f}% exceeds {limit:.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
