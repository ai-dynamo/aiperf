from __future__ import annotations

import os
import random
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

from tests.scripts.chaos.harness import CRASH_MARKERS, Case, Context

DEFAULT_SEED = 0xA1
DEFAULT_MAX_EXAMPLES = 10
PER_EXAMPLE_TIMEOUT_SECONDS = 30


def _seed() -> int:
    raw = os.environ.get("AIPERF_FUZZ_SEED")
    return int(raw, 0) if raw else DEFAULT_SEED


def _max_examples() -> int:
    raw = os.environ.get("AIPERF_FUZZ_MAX_EXAMPLES")
    return int(raw) if raw else DEFAULT_MAX_EXAMPLES


def _numeric_args(rng: random.Random) -> list[str]:
    pool: list[Callable[[], list[str]]] = [
        lambda: ["--request-count", str(rng.randint(-5, 100))],
        lambda: ["--concurrency", str(rng.choice([-1, 0, 1, 2, 100, 10_000]))],
        lambda: ["--request-rate", str(rng.choice([-1.5, 0, 0.5, 5, 1e9]))],
        lambda: ["--benchmark-duration", str(rng.choice([-1, 0, 1, 3600]))],
        lambda: ["--num-conversations", str(rng.choice([-1, 0, 1, 1000]))],
        lambda: ["--warmup-request-count", str(rng.choice([-1, 0, 5]))],
        lambda: ["--request-cancellation-rate", str(rng.choice([-1, 0, 50, 101, 999]))],
    ]
    chosen = rng.sample(pool, k=rng.randint(1, len(pool)))
    args: list[str] = ["--endpoint-type", "chat"]
    for build in chosen:
        args.extend(build())
    return args


_MUTUALLY_EXCLUSIVE_POOL: list[list[str]] = [
    ["--public-dataset", "sharegpt"],
    [
        "--input-file",
        "/tmp/aiperf_fuzz_missing.jsonl",
        "--custom-dataset-type",
        "single-turn",
    ],
    ["--fixed-schedule"],
    ["--fixed-schedule-auto-offset"],
    ["--fixed-schedule-start-offset", "0"],
    ["--fixed-schedule-end-offset", "0"],
    ["--num-prefix-prompts", "2", "--prefix-prompt-length", "8"],
    ["--shared-system-prompt-length", "4"],
    ["--streaming"],
    ["--use-legacy-max-tokens"],
    ["--gpu-telemetry", "dashboard"],
    ["--no-gpu-telemetry"],
]


def _flag_combos(rng: random.Random) -> list[str]:
    pieces = rng.sample(_MUTUALLY_EXCLUSIVE_POOL, k=rng.randint(2, 5))
    args: list[str] = [
        "--endpoint-type",
        "chat",
        "--request-count",
        "1",
        "--concurrency",
        "1",
    ]
    for piece in pieces:
        args.extend(piece)
    return args


_YAML_BODIES: list[str] = [
    "model: mock-model\nendpoint:\n  urls: [{url}]\n  type: chat\n  unknownNested: yes\n",
    "model: mock-model\nendpoint:\n  urls: [{url}]\n  type: chat\nphases:\n  type: concurrency\n  concurrency: -1\n  requests: 1\n",
    "model: mock-model\nendpoint:\n  urls: [{url}]\n  type: not-a-real-type\n",
    "model: mock-model\nendpoint:\n  urls: [{url}]\n  type: chat\ndataset:\n  type: synthetic\n  prompts:\n    isl: -5\n    osl: 0\n",
    "model: mock-model\nendpoint:\n  urls: [{url}]\n  type: template\n  path: /v1/x\n  template:\n    body: 'not jinja'\n    responseField: ''\n",
    "model: mock-model\nendpoint:\n  urls: [{url}]\n  type: chat\nphases:\n  type: concurrency\n  concurrency: 9999999999999\n  requests: 9999999999999\n",
]


def _config_yaml(rng: random.Random, ctx: Context) -> tuple[Path, list[str]]:
    template = rng.choice(_YAML_BODIES)
    cfg = ctx.fixtures / f"fuzz_config_{rng.randint(0, 1_000_000):06d}.yaml"
    cfg.write_text(template.format(url=ctx.url))
    return cfg, ["uv", "run", "aiperf", "config", "validate", "--path", str(cfg)]


def _run_one(cmd: list[str], ctx: Context, log: Path, header: str) -> tuple[int, str]:
    with log.open("a") as out:
        out.write(f"\n--- {header} ---\n$ {shlex.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            cwd=ctx.base,
            env=ctx.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = proc.communicate(timeout=PER_EXAMPLE_TIMEOUT_SECONDS)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            import os as _os
            import signal as _signal

            _os.killpg(proc.pid, _signal.SIGKILL)
            proc.wait(timeout=2)
            stdout = "TIMEOUT"
            rc = 124
        if any(marker in stdout for marker in CRASH_MARKERS):
            out.write(stdout)
            out.write(f"\nrc={rc} CRASH_DETECTED\n")
            return rc, stdout
        out.write(f"rc={rc} (no crash markers)\n")
    return rc, stdout


def _fuzz_runner(
    arg_factory: Callable[[random.Random, Context], tuple[list[str], list[str]]],
    label: str,
) -> Callable[[Context, str, Path], tuple[int, str]]:
    def _run(ctx: Context, name: str, log: Path) -> tuple[int, str]:
        seed = _seed()
        rng = random.Random(seed)
        max_examples = _max_examples()
        log.write_text(f"FUZZ {label} seed={seed} max_examples={max_examples}\n")
        crash_count = 0
        for idx in range(max_examples):
            base, extra = arg_factory(rng, ctx)
            cmd = base + extra
            _, stdout = _run_one(cmd, ctx, log, f"example {idx + 1}/{max_examples}")
            if any(marker in stdout for marker in CRASH_MARKERS):
                crash_count += 1
        if crash_count:
            with log.open("a") as out:
                out.write(
                    f"\nFUZZ_SUMMARY: {crash_count}/{max_examples} examples crashed\n"
                )
            return 1, log.read_text(errors="replace")
        with log.open("a") as out:
            out.write(f"\nFUZZ_SUMMARY: 0/{max_examples} examples crashed\n")
        return 0, log.read_text(errors="replace")

    return _run


def _profile_arg_factory(name: str):
    def _factory(rng: random.Random, ctx: Context) -> tuple[list[str], list[str]]:
        base = [
            "uv",
            "run",
            "aiperf",
            "profile",
            "--model",
            "mock-model",
            "--url",
            ctx.url,
            "--tokenizer",
            "builtin",
            "--ui",
            "none",
            "--request-timeout-seconds",
            "5",
            "--wait-for-model-timeout",
            "0",
            "--workers-max",
            "1",
            "--no-gpu-telemetry",
            "--artifact-dir",
            str(ctx.artifacts / f"{name}-{rng.randint(0, 1_000_000):06d}"),
        ]
        extra = _numeric_args(rng) if name == "fuzz-numeric-args" else _flag_combos(rng)
        return base, extra

    return _factory


def _config_arg_factory() -> Callable[
    [random.Random, Context], tuple[list[str], list[str]]
]:
    def _factory(rng: random.Random, ctx: Context) -> tuple[list[str], list[str]]:
        _, cmd = _config_yaml(rng, ctx)
        return cmd, []

    return _factory


def build_fuzz_cases() -> list[Case]:
    return [
        Case(
            name="fuzz-numeric-args",
            expected="PASS_REQUIRED",
            run=_fuzz_runner(_profile_arg_factory("fuzz-numeric-args"), "numeric-args"),
            why="random numeric CLI values must never crash; graceful failure or success only",
        ),
        Case(
            name="fuzz-flag-combos",
            expected="PASS_REQUIRED",
            run=_fuzz_runner(_profile_arg_factory("fuzz-flag-combos"), "flag-combos"),
            why="random mutually-exclusive flag combos must fail gracefully without crash",
        ),
        Case(
            name="fuzz-config-yaml",
            expected="PASS_REQUIRED",
            run=_fuzz_runner(_config_arg_factory(), "config-yaml"),
            why="random invalid config YAML inputs must reject without crash",
        ),
    ]
