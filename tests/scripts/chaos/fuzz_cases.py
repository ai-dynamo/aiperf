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
    # Legacy flat shape (kept for back-compat coverage)
    "model: mock-model\nendpoint:\n  urls: [{url}]\n  type: chat\n  unknownNested: yes\n",
    "model: mock-model\nendpoint:\n  urls: [{url}]\n  type: chat\nphases:\n  type: concurrency\n  concurrency: -1\n  requests: 1\n",
    "model: mock-model\nendpoint:\n  urls: [{url}]\n  type: not-a-real-type\n",
    "model: mock-model\nendpoint:\n  urls: [{url}]\n  type: chat\ndataset:\n  type: synthetic\n  prompts:\n    isl: -5\n    osl: 0\n",
    "model: mock-model\nendpoint:\n  urls: [{url}]\n  type: template\n  path: /v1/x\n  template:\n    body: 'not jinja'\n    responseField: ''\n",
    "model: mock-model\nendpoint:\n  urls: [{url}]\n  type: chat\nphases:\n  type: concurrency\n  concurrency: 9999999999999\n  requests: 9999999999999\n",
    # Schema-v2 envelope shape, malformed in different dimensions
    'schemaVersion: "2.0"\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n    UNKNOWN_NESTED_FIELD: yes\n  dataset:\n    type: synthetic\n    entries: 1\n    prompts:\n      isl: 8\n      osl: 4\n  phases:\n    type: concurrency\n    concurrency: 1\n    requests: 1\n',
    'schemaVersion: "2.0"\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n  dataset:\n    type: synthetic\n    entries: 1\n    prompts:\n      isl: -5\n      osl: 0\n  phases:\n    type: concurrency\n    concurrency: 1\n    requests: 1\n',
    'schemaVersion: "2.0"\nsweep:\n  type: grid\n  parameters:\n    "phases.profiling.rate": []\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n  dataset:\n    type: synthetic\n    entries: 1\n    prompts:\n      isl: 8\n      osl: 4\n  phases:\n    type: concurrency\n    concurrency: 1\n    requests: 1\n',
    'schemaVersion: "2.0"\nsweep:\n  type: not-a-real-sweep-type\n  parameters:\n    "phases.profiling.rate": [1.0]\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n  dataset:\n    type: synthetic\n    entries: 1\n    prompts:\n      isl: 8\n      osl: 4\n  phases:\n    type: concurrency\n    concurrency: 1\n    requests: 1\n',
    'schemaVersion: "2.0"\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n  dataset:\n    type: definitely-not-a-real-dataset-type\n  phases:\n    type: concurrency\n    concurrency: 1\n    requests: 1\n',
    'schemaVersion: "2.0"\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n  dataset:\n    type: synthetic\n    entries: 1\n    prompts:\n      isl: 8\n      osl: 4\n  phases:\n    type: concurrency\n    concurrency: 1\n    requests: 1\nplot:\n  not_a_real_plot_field: true\n',
    'schemaVersion: "2.0"\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n  dataset:\n    type: synthetic\n    entries: 1\n    prompts:\n      isl: 8\n      osl: 4\n  phases:\n    type: concurrency\n    concurrency: 1\n    requests: 1\nplot: /tmp/aiperf_chaos_definitely_missing.yaml\n',
    'schemaVersion: "2.0"\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n  dataset:\n    type: synthetic\n    entries: 1\n    prompts:\n      isl: 8\n      osl: 4\n  phases:\n    type: poisson\n    rate: -1.0\n    duration: -5\n',
    'schemaVersion: "2.0"\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n    timeout: -1.0\n  dataset:\n    type: synthetic\n    entries: 1\n    prompts:\n      isl: 8\n      osl: 4\n  phases:\n    type: concurrency\n    concurrency: 1\n    requests: 1\n',
    'schema_version: "99.0"\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n  dataset:\n    type: synthetic\n    entries: 1\n    prompts:\n      isl: 8\n      osl: 4\n  phases:\n    type: concurrency\n    concurrency: 1\n    requests: 1\n',
    'schemaVersion: "2.0"\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n  dataset:\n    type: synthetic\n    entries: 1\n    prompts:\n      isl: .nan\n      osl: .inf\n  phases:\n    type: concurrency\n    concurrency: 1\n    requests: 1\n',
    'schemaVersion: "2.0"\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n  dataset:\n    type: synthetic\n    entries: 1\n    prompts:\n      isl: 8\n      osl: 4\n  phases:\n    type: concurrency\n    concurrency: 1\n    requests: 1\nUNKNOWN_TOP_LEVEL_KEY: 42\n',
    'schemaVersion: "2.0"\nmulti_run:\n  numRuns: -1\n  cooldownSeconds: -5\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n  dataset:\n    type: synthetic\n    entries: 1\n    prompts:\n      isl: 8\n      osl: 4\n  phases:\n    type: concurrency\n    concurrency: 1\n    requests: 1\n',
    'schemaVersion: "2.0"\nbenchmark:\n  model: mock-model\n  endpoint:\n    url: {url}\n    type: chat\n  dataset:\n    type: file\n    records: []\n  phases:\n    type: concurrency\n    concurrency: 1\n    requests: 1\n',
]


_JINJA_FRAGMENTS: list[str] = [
    "{{ unclosed",
    "{% if true %}only-open",
    "{{ undefined_variable_xyz }}",
    "{% for x in [1,2,3 %}{{ x }}",
    "{{ ''.__class__.__mro__ }}",
    "{# unclosed comment",
    "{{ }}",
    "{{ a + }}",
]


def _jinja_fuzz_yaml(rng: random.Random, ctx: Context) -> tuple[Path, list[str]]:
    fragment = rng.choice(_JINJA_FRAGMENTS)
    body = (
        f'schemaVersion: "2.0"\n'
        f"benchmark:\n"
        f'  model: "mock-{fragment}"\n'
        f"  endpoint:\n"
        f"    url: {ctx.url}\n"
        f"    type: chat\n"
        f"  dataset:\n"
        f"    type: synthetic\n"
        f"    entries: 1\n"
        f"    prompts:\n"
        f"      isl: 8\n"
        f"      osl: 4\n"
        f"  phases:\n"
        f"    type: concurrency\n"
        f"    concurrency: 1\n"
        f"    requests: 1\n"
    )
    cfg = ctx.fixtures / f"fuzz_jinja_{rng.randint(0, 1_000_000):06d}.yaml"
    cfg.write_text(body)
    return cfg, ["uv", "run", "aiperf", "config", "validate", str(cfg)]


_ENVVAR_PATTERNS: list[str] = [
    "${DEFINITELY_UNSET_FUZZ_VAR_XYZZY}",
    "${DEFINITELY_UNSET_FUZZ_VAR:fallback}",
    "${DEFINITELY_UNSET_FUZZ_VAR:}",
    "${1INVALID_IDENT}",
    "${VAR-WITH-DASHES}",
    "${UNTERMINATED",
    "${NESTED_${INNER}}",
    "${}",
]


def _envvar_fuzz_yaml(rng: random.Random, ctx: Context) -> tuple[Path, list[str]]:
    pattern = rng.choice(_ENVVAR_PATTERNS)
    target_field = rng.choice(["model", "concurrency", "url"])
    if target_field == "concurrency":
        concurrency = f'"{pattern}"'
        model = "mock-model"
        url = ctx.url
    elif target_field == "url":
        concurrency = "1"
        model = "mock-model"
        url = pattern
    else:
        concurrency = "1"
        model = f'"mock-{pattern}"'
        url = ctx.url
    body = (
        f'schemaVersion: "2.0"\n'
        f"benchmark:\n"
        f"  model: {model}\n"
        f"  endpoint:\n"
        f"    url: {url}\n"
        f"    type: chat\n"
        f"  dataset:\n"
        f"    type: synthetic\n"
        f"    entries: 1\n"
        f"    prompts:\n"
        f"      isl: 8\n"
        f"      osl: 4\n"
        f"  phases:\n"
        f"    type: concurrency\n"
        f"    concurrency: {concurrency}\n"
        f"    requests: 1\n"
    )
    cfg = ctx.fixtures / f"fuzz_envvar_{rng.randint(0, 1_000_000):06d}.yaml"
    cfg.write_text(body)
    return cfg, ["uv", "run", "aiperf", "config", "validate", str(cfg)]


_SWEEP_PATH_PATTERNS: list[str] = [
    "",
    ".phases.profiling.rate",
    "phases.profiling.rate.",
    "phases..profiling.rate",
    "benchmark.phases.profiling.rate",
    "sweep.something",
    "multi_run.numRuns",
    "random_seed",
    "phases.profiling.NOT_A_REAL_FIELD",
    "datasets.default.prompts.NOT_A_REAL",
    "....",
    "phases.profiling.rate.subfield.too.deep",
]


def _sweep_path_fuzz_yaml(rng: random.Random, ctx: Context) -> tuple[Path, list[str]]:
    path = rng.choice(_SWEEP_PATH_PATTERNS)
    values = rng.choice(["[1.0, 2.0, 3.0]", "[1, 2]", '["a"]', "[]"])
    body = (
        f'schemaVersion: "2.0"\n'
        f"sweep:\n"
        f"  type: grid\n"
        f"  parameters:\n"
        f'    "{path}": {values}\n'
        f"benchmark:\n"
        f"  model: mock-model\n"
        f"  endpoint:\n"
        f"    url: {ctx.url}\n"
        f"    type: chat\n"
        f"  dataset:\n"
        f"    type: synthetic\n"
        f"    entries: 1\n"
        f"    prompts:\n"
        f"      isl: 8\n"
        f"      osl: 4\n"
        f"  phases:\n"
        f"    type: concurrency\n"
        f"    concurrency: 1\n"
        f"    requests: 1\n"
    )
    cfg = ctx.fixtures / f"fuzz_sweep_path_{rng.randint(0, 1_000_000):06d}.yaml"
    cfg.write_text(body)
    return cfg, ["uv", "run", "aiperf", "config", "validate", str(cfg)]


def _config_yaml(rng: random.Random, ctx: Context) -> tuple[Path, list[str]]:
    template = rng.choice(_YAML_BODIES)
    cfg = ctx.fixtures / f"fuzz_config_{rng.randint(0, 1_000_000):06d}.yaml"
    cfg.write_text(template.format(url=ctx.url))
    return cfg, ["uv", "run", "aiperf", "config", "validate", str(cfg)]


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


def _jinja_arg_factory() -> Callable[
    [random.Random, Context], tuple[list[str], list[str]]
]:
    def _factory(rng: random.Random, ctx: Context) -> tuple[list[str], list[str]]:
        _, cmd = _jinja_fuzz_yaml(rng, ctx)
        return cmd, []

    return _factory


def _envvar_arg_factory() -> Callable[
    [random.Random, Context], tuple[list[str], list[str]]
]:
    def _factory(rng: random.Random, ctx: Context) -> tuple[list[str], list[str]]:
        _, cmd = _envvar_fuzz_yaml(rng, ctx)
        return cmd, []

    return _factory


def _sweep_path_arg_factory() -> Callable[
    [random.Random, Context], tuple[list[str], list[str]]
]:
    def _factory(rng: random.Random, ctx: Context) -> tuple[list[str], list[str]]:
        _, cmd = _sweep_path_fuzz_yaml(rng, ctx)
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
        Case(
            name="fuzz-config-v2-jinja",
            expected="PASS_REQUIRED",
            run=_fuzz_runner(_jinja_arg_factory(), "config-v2-jinja"),
            why="random malformed jinja in v2 config must reject without crash",
        ),
        Case(
            name="fuzz-config-v2-envvar",
            expected="PASS_REQUIRED",
            run=_fuzz_runner(_envvar_arg_factory(), "config-v2-envvar"),
            why="random env-var patterns in v2 config must reject without crash",
        ),
        Case(
            name="fuzz-config-v2-sweep-path",
            expected="PASS_REQUIRED",
            run=_fuzz_runner(_sweep_path_arg_factory(), "config-v2-sweep-path"),
            why="random malformed sweep dotted paths must reject without crash",
        ),
    ]
