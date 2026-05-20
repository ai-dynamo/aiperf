#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Suggest test-docs-end-to-end ``weight=`` updates from observed nightly runtimes.

The matrix sharder in ``tests/ci/test_docs_end_to_end/main.py`` uses LPT
bin-packing keyed on each tutorial command's ``weight=`` annotation. Weights
are author-supplied guesses and drift as the tutorial set evolves. This script
reads the actual per-command runtimes from a finished Nightly workflow run,
matches each log entry back to the markdown bash block that produced it, and
prints a markdown table of suggested ``weight=`` updates.

Purely informational — nothing is mutated. Designed to be run as a
non-blocking CI step that writes to ``$GITHUB_STEP_SUMMARY``, or locally
against any prior run via ``--run-id <id>``.

Usage:

    python3 tools/suggest_test_docs_weights.py --run-id 26095921607
    python3 tools/suggest_test_docs_weights.py        # in CI, picks $GITHUB_RUN_ID

Exits 0 even when suggestions are emitted; it's a report, not a gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Reach the existing parser/data_types so we don't duplicate the bash-block
# extraction logic. Both modules import each other by bare name, so we have
# to extend sys.path rather than importing them as a package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests" / "ci" / "test_docs_end_to_end"))

from data_types import Command  # noqa: E402
from parser import MarkdownParser  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("suggest-weights")

DEFAULT_WEIGHT = Command.__dataclass_fields__["weight"].default
SAFETY_MARGIN = 1.3  # observed × this → suggested
ROUND_TO = 50


@dataclass
class TestRun:
    """One observed test execution from a CI shard log."""

    job_id: int
    job_name: str  # e.g. "vllm-default-openai-1of4"
    test_index: int  # 1-based within the shard
    status: str  # "passed" / "failed" / "exceeded"
    actual_seconds: int
    bash_content: str  # canonicalized for matching
    passed_line_index: int  # 1-based line number in the step log


@dataclass
class Suggestion:
    """One proposed weight change."""

    file_path: str
    start_line: int
    current_weight: int
    observed_seconds: int
    suggested_weight: int
    reason: str  # "default underweight" / "underweight" / "overweighted"
    job_log_url: str | None


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #

_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})\.")
_EXEC_BLOCK_RE = re.compile(
    r"Executing AIPerf command (?P<n>\d+)/\d+ against [\w-]+:.*?"
    r"Command:\s*(?P<body>.*?)(?:\n[^\n]*With UI flag|=========)",
    re.DOTALL,
)


def _ts_seconds(line: str) -> int | None:
    m = _TS_RE.search(line)
    if not m:
        return None
    return int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))


def _canonicalize_bash(body: str) -> str:
    """Strip log-prefix timestamps and collapse whitespace for matching."""
    stripped = re.sub(r"^[^\n]*?- INFO - ", "", body, flags=re.M)
    stripped = re.sub(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+", "", stripped, flags=re.M)
    return re.sub(r"\s+", " ", stripped).strip()


def parse_shard_log(log_text: str, job_id: int, job_name: str) -> list[TestRun]:
    """Extract test-run records from one shard's step log."""
    start_seconds: dict[int, int] = {}
    end_seconds: dict[int, int] = {}
    end_line: dict[int, int] = {}
    status: dict[int, str] = {}

    for line_idx, line in enumerate(log_text.splitlines(), start=1):
        m = re.search(r"Running AIPerf test (\d+)/", line)
        if m:
            ts = _ts_seconds(line)
            if ts is not None:
                start_seconds[int(m.group(1))] = ts
            continue
        m = re.search(r"AIPerf test (\d+) (passed|failed|exceeded)", line)
        if m:
            ts = _ts_seconds(line)
            n = int(m.group(1))
            if ts is not None:
                end_seconds[n] = ts
                end_line[n] = line_idx
                status[n] = m.group(2)

    runs: list[TestRun] = []
    for m in _EXEC_BLOCK_RE.finditer(log_text):
        n = int(m.group("n"))
        body = m.group("body")
        bash = _canonicalize_bash(body)
        s = start_seconds.get(n)
        e = end_seconds.get(n)
        if s is None or e is None:
            continue
        # Handle UTC midnight crossing within a long shard.
        actual = (e - s) % (24 * 3600)
        runs.append(
            TestRun(
                job_id=job_id,
                job_name=job_name,
                test_index=n,
                status=status.get(n, "unknown"),
                actual_seconds=actual,
                bash_content=bash,
                passed_line_index=end_line.get(n, 0),
            )
        )
    return runs


# --------------------------------------------------------------------------- #
# GH API
# --------------------------------------------------------------------------- #


def _gh(*args: str) -> str:
    """Run ``gh`` and return stdout. Errors propagate."""
    out = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return out.stdout


def fetch_test_docs_jobs(run_id: int) -> list[tuple[int, str]]:
    """Return [(job_id, name)] for every test-docs matrix shard in a run."""
    data = json.loads(
        _gh("api", f"/repos/{_repo_slug()}/actions/runs/{run_id}/jobs?per_page=100")
    )
    out = []
    for j in data.get("jobs", []):
        name = j.get("name", "")
        # Matrix-job names: "Test Docs End-to-End / Test Docs End-to-End (vllm-…)"
        if "Test Docs End-to-End / Test Docs End-to-End (" in name:
            shard_label = name.rsplit("(", 1)[-1].rstrip(")")
            out.append((j["id"], shard_label))
    return out


def fetch_job_log(job_id: int) -> str:
    return _gh("api", f"/repos/{_repo_slug()}/actions/jobs/{job_id}/logs")


def _repo_slug() -> str:
    slug = os.environ.get("GITHUB_REPOSITORY")
    if slug:
        return slug
    # Best-effort: parse from git remote
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True
        ).strip()
        m = re.search(r"[:/]([\w-]+/[\w-]+?)(?:\.git)?$", url)
        if m:
            return m.group(1)
    except subprocess.CalledProcessError:
        pass
    raise RuntimeError(
        "Cannot determine repo slug; set $GITHUB_REPOSITORY or run inside a git checkout."
    )


# --------------------------------------------------------------------------- #
# Suggestion logic
# --------------------------------------------------------------------------- #


def _round_to(n: int, step: int = ROUND_TO) -> int:
    return max(step, int(round(n / step)) * step)


def derive_suggestions(
    commands: list[Command],
    runs: list[TestRun],
    *,
    threshold: float,
    default_threshold: int,
    run_url: str,
) -> tuple[list[Suggestion], int, int]:
    """Match TestRuns to Commands, compute suggestions.

    Returns (suggestions, matched_count, unmatched_count).
    """
    # Multiple markdown blocks can canonicalize to the same bash content
    # (e.g. a tutorial example duplicated across two pages). Track every
    # Command per key so a single log entry maps to all matching blocks
    # instead of silently dropping all but one.
    by_bash: dict[str, list[Command]] = defaultdict(list)
    for c in commands:
        by_bash[_canonicalize_bash(c.command)].append(c)
    for cmds in by_bash.values():
        if len(cmds) > 1:
            log.info(
                "Bash content shared by %d markdown blocks (suggestions will list each): %s",
                len(cmds),
                ", ".join(f"{c.file_path}:{c.start_line}" for c in cmds),
            )

    matched = 0
    unmatched = 0
    suggestions: list[Suggestion] = []
    for r in runs:
        cmds = by_bash.get(r.bash_content, [])
        if not cmds:
            unmatched += 1
            log.debug("No markdown match for test %d in %s", r.test_index, r.job_name)
            continue
        matched += 1
        if r.status == "failed":
            # A 1-second-fail test gives us no useful timing; skip.
            continue

        # When several markdown blocks share canonicalized bash, the run's
        # runtime applies to all of them — they need the same weight, so
        # emit one Suggestion per Command pointing at each markdown location.
        for cmd in cmds:
            current = cmd.weight
            observed = r.actual_seconds
            suggested = _round_to(int(observed * SAFETY_MARGIN))

            is_default = current == DEFAULT_WEIGHT
            ratio = observed / max(current, 1)
            reason: str | None = None
            if is_default and observed >= default_threshold:
                reason = "default underweight"
            elif ratio > threshold and abs(observed - current) > 60:
                reason = "underweight"
            elif current > 100 and observed * 2 < current:
                reason = "overweighted"

            if reason is None:
                continue

            anchor = (
                f"{run_url}/job/{r.job_id}#step:3:{r.passed_line_index}"
                if r.passed_line_index
                else None
            )
            suggestions.append(
                Suggestion(
                    file_path=str(Path(cmd.file_path).resolve().relative_to(REPO_ROOT)),
                    start_line=cmd.start_line,
                    current_weight=current,
                    observed_seconds=observed,
                    suggested_weight=suggested,
                    reason=reason,
                    job_log_url=anchor,
                )
            )

    return suggestions, matched, unmatched


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def render_markdown(
    suggestions: list[Suggestion],
    *,
    run_id: int,
    run_url: str,
    total_matched: int,
    total_unmatched: int,
) -> str:
    lines = [
        f"## Suggested `weight=` changes from run [#{run_id}]({run_url})",
        "",
    ]
    if not suggestions:
        lines.append(
            f"_All {total_matched} matched commands are within ±50 % of "
            f"their declared weight. No suggestions._"
        )
        if total_unmatched:
            lines.append(
                f"\n_({total_unmatched} log entries had no markdown match — "
                f"likely tutorials that changed between the run and HEAD.)_"
            )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| Tutorial | Line | Current | Observed | Suggested | Reason | Job |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for s in sorted(suggestions, key=lambda x: (-x.observed_seconds, x.file_path)):
        cur = (
            "_default_" if s.current_weight == DEFAULT_WEIGHT else str(s.current_weight)
        )
        link = f"[log]({s.job_log_url})" if s.job_log_url else "—"
        lines.append(
            f"| `{s.file_path}` | {s.start_line} | {cur} | "
            f"{s.observed_seconds} s | **{s.suggested_weight}** | "
            f"{s.reason} | {link} |"
        )
    lines.append("")
    lines.append(
        f"_{len(suggestions)} of {total_matched} matched commands "
        f"recommended for re-weighting "
        f"({total_unmatched} log entries had no markdown match)._"
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        type=int,
        default=int(os.environ.get("GITHUB_RUN_ID", 0)) or None,
        help="GitHub Actions run ID to analyze. Defaults to $GITHUB_RUN_ID when set.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Flag a test when observed/declared exceeds this ratio (default 1.5).",
    )
    parser.add_argument(
        "--default-threshold",
        type=int,
        default=120,
        help="Also flag any default-weighted test whose observed runtime exceeds "
        "this many seconds (default 120).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.environ.get("GITHUB_STEP_SUMMARY") or "-",
        help='Write the markdown report here. "-" or omitted prints to stdout.',
    )
    args = parser.parse_args(argv)

    if args.run_id is None:
        parser.error("--run-id is required (or set $GITHUB_RUN_ID).")

    repo_slug = _repo_slug()
    run_url = f"https://github.com/{repo_slug}/actions/runs/{args.run_id}"

    log.info("Enumerating tutorial commands...")
    servers = MarkdownParser().parse_directory(str(REPO_ROOT))
    all_commands: list[Command] = []
    for server in servers.values():
        all_commands.extend(server.aiperf_commands)
    log.info(
        "Found %d aiperf-run blocks across %d servers", len(all_commands), len(servers)
    )

    log.info("Fetching test-docs shard jobs from run %d...", args.run_id)
    jobs = fetch_test_docs_jobs(args.run_id)
    if not jobs:
        log.error("No test-docs-end-to-end matrix shards found in run %d.", args.run_id)
        return 1

    runs: list[TestRun] = []
    for job_id, job_name in jobs:
        log.info("Parsing log for shard %s (job %d)...", job_name, job_id)
        log_text = fetch_job_log(job_id)
        runs.extend(parse_shard_log(log_text, job_id=job_id, job_name=job_name))

    log.info("Collected %d test-run records.", len(runs))

    suggestions, matched, unmatched = derive_suggestions(
        all_commands,
        runs,
        threshold=args.threshold,
        default_threshold=args.default_threshold,
        run_url=run_url,
    )

    md = render_markdown(
        suggestions,
        run_id=args.run_id,
        run_url=run_url,
        total_matched=matched,
        total_unmatched=unmatched,
    )

    if args.output == "-":
        print(md)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(md)
        log.info("Wrote suggestion report to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
