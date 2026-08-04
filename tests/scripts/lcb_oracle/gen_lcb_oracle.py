#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a mock-server accuracy oracle for the LiveCodeBench codegen benchmark.

``lcb_codegeneration`` is graded by executing the model's code against real test
cases, so a mock server cannot fake it by echoing a letter the way mmlu can — it
has to return a program that genuinely passes. LiveCodeBench ships no reference
solutions, and solving the problems is not the point.

The trick: for each stdin/stdout problem, emit a program that maps every test
case's exact stdin to its expected stdout. The grader runs those same cases, so
the lookup passes all of them and scores ``pass@1 = 1.0``. No problem-solving
required, and the result is deterministic.

Every generated solution is executed through the real grading worker before it is
written, so a row only lands in the oracle if it actually scores 1.0. All rows are
correct solutions — let the mock decide which responses come back *wrong*, via
``--random-seed`` + ``--accuracy-correct-rate``, so its own tally stays an
independent oracle you can check AIPerf's grades against.

See ``README.md`` in this directory for the full end-to-end recipe.

Usage:
    python gen_lcb_oracle.py --out lcb_oracle.jsonl --count 6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import orjson


def build_lookup_solution(cases: list[dict]) -> str:
    """A program that reads all of stdin and prints the expected output for it.

    Falls back to a whitespace-insensitive comparison so a trailing-newline
    difference between the dataset and the harness cannot fail an otherwise
    correct lookup.
    """
    table = {c["input"]: c["output"] for c in cases}
    return (
        "import sys\n"
        f"_T = {table!r}\n"
        "_d = sys.stdin.read()\n"
        "if _d in _T:\n"
        "    sys.stdout.write(_T[_d])\n"
        "else:\n"
        "    _s = _d.strip()\n"
        "    for _k, _v in _T.items():\n"
        "        if _k.strip() == _s:\n"
        "            sys.stdout.write(_v)\n"
        "            break\n"
    )


def grade_once(worker_cmd: list[str], sample: list[dict], code: str) -> float:
    """Run one grade through the real worker subprocess; return pass@1.

    ``AIPERF_CODEGEN_DEATH_FD`` is cleared: when the client sets it the worker
    registers a second at-fork handler that corrupts lighteval's forked sandbox
    children and zeroes every grade. See the caveat in README.md.
    """
    env = dict(os.environ)
    env.pop("AIPERF_CODEGEN_DEATH_FD", None)
    req = (
        orjson.dumps({"id": 1, "evaluation_sample": sample, "generated_code": [[code]]})
        + b"\n"
    )
    proc = subprocess.Popen(
        worker_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    out, _ = proc.communicate(req, timeout=600)
    if not out.strip():
        return -1.0
    resp = orjson.loads(out.splitlines()[0])
    if not resp.get("ok"):
        return -1.0
    return float(resp.get("metrics", {}).get("pass@1", -1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a validated LCB codegen oracle for the mock server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--count", type=int, default=6, help="number of problems to emit")
    p.add_argument(
        "--release-tag",
        default=None,
        help="LCB subset. MUST match what AIPerf loads "
        "(Environment.ACCURACY.LCB_RELEASE_TAG); a different subset yields "
        "different problems and the mock will never match the prompts.",
    )
    p.add_argument(
        "--validator-worker",
        default=None,
        help="path to a grading worker script to validate against. Defaults to "
        "the installed 'python -m aiperf.accuracy.graders._codegen_worker'. "
        "Point this at a known-good worker when the installed one is suspect.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    worker_cmd = (
        [sys.executable, args.validator_worker]
        if args.validator_worker
        else [sys.executable, "-m", "aiperf.accuracy.graders._codegen_worker"]
    )

    from lighteval.tasks.tasks.lcb.codegen_metrics import translate_private_test_cases

    from aiperf.accuracy.benchmarks._datasets_compat import load_dataset
    from aiperf.common.environment import Environment

    release_tag = args.release_tag or Environment.ACCURACY.LCB_RELEASE_TAG
    print(f"LCB subset (must match AIPerf): {release_tag!r}")
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        release_tag,
        split="test",
        trust_remote_code=True,
    )

    rows: list[dict] = []
    scanned = skipped = 0
    for row in ds:
        if len(rows) >= args.count:
            break
        scanned += 1
        # stdin/stdout problems only: a starter_code or func_name means the
        # grader calls a function, which a stdin lookup table cannot satisfy.
        if (row.get("starter_code") or "").strip():
            skipped += 1
            continue
        meta = json.loads(row["metadata"]) if row.get("metadata") else {}
        if meta.get("func_name"):
            skipped += 1
            continue

        public = json.loads(row["public_test_cases"])
        try:
            private = translate_private_test_cases(row["private_test_cases"])
        except Exception:
            private = []
        cases = [c for c in (public + private) if c.get("testtype") == "stdin"]
        if not cases or len(cases) != len(public) + len(private):
            skipped += 1
            continue

        code = build_lookup_solution(cases)
        sample = [
            {
                "input_output": orjson.dumps(
                    {
                        "inputs": [c["input"] for c in cases],
                        "outputs": [c["output"] for c in cases],
                        "fn_name": None,
                    }
                ).decode()
            }
        ]
        score = grade_once(worker_cmd, sample, code)
        keep = score == 1.0
        print(
            f"  [{'KEEP' if keep else 'drop'}] {row['question_id']:>10s}  "
            f"cases={len(cases):3d}  pass@1={score}",
            flush=True,
        )
        if not keep:
            skipped += 1
            continue

        rows.append(
            {
                # question_content is a substring of AIPerf's wire prompt, which
                # is what --accuracy-match substring keys on. Do NOT use
                # question_id: it never appears in the prompt.
                "text": row["question_content"],
                "ground_truth": f"```python\n{code}\n```",
                "format": "passthrough",
                "task": row["question_id"],
            }
        )

    with open(args.out, "wb") as fh:
        for r in rows:
            fh.write(orjson.dumps(r) + b"\n")

    print(
        f"\nscanned {scanned}, skipped {skipped}, wrote {len(rows)} rows -> {args.out}"
    )
    if not rows:
        print(
            "no validated rows produced; see README.md troubleshooting", file=sys.stderr
        )
        return 1
    print(
        "All rows are correct solutions. Use --accuracy-correct-rate < 1.0 with a "
        "pinned --random-seed to have the mock decide which come back wrong."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
