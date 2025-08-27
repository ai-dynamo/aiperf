# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(os.getenv("AIPERF_SOURCE_DIR", Path.cwd())).resolve()
JOB_NAME = os.getenv("CI_JOB_NAME", "test_docs")
DOCS = os.getenv(
    "DOCS", ""
)  # comma-separated files/patterns, e.g. "README.md,docs/**/*.md"
BLOCK_TIMEOUT = 600

BASH_BLOCK_RE = re.compile(
    r"((?:<!--.*?-->\s*)*)```bash(.*?)```", re.DOTALL | re.IGNORECASE
)
HTML_TOKEN_RE = re.compile(r"<!--\s*([a-zA-Z0-9_:-]+)(?:\s*:\s*([^->]+))?\s*-->")

# Precompiled error patterns (case-insensitive)
LOG_ERROR_PATTERNS = [
    re.compile(r"(?i)\berror\b"),
    re.compile(r"(?i)\bexception\b"),
    re.compile(r"(?i)\btraceback\b"),
    re.compile(r"(?i)\bfailed\b"),
]


# ===============================
# Helper Functions
# ===============================
def _run_code_block(
    cmd: str, cwd: Path, env: dict, timeout: int | None
) -> tuple[int, str]:
    """
    Run a bash command, streaming output to CI while capturing logs.
    Returns (exit_code, combined_stdout).
    """
    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    start = time.time()
    captured_logs: list[str] = []
    try:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()
            captured_logs.append(line)
            if timeout and (time.time() - start) > timeout:
                proc.kill()
                raise TimeoutError(f"Command timed out after {timeout}s")
        proc.wait()
        return proc.returncode, "".join(captured_logs)
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass


def _logs_have_error(log_text: str) -> tuple[bool, str]:
    """
    Scan logs for error-like patterns (case-insensitive).
    Returns (has_error, offending_line).
    """
    for line in log_text.splitlines():
        if any(p.search(line) for p in LOG_ERROR_PATTERNS):
            return True, line.strip()
    return False, ""


def _resolve_doc_paths(paths: str) -> list[Path]:
    """
    Parse comma-separated paths/patterns, resolve relative to REPO_ROOT,
    expand globs (supports **), and return files in found order.
    """
    if not paths.strip():
        return []

    docs: list[Path] = []
    missing: list[str] = []

    for raw in (p.strip() for p in paths.split(",") if p.strip()):
        base = raw if raw.startswith("/") else str(REPO_ROOT / raw)
        matches = glob.glob(base, recursive=True)

        if not matches:
            missing.append(raw)
            continue

        for m in matches:
            p = Path(m).resolve()
            if p.is_file():
                docs.append(p)

    if missing:
        print(
            f"[pytest] Warning: no matches for entries: {', '.join(missing)}",
            flush=True,
        )

    return docs


def _extract_bash_blocks(md_path: Path):
    """
    Yield (case_id, tokens:list[str], code:str).
    Tokens are hidden HTML directives immediately above the block
    (e.g., <!-- skip -->).
    """
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    for idx, (meta, code) in enumerate(BASH_BLOCK_RE.findall(text), start=1):
        tokens = [t[0].strip() for t in HTML_TOKEN_RE.findall(meta or "")]
        yield f"{md_path}:{idx}", tokens, code.strip()


def _collect_cases():
    cases = []
    doc_files = _resolve_doc_paths(DOCS)
    if not doc_files:
        print(f"[pytest] No docs matched for DOCS='{DOCS}'.", flush=True)
        return cases
    for doc in doc_files:
        cases.extend(_extract_bash_blocks(doc))
    return cases


# ===============================
# Fixtures
# ===============================
@pytest.fixture(scope="class")
def doc_env():
    """
    Shared environment/cwd for running bash blocks.
    """
    return {
        "cwd": Path.cwd(),
        "env": os.environ.copy(),
    }


# ===============================
# Tests
# ===============================
CASES = list(_collect_cases())


@pytest.mark.usefixtures("doc_env")
class TestDocumentationBashBlocks:
    """
    Executes ```bash code blocks from documentation files.
    - Skips blocks preceded by <!-- skip -->
    - Fails if exit code != 0
    - Fails if logs contain 'error'/'exception'/'traceback'/'failed' (any case)
    """

    @pytest.mark.parametrize(
        "case_id,tokens,code",
        CASES,
        ids=[c[0] for c in CASES] if CASES else [],
    )
    def test_bash_block_runs_cleanly(self, case_id, tokens, code, doc_env):
        if "skip" in tokens:
            pytest.skip("Test Skipped")

        print(f"\n[RUNNING] {case_id}\n{code}\n", flush=True)

        rc, logs = _run_code_block(
            code, cwd=doc_env["cwd"], env=doc_env["env"], timeout=BLOCK_TIMEOUT
        )

        assert rc == 0, f"{case_id}: command exited with {rc}"

        has_err, line = _logs_have_error(logs)
        assert not has_err, f"{case_id}: log error detected → {line}"
