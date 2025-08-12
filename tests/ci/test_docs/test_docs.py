# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(os.getenv("AIPERF_SOURCE_DIR", Path.cwd())).resolve()
JOB_NAME = os.getenv("CI_JOB_NAME", "test_docs")
DOCS = (REPO_ROOT / "README.md").resolve()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8080"))
URL = f"http://{HOST}:{PORT}"


START_SERVER_SCRIPT = (
    REPO_ROOT / "tests" / "ci" / JOB_NAME / "start_server.sh"
).resolve()

BLOCK_TIMEOUT = 600


def _run_dynamo() -> None:
    """
    Uses the start_server.sh to run dynamo
    """
    script = Path(START_SERVER_SCRIPT)
    if not script.exists():
        raise FileNotFoundError(f"File not found: {START_SERVER_SCRIPT}\n")

    env = os.environ.copy()
    env.setdefault("MODEL", "Qwen/Qwen3-0.6B")

    print(f"\nStarting Dynamo with (MODEL={env['MODEL']})\n", flush=True)
    subprocess.check_call(["bash", "-lc", str(script)], env=env)


def _run_code_block(cmd: str, cwd: Path, env: dict, timeout: int | None) -> int:
    """
    Runs the extracted code blocks.
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
    try:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()
            if timeout and (time.time() - start) > timeout:
                proc.kill()
                raise TimeoutError(f"Command timed out after {timeout}s")
        proc.wait()
        return proc.returncode
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass


BASH_BLOCK_RE = re.compile(
    r"((?:<!--.*?-->\s*)*)```bash(.*?)```", re.DOTALL | re.IGNORECASE
)
HTML_TOKEN_RE = re.compile(r"<!--\s*([a-zA-Z0-9_:-]+)(?:\s*:\s*([^->]+))?\s*-->")


def _extract_bash_blocks(md_path: Path):
    """
    Yield (case_id, tokens:list[str], code:str).
    Tokens are hidden HTML directives immediately above the block
    (e.g., <!-- skip -->, <!-- run_dynamo -->).
    """
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    for idx, (meta, code) in enumerate(BASH_BLOCK_RE.findall(text), start=1):
        tokens = [t[0].strip() for t in HTML_TOKEN_RE.findall(meta or "")]
        yield f"{md_path}:{idx}", tokens, code.strip()


CASES = list(_extract_bash_blocks(DOCS))


@pytest.mark.parametrize("case_id,tokens,code", CASES, ids=[c[0] for c in CASES])
def test_readme_bash_block(case_id, tokens, code):
    """
    Rules:
      - If the line above has <!-- skip -->, skip this block.
      - If the line above has <!-- run_dynamo -->, start Dynamo.
    """
    if "skip" in tokens:
        pytest.skip("Test Skipped")
    # if "run_dynamo" in tokens:
    #     _run_dynamo()

    print(f"\n[RUNNING] {case_id}\n{code}\n", flush=True)

    rc = _run_code_block(
        code, cwd=Path.cwd(), env=os.environ.copy(), timeout=BLOCK_TIMEOUT
    )
    assert rc == 0, f"{case_id}: command exited with {rc}"
