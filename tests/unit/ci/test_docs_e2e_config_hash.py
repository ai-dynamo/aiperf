# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The detect-changes fingerprint must cover everything the harness acts on.

``.github/workflows/test-docs-end-to-end.yml`` skips the GPU matrix when
``dump_config_hash.py`` prints the same digest for base and head, and runs it
when the script exits non-zero. Anything the harness reads but the digest
ignores is a documented change that ships without ever being executed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tests/ci/test_docs_end_to_end/dump_config_hash.py"

SETUP = """\
<!-- setup-probe-endpoint-server -->
```bash
docker run -p 8000:8000 example/server:latest
```
<!-- /setup-probe-endpoint-server -->

<!-- health-check-probe-endpoint-server -->
```bash
curl -sf localhost:8000/health
```
<!-- /health-check-probe-endpoint-server -->
"""

RUN = """\
<!-- aiperf-run-probe-endpoint-server{attrs} -->
```bash
aiperf profile --model m --url localhost:8000
```
<!-- /aiperf-run-probe-endpoint-server -->
"""

FIXTURE = """\
<!-- setup-file-probe-endpoint-server path={path} -->
```yaml
{content}
```
<!-- /setup-file-probe-endpoint-server -->
"""


def _hash(tmp_path: Path, doc: str) -> subprocess.CompletedProcess:
    """Run the real script against a tree holding exactly one guide."""
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "probe.md").write_text(doc)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def test_a_timeout_only_edit_changes_the_digest(tmp_path) -> None:
    """A deadline decides whether a documented command is killed mid-run."""
    base = _hash(tmp_path / "a", SETUP + RUN.format(attrs=""))
    head = _hash(tmp_path / "b", SETUP + RUN.format(attrs=" timeout=3600"))

    assert base.returncode == 0 and head.returncode == 0
    assert base.stdout != head.stdout


def test_a_fixture_content_edit_changes_the_digest(tmp_path) -> None:
    """The fixture is what a ``--config`` command actually benchmarks."""
    doc = SETUP + RUN.format(attrs="")
    base = _hash(tmp_path / "a", doc + FIXTURE.format(path="c.yaml", content="n: 1"))
    head = _hash(tmp_path / "b", doc + FIXTURE.format(path="c.yaml", content="n: 2"))

    assert base.returncode == 0 and head.returncode == 0
    assert base.stdout != head.stdout


def test_a_fixture_rename_changes_the_digest(tmp_path) -> None:
    """A renamed fixture is a different test even with identical content."""
    doc = SETUP + RUN.format(attrs="")
    base = _hash(tmp_path / "a", doc + FIXTURE.format(path="c.yaml", content="n: 1"))
    head = _hash(tmp_path / "b", doc + FIXTURE.format(path="d.yaml", content="n: 1"))

    assert base.returncode == 0 and head.returncode == 0
    assert base.stdout != head.stdout


def test_rejected_tags_fail_rather_than_hash(tmp_path) -> None:
    """A duplicate setup leaves no comparable fingerprint, so CI must not skip.

    The parser deliberately records the conflict rather than aborting, so
    nothing else stops this tree from producing the pre-conflict digest and
    comparing clean against its own base.
    """
    result = _hash(tmp_path, SETUP + SETUP + RUN.format(attrs=""))

    assert result.returncode == 1
    assert result.stdout.strip() == ""
    assert "probe" in result.stderr


def test_a_well_formed_tree_still_prints_one_digest(tmp_path) -> None:
    result = _hash(tmp_path, SETUP + RUN.format(attrs=""))

    assert result.returncode == 0
    assert len(result.stdout.strip()) == 64
