# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tag shown inside a code fence is an example, not a tag.

Any document that explains the tag syntax has to print it. A parser that does
not track fences registers those examples as real commands against real server
groups -- silently duplicating a benchmark, or writing a file fixture the guide
never asked for.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests/ci/test_docs_end_to_end"))

from parser import MarkdownParser  # noqa: E402

EXAMPLE_DOC = """\
# How to tag a guide

````markdown
<!-- setup-example-endpoint-server -->
```bash
docker run -d nginx
```
<!-- /setup-example-endpoint-server -->

<!-- aiperf-run-example-endpoint-server -->
```bash
aiperf profile --model m
```
<!-- /aiperf-run-example-endpoint-server -->

<!-- setup-file-example-endpoint-server path=config.yaml -->
```yaml
model: m
```
<!-- /setup-file-example-endpoint-server -->
````
"""

REAL_DOC = """\
<!-- setup-real-endpoint-server -->
```bash
docker run -d nginx
```
<!-- /setup-real-endpoint-server -->

<!-- health-check-real-endpoint-server -->
```bash
echo ok
```
<!-- /health-check-real-endpoint-server -->

<!-- aiperf-run-real-endpoint-server -->
```bash
aiperf profile --model m
```
<!-- /aiperf-run-real-endpoint-server -->
"""


def _parse(tmp_path: Path, **docs: str) -> dict:
    for name, text in docs.items():
        (tmp_path / f"{name}.md").write_text(text)
    return MarkdownParser().parse_directory(str(tmp_path))


def test_tags_inside_a_fence_register_no_server(tmp_path) -> None:
    assert _parse(tmp_path, example=EXAMPLE_DOC) == {}


def test_fenced_examples_do_not_pollute_a_real_server(tmp_path) -> None:
    """The example names the same group as a genuine one elsewhere."""
    shadow = EXAMPLE_DOC.replace("example-endpoint", "real-endpoint")
    servers = _parse(tmp_path, a_real=REAL_DOC, b_shadow=shadow)

    assert set(servers) == {"real"}
    real = servers["real"]
    assert len(real.aiperf_commands) == 1
    assert not real.files


def test_real_tags_outside_a_fence_still_parse(tmp_path) -> None:
    servers = _parse(tmp_path, real=REAL_DOC)
    assert set(servers) == {"real"}
    assert servers["real"].setup_command is not None
    assert servers["real"].health_check_command is not None
    assert len(servers["real"].aiperf_commands) == 1


INDENTED_CLOSER_DOC = """\
Explaining a nested block:

````markdown
<!-- setup-example-endpoint-server -->
```bash
docker run -d nginx
```
    ````
<!-- aiperf-run-example-endpoint-server -->
```bash
aiperf profile --model m
```
<!-- /aiperf-run-example-endpoint-server -->
````
"""


def test_an_indented_marker_does_not_close_the_fence(tmp_path) -> None:
    """Four spaces makes it indented content, not a fence boundary.

    The marker matches the opener's character and length, so only its
    indentation keeps it from closing the block. Stripping before the check
    would end the example here, and every tag after it would parse as a real
    command.
    """
    assert _parse(tmp_path, indented=INDENTED_CLOSER_DOC) == {}
