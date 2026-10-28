# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A command tag whose block cannot be read must fail the run, not vanish.

The fixture reader already refuses a block it cannot read and records a parse
error. The command reader did neither: a block labelled ```shell was dropped
with a log line and no parse error, and an unterminated ```bash fence left the
tag scanner inside a fence, so the *next* tag was skipped as documentation.
Both left a green run that executed fewer commands than the guides document.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

GOOD = """
<!-- setup-demo-endpoint-server -->
```bash
docker run demo
```

<!-- aiperf-run-demo-endpoint-server -->
```bash
aiperf profile --one
```

<!-- aiperf-run-demo-endpoint-server -->
```bash
aiperf profile --two
```
"""

SHELL_FENCE = GOOD.replace(
    "```bash\naiperf profile --one", "```shell\naiperf profile --one"
)
UNTERMINATED = GOOD.replace("aiperf profile --one\n```\n", "aiperf profile --one\n")


def _parse(tmp_path: Path, markdown: str):
    from parser import MarkdownParser

    (tmp_path / "guide.md").write_text(markdown)
    return MarkdownParser().parse_directory(str(tmp_path))


def test_a_well_formed_guide_still_parses_every_command(tmp_path: Path) -> None:
    servers = _parse(tmp_path, GOOD)
    assert sorted(c.command for c in servers["demo"].aiperf_commands) == [
        "aiperf profile --one",
        "aiperf profile --two",
    ]
    assert servers["demo"].parse_errors == []


@pytest.mark.parametrize(
    "markdown,surviving",
    [
        param(SHELL_FENCE, ["aiperf profile --two"], id="non-bash-fence"),
        param(UNTERMINATED, [], id="unterminated-fence"),
    ],
)  # fmt: skip
def test_an_unreadable_command_block_records_a_parse_error(
    tmp_path: Path, markdown: str, surviving: list[str]
) -> None:
    """The dropped command must be recorded, or the run reports a false pass."""
    servers = _parse(tmp_path, markdown)
    demo = servers["demo"]
    assert sorted(c.command for c in demo.aiperf_commands) == surviving
    assert demo.parse_errors, "a dropped command left no parse error"


def test_an_unterminated_fence_does_not_register_a_swallowed_tag(
    tmp_path: Path,
) -> None:
    """The open fence used to run to the *next* block's closer, so the tag in
    between landed inside the first command's body and registered nothing."""
    servers = _parse(tmp_path, UNTERMINATED)
    bodies = [c.command for c in servers["demo"].aiperf_commands]
    assert not any("aiperf-run-demo-endpoint-server" in b for b in bodies)


def test_the_real_guides_still_parse_clean(tmp_path: Path) -> None:
    """The refusals above must not reject anything the repository documents."""
    from parser import MarkdownParser

    repo = Path(__file__).resolve().parents[3]
    servers = MarkdownParser().parse_directory(str(repo))
    assert servers, "no servers discovered in the repository"
    for name, server in servers.items():
        assert server.parse_errors == [], f"{name} newly rejects a documented command"
