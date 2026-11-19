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
from typing import TYPE_CHECKING

import pytest
from pytest import param

if TYPE_CHECKING:
    from data_types import Server

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

# Each variant is GOOD with one edit. If an edit to GOOD stops either pattern
# matching, the "broken" guide silently becomes a copy of the good one and the
# case below stops testing anything it claims to.
assert SHELL_FENCE != GOOD, "the non-bash-fence edit no longer applies to GOOD"
assert UNTERMINATED != GOOD, "the unterminated-fence edit no longer applies to GOOD"


def _guide(opener: str, closer: str) -> str:
    """A complete one-server guide whose single run block uses this fence."""
    return f"""
<!-- setup-demo-endpoint-server -->
```bash
docker run demo
```

<!-- aiperf-run-demo-endpoint-server -->
{opener}
aiperf profile --x
{closer}
"""


def _parse(tmp_path: Path, markdown: str) -> dict[str, Server]:
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


def test_the_real_guides_still_parse_clean() -> None:
    """The refusals above must not reject anything the repository documents."""
    from parser import MarkdownParser

    repo = Path(__file__).resolve().parents[3]
    servers = MarkdownParser().parse_directory(str(repo))
    assert servers, "no servers discovered in the repository"
    for name, server in servers.items():
        assert server.parse_errors == [], f"{name} newly rejects a documented command"


@pytest.mark.parametrize(
    "opener,closer",
    [
        param("````bash", "````", id="four-backtick-opener"),
        param("```bash", "````", id="four-backtick-closer"),
        param("~~~bash", "~~~", id="tilde-fence"),
        param("```bash title=x", "```", id="extra-info-string"),
        param("```BASH", "```", id="uppercase"),
        param("``` bash", "```", id="space-before-info-string"),
        param("```\tbash", "```", id="tab-before-info-string"),
        param("```", "```", id="no-info-string"),
    ],
)  # fmt: skip
def test_a_fence_this_harness_never_ran_stays_unrun_and_is_recorded(
    tmp_path: Path, opener: str, closer: str
) -> None:
    """Recording the read failure must not widen what *runs*.

    Every form here was dropped silently before this change. Each must still
    be dropped -- accepting any of them would start executing a command that
    has never executed, on a GPU runner, which is a far worse outcome than
    failing loudly -- and each must now leave a parse error behind.
    """
    servers = _parse(tmp_path, _guide(opener, closer))
    demo = servers["demo"]
    assert demo.aiperf_commands == [], f"{opener!r} must not open a command block"
    assert demo.parse_errors, f"{opener!r} was dropped with no parse error"


@pytest.mark.parametrize(
    "opener,closer",
    [
        param("```bash", "    ```", id="indented-closer"),
        param("    ```bash", "    ```", id="indented-opener"),
    ],
)  # fmt: skip
def test_a_fence_this_harness_always_ran_keeps_running(
    tmp_path: Path, opener: str, closer: str
) -> None:
    """The other half of the same invariant: nothing that ran may stop running.

    Both forms are read by stripping each line, so an indented fence has
    always opened and closed a command block. Routing this reader through the
    CommonMark fence tracker would silently drop them, because CommonMark
    treats four spaces as an indented code block rather than a fence.
    """
    servers = _parse(tmp_path, _guide(opener, closer))
    demo = servers["demo"]
    assert [c.command.strip() for c in demo.aiperf_commands] == ["aiperf profile --x"]
    assert demo.parse_errors == []


def test_the_drop_reason_names_the_actual_cause(tmp_path: Path) -> None:
    """The recorded reason is the deliverable: it is what sends the guide
    author to the right line. A non-bash fence is neither missing nor
    unterminated, so it must not be reported as either."""
    servers = _parse(tmp_path, _guide("```shell", "```"))
    (problem,) = servers["demo"].parse_errors
    assert "never closed" not in problem, f"wrong cause reported: {problem}"
    assert "```bash" in problem, f"reason does not name the expected fence: {problem}"
