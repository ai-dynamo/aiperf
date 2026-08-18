# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct coverage for the two newest operator UI modules.

`components/token-modal.js` is the only place a user types a secret into the
console, and `lib/typography.js` is the only place Canvas font sizes are
allowed to live. Both arrived without a test of their own.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests.unit.ui.node_utils import CHART_TYPOGRAPHY_JS, run_node

_UI = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
_TOKEN_MODAL = _UI / "components" / "token-modal.js"
_TYPOGRAPHY = _UI / "lib" / "typography.js"


# ---------------------------------------------------------------------------
# components/token-modal.js
# ---------------------------------------------------------------------------


def test_token_modal_masks_the_secret_and_names_what_to_paste() -> None:
    """A bearer token typed in the clear is readable over a shoulder and lands
    in the browser's form history; and a prompt that does not name its source
    invites the user to paste some other credential."""
    src = _TOKEN_MODAL.read_text(encoding="utf-8")

    assert 'type="password"' in src
    assert "AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN" in src
    assert "sessionStorage" in src, (
        "the modal must tell the user where the token goes; storage itself "
        "stays in lib/api.js"
    )
    # Reading or writing storage here would fork the token's lifecycle away
    # from the getSessionToken/clearSessionToken pair that owns it.
    assert "sessionStorage.setItem" not in src
    assert "sessionStorage.getItem" not in src


def test_token_modal_refuses_to_confirm_an_empty_token() -> None:
    """An empty submit would close the modal and re-issue the request without
    credentials, which surfaces as a second, more confusing 401."""
    src = _TOKEN_MODAL.read_text(encoding="utf-8")

    assert "disabled=${!token.trim()}" in src
    assert "const t = token.trim();" in src
    assert "if (t) onConfirm(t);" in src


def test_token_modal_is_dismissable_by_keyboard_and_by_the_backdrop() -> None:
    """The overlay covers the page: a user who cannot see a way out has none."""
    src = _TOKEN_MODAL.read_text(encoding="utf-8")

    assert "if (e.key === 'Escape') onCancel();" in src
    assert "document.removeEventListener('keydown', onKeyDown)" in src
    assert "if (e.target === e.currentTarget) onCancel();" in src
    assert "inputRef.current?.focus();" in src


# ---------------------------------------------------------------------------
# lib/typography.js
# ---------------------------------------------------------------------------


def test_chart_typography_exposes_numeric_pixels_and_cannot_be_mutated() -> None:
    """Chart.js font sizes go straight into a Canvas context: a `var(--x)`
    string silently renders at the browser default instead of failing. The
    table is shared by every chart, so a component must not be able to edit it
    for everyone else."""
    out = json.loads(
        run_node(
            CHART_TYPOGRAPHY_JS
            + """
        const before = CHART_TYPOGRAPHY.AXIS_TICK;
        try { CHART_TYPOGRAPHY.AXIS_TICK = 99; } catch (_) {}
        console.log(JSON.stringify({
          entries: Object.entries(CHART_TYPOGRAPHY),
          frozen: Object.isFrozen(CHART_TYPOGRAPHY),
          afterWrite: CHART_TYPOGRAPHY.AXIS_TICK === before,
        }));
        """
        )
    )

    assert out["frozen"] is True
    assert out["afterWrite"] is True
    # Import-free by contract: the bare-node harnesses splice this module in.
    assert "import " not in _TYPOGRAPHY.read_text(encoding="utf-8")
    for name, value in out["entries"]:
        if name.endswith("_FONT"):
            # A full CSS font shorthand still has to start with a px size.
            assert re.match(r"^\d+px \S", value), (name, value)
        else:
            assert isinstance(value, int | float), (name, value)
            assert 8 <= value <= 32, (name, value)


def test_chart_components_take_canvas_font_sizes_only_from_the_shared_table() -> None:
    """The reason this module exists. A literal `font: { size: 10 }` in one
    chart is how the axis labels drifted apart across the console."""
    offenders = [
        f"{path.relative_to(_UI).as_posix()}: {match.group(0)}"
        for path in sorted(_UI.rglob("*.js"))
        if "vendor" not in path.parts
        for match in re.finditer(
            r"font:\s*\{\s*size:\s*(?P<size>[^}]+?)\s*\}",
            path.read_text(encoding="utf-8"),
        )
        if "CHART_TYPOGRAPHY" not in match.group("size")
    ]

    assert offenders == []
