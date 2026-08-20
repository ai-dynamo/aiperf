# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node helpers for pure operator UI JavaScript tests."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"


def local_module_js(relpath: str) -> str:
    """Return a dependency-free UI module rewritten for bare-node evaluation.

    The harnesses in this directory strip `import` statements out of the module
    under test and replace the imported symbols with stubs. For modules that are
    themselves import-free constant/helper tables, splicing in the real source is
    strictly better than a hand-written stub: a value change in the source (a
    chart font size, a rounding rule) reaches every harness instead of silently
    diverging from a frozen copy.
    """
    source = (UI_DIR / relpath).read_text(encoding="utf-8")
    assert "\nimport " not in f"\n{source}", (
        f"{relpath} gained an import; it can no longer be spliced into a "
        "bare-node harness without resolving that dependency"
    )
    return re.sub(r"^export ", "", source, flags=re.MULTILINE)


def js_for_template_literal(js: str) -> str:
    """Escape `js` for splicing into a JS template literal.

    Several harnesses assemble their module text inside backticks. A backtick
    or `${` in the spliced module's own doc comments would end the literal
    early and turn prose into code.
    """
    return js.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


# `lib/typography.js` holds the Canvas font sizes the chart components read.
# Chart harnesses that stub imports away must define it or the module throws
# `CHART_TYPOGRAPHY is not defined` at first render.
CHART_TYPOGRAPHY_JS = local_module_js("lib/typography.js")
CHART_TYPOGRAPHY_JS_IN_TEMPLATE = js_for_template_literal(CHART_TYPOGRAPHY_JS)

# `lib/format.js` is the console's only number formatter. Harnesses used to
# hand-mirror it, which is how a stub kept claiming `fmtThroughput` was one
# decimal long after the source moved to two.
FORMAT_JS = local_module_js("lib/format.js")
FORMAT_JS_IN_TEMPLATE = js_for_template_literal(FORMAT_JS)


# `poll()` in lib/api.js suspends itself while the tab is hidden, so any
# harness that loads it needs a visible `document`. The listener registry is
# real so a test can drive a visibilitychange if it wants one.
VISIBLE_DOCUMENT_STUB_JS = """
globalThis.document = globalThis.document ?? {
  hidden: false,
  visibilityState: 'visible',
  __listeners: {},
  addEventListener(type, fn) { (this.__listeners[type] ??= []).push(fn); },
  removeEventListener(type, fn) {
    this.__listeners[type] = (this.__listeners[type] ?? []).filter((f) => f !== fn);
  },
};
"""


def run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
        # lib/format.js formats with the viewer's locale (`toLocaleString`
        # without an explicit locale), and node takes that from the ambient
        # LC_ALL/LANG. Pin it so a developer's shell locale cannot change
        # which separators the assertions see.
        env={**os.environ, "LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"},
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()
