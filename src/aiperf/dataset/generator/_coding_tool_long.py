# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-form tool generators: tool_read_long, tool_bash_verbose, tool_search_verbose.

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiperf.dataset.generator._coding_vocab import (
    _CLASSES,
    _CLI_COMMANDS,
    _ERROR_MESSAGES,
    _METHODS,
    _MODULES,
    _TYPES,
    _VARS,
)


@dataclass(frozen=True, slots=True)
class _ReadLongCtx:
    """Pre-sampled randomness shared across per-language tool-read-long builders."""

    cls: str
    m1: str
    m2: str
    m3: str
    v1: str
    v2: str
    v3: str
    mod: str
    err: str
    t1: str
    t2: str


def _read_long_python(c: _ReadLongCtx) -> list[str]:
    return [
        f"class {c.cls}:",
        f'    """{c.cls} handles {c.m1} operations for {c.mod}."""',
        "",
        f"    _default_{c.v3} = 64",
        "",
        f"    def __init__(self, {c.v1}: {c.t1}, {c.v2}: {c.t2} = None):",
        f"        self._{c.v1} = {c.v1}",
        f"        self._{c.v2} = {c.v2}",
        f"        self._{c.v3} = self._default_{c.v3}",
        "        self._initialized = False",
        "        self._lock = asyncio.Lock()",
        "",
        f"    async def {c.m1}(self, {c.v1}: {c.t1}) -> {c.t2}:",
        "        if not self._initialized:",
        f'            raise RuntimeError("{c.cls} not initialized")',
        "        async with self._lock:",
        f"            {c.v2} = await self._{c.m2}({c.v1})",
        f"            if {c.v2} is None:",
        f'                raise ValueError("{c.err}")',
        f"            return {c.v2}",
        "",
        f"    async def _{c.m2}(self, {c.v1}: {c.t1}) -> {c.t2}:",
        "        try:",
        f"            {c.v2} = await {c.mod}.{c.m2}({c.v1})",
        f'            logger.debug(f"{c.cls}.{c.m2}: {{{{{c.v1}}}}}")',
        f"            return {c.v2}",
        "        except Exception as e:",
        f'            logger.error("{c.err}: %s", e)',
        f'            raise ValueError("{c.err}") from e',
        "",
        f"    async def {c.m3}(self, {c.v1}: {c.t1}, {c.v2}: {c.t2}) -> None:",
        f"        if {c.v1} is None:",
        "            return",
        f"        existing = await self._{c.m2}({c.v1})",
        "        if existing is not None:",
        f"            existing.{c.v3} = {c.v2}",
        "            await existing.save()",
        "        else:",
        f"            await {c.mod}.{c.m3}({c.v1}, {c.v2})",
        "",
        f"    def {c.m1}_sync(self) -> None:",
        "        self._initialized = True",
        f"        self._{c.v3} = 0",
    ]


def _read_long_go(c: _ReadLongCtx) -> list[str]:
    return [
        f"type {c.cls} struct {{",
        f"\t{c.v1} {c.t1}",
        f"\t{c.v2} {c.t2}",
        "\tmu   sync.RWMutex",
        "\tlog  *zap.Logger",
        "}",
        "",
        f"func New{c.cls}({c.v1} {c.t1}, log *zap.Logger) *{c.cls} {{",
        f"\treturn &{c.cls}{{",
        f"\t\t{c.v1}: {c.v1},",
        "\t\tlog: log,",
        "\t}",
        "}",
        "",
        f"func (s *{c.cls}) {c.m1.title()}(ctx context.Context) error {{",
        "\ts.mu.Lock()",
        "\tdefer s.mu.Unlock()",
        "",
        f"\t{c.v2}, err := s.{c.m2.title()}(ctx)",
        "\tif err != nil {",
        f'\t\treturn fmt.Errorf("{c.err}: %w", err)',
        "\t}",
        f"\ts.{c.v1} = {c.v2}",
        "\treturn nil",
        "}",
        "",
        f"func (s *{c.cls}) {c.m2.title()}(ctx context.Context) ({c.t2}, error) {{",
        f'\ts.log.Debug("{c.cls}.{c.m2.title()}", zap.String("{c.v1}", s.{c.v1}))',
        f"\tresult, err := {c.mod}.{c.m2.title()}(ctx, s.{c.v1})",
        "\tif err != nil {",
        f'\t\treturn "", fmt.Errorf("{c.err}: %w", err)',
        "\t}",
        "\treturn result, nil",
        "}",
    ]


def _read_long_rust(c: _ReadLongCtx) -> list[str]:
    return [
        f"pub struct {c.cls} {{",
        f"    {c.v1}: {c.t1},",
        f"    {c.v2}: Option<{c.t2}>,",
        "    initialized: bool,",
        "}",
        "",
        f"impl {c.cls} {{",
        f"    pub fn new({c.v1}: {c.t1}) -> Self {{",
        f"        Self {{ {c.v1}, {c.v2}: None, initialized: false }}",
        "    }",
        "",
        f"    pub async fn {c.m1}(&mut self) -> Result<{c.t2}> {{",
        f'        anyhow::ensure!(self.initialized, "{c.cls} not initialized");',
        f"        let {c.v2} = self.{c.m2}().await?;",
        f"        if {c.v2}.is_empty() {{",
        f'            anyhow::bail!("{c.err}");',
        "        }",
        f"        Ok({c.v2})",
        "    }",
        "",
        f"    async fn {c.m2}(&self) -> Result<{c.t2}> {{",
        f"        let {c.v2} = {c.mod}::{c.m2}(&self.{c.v1}).await?;",
        f'        tracing::debug!("{c.cls}.{c.m2}: {{}}", self.{c.v1});',
        f"        Ok({c.v2})",
        "    }",
        "",
        f"    pub async fn {c.m3}(&mut self, {c.v1}: {c.t1}) -> Result<()> {{",
        f"        let existing = self.{c.m2}().await.ok();",
        "        match existing {",
        "            Some(val) if !val.is_empty() => {",
        f"                self.{c.v2} = Some(val);",
        "            }",
        "            _ => {",
        f"                {c.mod}::{c.m3}(&{c.v1}).await?;",
        "            }",
        "        }",
        "        Ok(())",
        "    }",
        "}",
    ]


def _read_long_typescript(c: _ReadLongCtx) -> list[str]:
    return [
        f"export class {c.cls} {{",
        f"  private {c.v1}: {c.t1};",
        f"  private {c.v2}: {c.t2} | null = null;",
        "  private initialized = false;",
        "",
        f"  constructor({c.v1}: {c.t1}) {{",
        f"    this.{c.v1} = {c.v1};",
        "  }",
        "",
        f"  async {c.m1}({c.v1}: {c.t1}): Promise<{c.t2}> {{",
        "    if (!this.initialized) {",
        f"      throw new Error('{c.cls} not initialized');",
        "    }",
        f"    const {c.v2} = await this.{c.m2}({c.v1});",
        f"    if (!{c.v2}) {{",
        f"      throw new Error('{c.err}');",
        "    }",
        f"    return {c.v2};",
        "  }",
        "",
        f"  private async {c.m2}({c.v1}: {c.t1}): Promise<{c.t2} | null> {{",
        "    try {",
        f"      const {c.v2} = await {c.mod}.{c.m2}({c.v1});",
        f"      console.debug(`{c.cls}.{c.m2}: ${{{{{c.v1}}}}}`);",
        f"      return {c.v2};",
        "    } catch (err) {",
        f"      console.error('{c.err}:', err);",
        "      throw err;",
        "    }",
        "  }",
        "",
        f"  async {c.m3}({c.v1}: {c.t1}, {c.v2}: {c.t2}): Promise<void> {{",
        f"    const existing = await this.{c.m2}({c.v1}).catch(() => null);",
        "    if (existing) {",
        f"      Object.assign(existing, {{ {c.v3}: {c.v2} }});",
        "      await existing.save();",
        "    } else {",
        f"      await {c.mod}.{c.m3}({c.v1}, {c.v2});",
        "    }",
        "  }",
        "}",
    ]


class _ToolLongMixin:
    def _gen_tool_read_long(self, language: str | None = None) -> str:
        """Like _gen_tool_read but with 40-80 lines for realistic large file reads."""
        r = self._template_rng
        file_pool = self._file_pool(language)
        f = r.choice(file_pool)
        start_line = r.randint(1, 200)
        cls = r.choice(_CLASSES)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2, v3 = r.sample(_VARS, 3)
        mod = r.choice(_MODULES)
        err = r.choice(_ERROR_MESSAGES)
        t1, t2 = r.sample(_TYPES, 2)

        ctx = _ReadLongCtx(
            cls=cls,
            m1=m1,
            m2=m2,
            m3=m3,
            v1=v1,
            v2=v2,
            v3=v3,
            mod=mod,
            err=err,
            t1=t1,
            t2=t2,
        )
        builder = {
            "python": _read_long_python,
            "go": _read_long_go,
            "rust": _read_long_rust,
            "typescript": _read_long_typescript,
        }.get(language, _read_long_python)
        code_lines = builder(ctx)

        lines = []
        for i, content in enumerate(code_lines, start=start_line):
            lines.append(f"{i:>6}\t{content}")

        content = "\n".join(lines)
        return f"""\
<tool_name>read</tool_name>
<parameter name="file_path">{f}</parameter>
<result>
{content}
</result>
"""

    def _gen_tool_bash_verbose(self, language: str | None = None) -> str:
        """Like _gen_tool_bash but with longer, more realistic test output."""
        r = self._template_rng
        mod = r.choice(_MODULES)
        cls = r.choice(_CLASSES)
        methods = r.sample(list(_METHODS), r.randint(8, 15))
        n_pass = r.randint(30, 150)
        n_fail = r.randint(0, 3)
        dur = r.uniform(2.0, 45.0)

        lang_cmds: dict[str | None, str] = {
            "python": "pytest -xvs tests/",
            "go": "go test -v ./...",
            "rust": "cargo test",
            "typescript": "npx vitest run",
        }
        cmd = lang_cmds.get(language, r.choice(_CLI_COMMANDS))

        test_lines = []
        for m in methods:
            passed = r.random() > 0.15
            t = r.uniform(0.001, 3.0)
            if language == "go":
                status = "ok" if passed else "FAIL"
                test_lines.append(f"--- {status}: Test{m.title()} ({t:.3f}s)")
                if not passed:
                    v = r.choice(_VARS)
                    test_lines.append(
                        f"        {mod}_test.go:{r.randint(20, 300)}: "
                        f"expected {v} to be non-nil"
                    )
            elif language == "rust":
                status = "ok" if passed else "FAILED"
                test_lines.append(f"test {mod}::{cls.lower()}::test_{m} ... {status}")
                if not passed:
                    err = r.choice(_ERROR_MESSAGES)
                    test_lines.append(f"  thread '{m}' panicked at '{err}'")
            elif language == "typescript":
                mark = "\u2713" if passed else "\u2717"
                test_lines.append(f"  {mark} {cls} > {m} ({r.randint(1, 800)} ms)")
                if not passed:
                    test_lines.append("    Expected: true\n    Received: false")
            else:
                status = "PASSED" if passed else "FAILED"
                test_lines.append(f"tests/test_{mod}.py::Test{cls}::test_{m} {status}")
                if not passed:
                    err = r.choice(_ERROR_MESSAGES)
                    v = r.choice(_VARS)
                    test_lines.extend(
                        [
                            f"    FAILED tests/test_{mod}.py::Test{cls}::test_{m}",
                            f"    AssertionError: assert {v} == expected",
                            f"      where {v} = {cls}().{m}()",
                            f"    {err}",
                        ]
                    )
        test_output = "\n".join(test_lines)

        warnings = ""
        if r.random() < 0.4:
            w_count = r.randint(1, 5)
            warnings = f"\n\n{w_count} warning(s)"

        return f"""\
<tool_name>bash</tool_name>
<parameter name="command">{cmd}</parameter>
<result>
{test_output}
{warnings}
========================= {n_pass} passed, {n_fail} failed in {dur:.2f}s =========================
</result>
"""

    def _gen_tool_search_verbose(self, language: str | None = None) -> str:
        """Like _gen_tool_search but returns many matches across multiple files."""
        r = self._template_rng
        file_pool = self._file_pool(language)
        pattern = r.choice(_CLASSES)

        files = r.sample(list(file_pool), min(r.randint(6, 12), len(file_pool)))
        matches = []
        for f in files:
            n_hits = r.randint(1, 4)
            for _ in range(n_hits):
                line_num = r.randint(1, 500)
                v = r.choice(_VARS)
                m = r.choice(_METHODS)
                ctx = r.choice(
                    [
                        f"class {pattern}({r.choice(_CLASSES)}):",
                        f"    {m} = {pattern}({v})",
                        f"from {r.choice(_MODULES)} import {pattern}",
                        f"    self._{v} = {pattern}.{m}()",
                        f"    result: {pattern} = await svc.{m}({v})",
                        f"# TODO: refactor {pattern} to use async",
                    ]
                )
                matches.append(f"{f}:{line_num}:{ctx}")

        content = "\n".join(matches)
        return f"""\
<tool_name>search</tool_name>
<parameter name="pattern">{pattern}</parameter>
<result>
Found {len(matches)} matches in {len(files)} files:

{content}
</result>
"""
