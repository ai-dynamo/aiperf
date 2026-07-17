# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-use and bash-template generators (mixin for CodingContentGenerator).

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from aiperf.dataset.generator._coding_vocab import (
    _CLASSES,
    _CLI_COMMANDS,
    _ERROR_MESSAGES,
    _FILE_PATHS,
    _GO_PACKAGES,
    _LANG_FILE_PATHS,
    _METHODS,
    _MODULES,
    _RUST_CRATES,
    _VARS,
)


class _ToolMixin:
    def _file_pool(self, language: str | None) -> tuple[str, ...]:
        if language:
            return _LANG_FILE_PATHS.get(language, _FILE_PATHS)
        return _FILE_PATHS

    def _gen_tool_use_block(self, language: str | None = None) -> str:
        r = self._template_rng
        return r.choice(
            [
                lambda: self._gen_tool_read(language=language),
                lambda: self._gen_tool_edit(language=language),
                lambda: self._gen_tool_search(language=language),
                lambda: self._gen_tool_bash(language=language),
            ]
        )()

    def _gen_tool_read(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        f = r.choice(file_pool)
        start_line = r.randint(1, 200)
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        mod = r.choice(_MODULES)
        err = r.choice(_ERROR_MESSAGES)

        lang_lines: dict[str | None, list[str]] = {
            "python": [
                f"def {m1}(self, {v1}):",
                f"self._{v1} = {v1}",
                f"{v2} = {mod}.{m2}({v1})",
                f"if {v1} is None:",
                f'    raise ValueError("{err}")',
                f"return {v2}",
                f'logger.debug(f"{cls}.{m1}: {{{{{v1}}}}}")',
                "",
            ],
            "go": [
                f"func (s *{cls}) {m1.title()}(ctx context.Context) error {{",
                f"s.{v1} = {v1}",
                f"{v2}, err := s.{m2.title()}(ctx)",
                "if err != nil {",
                f'return fmt.Errorf("{err}: %w", err)',
                "}",
                "return nil",
                "",
            ],
            "rust": [
                f"pub async fn {m1}(&mut self) -> Result<()> {{",
                f"let {v1} = self.{v2}.clone();",
                f"let {v2} = self.{m2}(&{v1}).await?;",
                f"if {v2}.is_empty() {{",
                f'anyhow::bail!("{err}");',
                "}",
                "Ok(())",
                "",
            ],
            "typescript": [
                f"async {m1}({v1}: string): Promise<void> {{",
                f"this.{v1} = {v1};",
                f"const {v2} = await this.{m2}({v1});",
                f"if (!{v2}) {{",
                f"  throw new Error('{err}');",
                "}",
                f"console.log(`{cls}.{m1}: ${{{{{v2}}}}}`);",
                "",
            ],
        }
        code_lines = lang_lines.get(language, lang_lines["python"])

        lines = []
        for i in range(start_line, start_line + r.randint(15, 30)):
            indent = "    " if r.random() > 0.3 else "        "
            line_content = r.choice(code_lines)
            lines.append(f"{i:>6}\t{indent}{line_content}")

        content = "\n".join(lines)
        return f"""\
<tool_name>read</tool_name>
<parameter name="file_path">{f}</parameter>
<result>
{content}
</result>
"""

    def _gen_tool_edit(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        f = r.choice(file_pool)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        cls = r.choice(_CLASSES)
        err = r.choice(_ERROR_MESSAGES)

        edits: dict[str | None, tuple[str, str]] = {
            "python": (
                f"    def {m1}(self, {v1}):\n        return self._{m2}({v1})",
                f"    async def {m1}(self, {v1}: str) -> dict:\n"
                f"        try:\n"
                f"            {v2} = await self._{m2}({v1})\n"
                f"            if {v2} is None:\n"
                f'                raise ValueError("{err}")\n'
                f'            return {{{{"status": "ok", "data": {v2}}}}}\n'
                f"        except Exception as exc:\n"
                f'            logger.error("{cls}.{m1} failed: %s", exc)\n'
                f"            raise",
            ),
            "go": (
                f"func (s *{cls}) {m1.title()}() error {{{{\n    return nil\n}}}}",
                f"func (s *{cls}) {m1.title()}(ctx context.Context) error {{{{\n"
                f"    {v2}, err := s.{m2.title()}(ctx)\n"
                f"    if err != nil {{{{\n"
                f'        return fmt.Errorf("{err}: %w", err)\n'
                f"    }}}}\n"
                f"    s.{v1} = {v2}\n"
                f"    return nil\n"
                f"}}}}",
            ),
            "rust": (
                f"fn {m1}(&self) -> Result<()> {{{{\n    Ok(())\n}}}}",
                f"async fn {m1}(&mut self) -> Result<()> {{{{\n"
                f"    let {v2} = self.{m2}().await?;\n"
                f'    anyhow::ensure!(!{v2}.is_empty(), "{err}");\n'
                f"    self.{v1} = {v2};\n"
                f"    Ok(())\n"
                f"}}}}",
            ),
            "typescript": (
                f"{m1}({v1}: string) {{{{\n    return this.{m2}({v1});\n}}}}",
                f"async {m1}({v1}: string): Promise<Record<string, unknown>> {{{{\n"
                f"    const {v2} = await this.{m2}({v1});\n"
                f"    if (!{v2}) throw new Error('{err}');\n"
                f"    return {{ status: 'ok', data: {v2} }};\n"
                f"}}}}",
            ),
        }
        old_str, new_str = edits.get(language, edits["python"])

        return f"""\
<tool_name>edit</tool_name>
<parameter name="file_path">{f}</parameter>
<parameter name="old_string">{old_str}</parameter>
<parameter name="new_string">{new_str}</parameter>
<result>
The file {f} has been updated successfully.
</result>
"""

    def _gen_tool_search(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)

        lang_patterns: dict[str | None, list[str]] = {
            "python": [
                f"class {r.choice(_CLASSES)}",
                f"def {r.choice(_METHODS)}",
                f"import {r.choice(_MODULES)}",
                f"async def {r.choice(_METHODS)}",
            ],
            "go": [
                f"func {r.choice(_METHODS).title()}",
                f"type {r.choice(_CLASSES)} struct",
                f'"{r.choice(list(_GO_PACKAGES))}"',
                f"func New{r.choice(_CLASSES)}",
            ],
            "rust": [
                f"fn {r.choice(_METHODS)}",
                f"pub struct {r.choice(_CLASSES)}",
                f"use {r.choice(list(_RUST_CRATES))}",
                f"impl {r.choice(_CLASSES)}",
            ],
            "typescript": [
                f"class {r.choice(_CLASSES)}",
                f"export function {r.choice(_METHODS)}",
                f"import {{ {r.choice(_CLASSES)} }}",
                f"interface {r.choice(_CLASSES)}",
            ],
        }
        patterns = lang_patterns.get(language, lang_patterns["python"])
        pattern = r.choice([*patterns, r.choice(_ERROR_MESSAGES)])

        files = r.sample(list(file_pool), min(r.randint(3, 6), len(file_pool)))
        matches = []
        for f in files:
            line_num = r.randint(1, 400)
            ctx = r.choice(_VARS)
            matches.append(f"{f}:{line_num}:    {pattern}({ctx})")

        content = "\n".join(matches)
        return f"""\
<tool_name>search</tool_name>
<parameter name="pattern">{pattern}</parameter>
<result>
{content}
</result>
"""

    def _gen_tool_bash(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        cls = r.choice(_CLASSES)
        methods = r.sample(list(_METHODS), 4)
        n_pass = r.randint(10, 80)
        n_fail = r.randint(0, 3)
        dur = r.uniform(0.5, 30.0)

        lang_cmds: dict[str | None, str] = {
            "python": "pytest -xvs tests/",
            "go": "go test -v ./...",
            "rust": "cargo test",
            "typescript": "npx vitest run",
        }
        cmd = lang_cmds.get(language, r.choice(_CLI_COMMANDS))

        test_lines = []
        for m in methods:
            passed = r.random() > 0.2
            if language == "go":
                status = "ok" if passed else "FAIL"
                test_lines.append(
                    f"--- {status}: Test{m.title()} ({r.uniform(0.001, 2.0):.3f}s)"
                )
            elif language == "rust":
                status = "ok" if passed else "FAILED"
                test_lines.append(f"test {mod}::{cls.lower()}::test_{m} ... {status}")
            elif language == "typescript":
                mark = "\u2713" if passed else "\u2717"
                test_lines.append(f"  {mark} {cls} > {m} ({r.randint(1, 500)} ms)")
            else:
                status = "PASSED" if passed else "FAILED"
                test_lines.append(f"tests/test_{mod}.py::Test{cls}::test_{m} {status}")
        test_output = "\n".join(test_lines)

        return f"""\
<tool_name>bash</tool_name>
<parameter name="command">{cmd}</parameter>
<result>
{test_output}

{n_pass} passed, {n_fail} failed in {dur:.2f}s
</result>
"""

    def _gen_bash_output(self, language: str | None = None) -> str:
        r = self._template_rng
        return r.choice(
            [
                lambda: self._gen_bash_file_explore(language=language),
                lambda: self._gen_bash_build_test(language=language),
                lambda: self._gen_bash_git_workflow(language=language),
            ]
        )()

    def _gen_bash_file_explore(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        ext_cmds: dict[str | None, tuple[str, str]] = {
            "python": ("find . -name '*.py'", "src/**/*.py"),
            "go": ("find . -name '*.go'", "**/*.go"),
            "rust": ("find . -name '*.rs'", "src/**/*.rs"),
            "typescript": ("find . -name '*.ts'", "src/**/*.ts"),
        }
        find_cmd, glob_pat = ext_cmds.get(language, ext_cmds["python"])
        cmd = r.choice(("ls -la", find_cmd, "tree src/", "wc -l"))
        files = r.sample(list(file_pool), min(r.randint(4, 8), len(file_pool)))
        file_listing = "\n".join(
            f"  {f:<42} {r.randint(1, 500):>4} lines  {r.randint(1, 50):>3}K"
            for f in files
        )
        total_lines = r.randint(500, 15000)

        return f"""\
$ {cmd}
{file_listing}
$ wc -l {glob_pat} | tail -1
  {total_lines} total
$ du -sh .
  {r.randint(1, 500)}M\t.
"""

    def _gen_bash_build_test(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        n_pkgs = r.randint(10, 200)
        build_time = r.uniform(0.5, 30.0)
        n_pass = r.randint(20, 150)
        n_fail = r.randint(0, 5)
        test_time = r.uniform(1.0, 60.0)

        lang_build: dict[str | None, tuple[str, str]] = {
            "python": (
                "pip install -e '.[dev]'",
                f"pytest tests/ -x\n  {n_pass} passed, {n_fail} failed in {test_time:.1f}s",
            ),
            "go": (
                f"go build ./cmd/{mod}\n  Compiled {n_pkgs} packages in {build_time:.1f}s",
                f"go test -v -race ./...\n  {n_pass} passed, {n_fail} failed in {test_time:.1f}s",
            ),
            "rust": (
                f"cargo build --release\n  Compiling {n_pkgs} crates\n  Finished in {build_time:.1f}s",
                f"cargo test\n  {n_pass} passed, {n_fail} failed in {test_time:.1f}s",
            ),
            "typescript": (
                f"npm ci && npm run build\n  Resolved {n_pkgs} packages in {build_time:.1f}s",
                f"npx vitest run\n  {n_pass} passed, {n_fail} failed in {test_time:.1f}s",
            ),
        }
        build_cmd, test_cmd = lang_build.get(language, lang_build["python"])

        return f"""\
$ {build_cmd}
$ {test_cmd}
$ echo $?
{"0" if n_fail == 0 else "1"}
"""

    def _gen_bash_git_workflow(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        branch = f"{r.choice(_MODULES)}/{r.choice(_METHODS)}-{r.choice(_VARS)}"
        mod = r.choice(_MODULES)
        files = r.sample(list(file_pool), min(3, len(file_pool)))
        changed = "\n".join(f"  M {f}" for f in files)
        hash1 = f"{r.randint(1000000, 9999999):07x}"
        hash2 = f"{r.randint(1000000, 9999999):07x}"

        return f"""\
$ git checkout -b {branch}
Switched to a new branch '{branch}'
$ git status
On branch {branch}
Changes not staged for commit:
{changed}
$ git add -A && git commit -m "feat: {r.choice(_METHODS)} {r.choice(_VARS)} in {mod}"
[{branch} {hash1}] feat: {r.choice(_METHODS)} {r.choice(_VARS)} in {mod}
 {len(files)} files changed, {r.randint(10, 200)} insertions(+), {r.randint(1, 50)} deletions(-)
$ git log --oneline -3
{hash1} feat: {r.choice(_METHODS)} {r.choice(_VARS)} in {mod}
{hash2} fix: {r.choice(_ERROR_MESSAGES)}
"""
