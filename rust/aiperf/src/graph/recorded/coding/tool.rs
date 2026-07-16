// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-use and bash-output template renderers.
//!
//! Every generator here takes a `lang: Option<usize>` language selector
//! (`Some(0..=3)` = python/go/rust/typescript, `None` = the pool path). The pool
//! (`_gen_tool_use_block`/`_gen_bash_output`) passes `None`; the multi-turn
//! conversation renderers pass the conversation's chosen language, so the
//! language-specific code branches AND file pools must be reproduced exactly.

use super::templates::TemplateRenderer;
use super::vocab::*;
use crate::graph::recorded::RecordedTraceError;

/// `_gen_tool_use_block`: dispatch across the read/edit/search/bash tool variants.
pub(super) fn tool_use_block(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    match r.index(4)? {
        0 => tool_read(r, None),
        1 => tool_edit(r, None),
        2 => tool_search(r, None),
        _ => tool_bash(r, None),
    }
}

/// `_gen_bash_output`: dispatch across the file-explore/build-test/git-workflow variants.
pub(super) fn bash_output(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    match r.index(3)? {
        0 => bash_file_explore(r),
        1 => bash_build_test(r),
        _ => bash_git_workflow(r),
    }
}

pub(super) fn tool_read(
    r: &mut TemplateRenderer,
    lang: Option<usize>,
) -> Result<String, RecordedTraceError> {
    let f = r.choose(file_pool(lang))?;
    let start_line = r.number(1, 200)?;
    let cls = r.pick(CLASSES)?;
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let v = r.sample(VARS, 2)?;
    let (v1, v2) = (v[0], v[1]);
    let mod_ = r.pick(MODULES)?;
    let err = r.pick(ERRORS)?;

    let m1t = TemplateRenderer::title_case(m1);
    let m2t = TemplateRenderer::title_case(m2);
    let code_lines: [String; 8] = match lang {
        Some(1) => [
            format!("func (s *{cls}) {m1t}(ctx context.Context) error {{"),
            format!("s.{v1} = {v1}"),
            format!("{v2}, err := s.{m2t}(ctx)"),
            "if err != nil {".to_string(),
            format!("return fmt.Errorf(\"{err}: %w\", err)"),
            "}".to_string(),
            "return nil".to_string(),
            String::new(),
        ],
        Some(2) => [
            format!("pub async fn {m1}(&mut self) -> Result<()> {{"),
            format!("let {v1} = self.{v2}.clone();"),
            format!("let {v2} = self.{m2}(&{v1}).await?;"),
            format!("if {v2}.is_empty() {{"),
            format!("anyhow::bail!(\"{err}\");"),
            "}".to_string(),
            "Ok(())".to_string(),
            String::new(),
        ],
        Some(3) => [
            format!("async {m1}({v1}: string): Promise<void> {{"),
            format!("this.{v1} = {v1};"),
            format!("const {v2} = await this.{m2}({v1});"),
            format!("if (!{v2}) {{"),
            format!("  throw new Error('{err}');"),
            "}".to_string(),
            format!("console.log(`{cls}.{m1}: ${{{{{v2}}}}}`);"),
            String::new(),
        ],
        _ => [
            format!("def {m1}(self, {v1}):"),
            format!("self._{v1} = {v1}"),
            format!("{v2} = {mod_}.{m2}({v1})"),
            format!("if {v1} is None:"),
            format!("    raise ValueError(\"{err}\")"),
            format!("return {v2}"),
            format!("logger.debug(f\"{cls}.{m1}: {{{{{v1}}}}}\")"),
            String::new(),
        ],
    };

    let span = r.number(15, 30)?;
    let mut lines: Vec<String> = Vec::new();
    for offset in 0..span {
        let i = start_line + offset;
        let indent = if r.random() > 0.3 { "    " } else { "        " };
        let idx = r.index(code_lines.len())?;
        let line_content = &code_lines[idx];
        lines.push(format!("{i:>6}\t{indent}{line_content}"));
    }
    let content = lines.join("\n");

    Ok(format!(
        r#"<tool_name>read</tool_name>
<parameter name="file_path">{f}</parameter>
<result>
{content}
</result>
"#
    ))
}

pub(super) fn tool_edit(
    r: &mut TemplateRenderer,
    lang: Option<usize>,
) -> Result<String, RecordedTraceError> {
    let f = r.choose(file_pool(lang))?;
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let v = r.sample(VARS, 2)?;
    let (v1, v2) = (v[0], v[1]);
    let cls = r.pick(CLASSES)?;
    let err = r.pick(ERRORS)?;
    let m1t = TemplateRenderer::title_case(m1);
    let m2t = TemplateRenderer::title_case(m2);

    let (old_str, new_str) = match lang {
        Some(1) => (
            format!("func (s *{cls}) {m1t}() error {{{{\n    return nil\n}}}}"),
            format!(
                "func (s *{cls}) {m1t}(ctx context.Context) error {{{{\n    {v2}, err := s.{m2t}(ctx)\n    if err != nil {{{{\n        return fmt.Errorf(\"{err}: %w\", err)\n    }}}}\n    s.{v1} = {v2}\n    return nil\n}}}}"
            ),
        ),
        Some(2) => (
            format!("fn {m1}(&self) -> Result<()> {{{{\n    Ok(())\n}}}}"),
            format!(
                "async fn {m1}(&mut self) -> Result<()> {{{{\n    let {v2} = self.{m2}().await?;\n    anyhow::ensure!(!{v2}.is_empty(), \"{err}\");\n    self.{v1} = {v2};\n    Ok(())\n}}}}"
            ),
        ),
        Some(3) => (
            format!("{m1}({v1}: string) {{{{\n    return this.{m2}({v1});\n}}}}"),
            format!(
                "async {m1}({v1}: string): Promise<Record<string, unknown>> {{{{\n    const {v2} = await this.{m2}({v1});\n    if (!{v2}) throw new Error('{err}');\n    return {{ status: 'ok', data: {v2} }};\n}}}}"
            ),
        ),
        _ => (
            format!("    def {m1}(self, {v1}):\n        return self._{m2}({v1})"),
            format!(
                "    async def {m1}(self, {v1}: str) -> dict:\n        try:\n            {v2} = await self._{m2}({v1})\n            if {v2} is None:\n                raise ValueError(\"{err}\")\n            return {{{{\"status\": \"ok\", \"data\": {v2}}}}}\n        except Exception as exc:\n            logger.error(\"{cls}.{m1} failed: %s\", exc)\n            raise"
            ),
        ),
    };

    Ok(format!(
        r#"<tool_name>edit</tool_name>
<parameter name="file_path">{f}</parameter>
<parameter name="old_string">{old_str}</parameter>
<parameter name="new_string">{new_str}</parameter>
<result>
The file {f} has been updated successfully.
</result>
"#
    ))
}

pub(super) fn tool_search(
    r: &mut TemplateRenderer,
    lang: Option<usize>,
) -> Result<String, RecordedTraceError> {
    let pool = file_pool(lang);
    // Python builds the ENTIRE `lang_patterns` dict literal (python/go/rust/ts)
    // before selecting one language's list, so all 16 embedded `choice(...)`
    // draws fire; only the chosen language's 4 patterns are then used.
    let py = [
        format!("class {}", r.pick(CLASSES)?),
        format!("def {}", r.pick(METHODS)?),
        format!("import {}", r.pick(MODULES)?),
        format!("async def {}", r.pick(METHODS)?),
    ];
    let go = [
        format!("func {}", TemplateRenderer::title_case(r.pick(METHODS)?)),
        format!("type {} struct", r.pick(CLASSES)?),
        format!("\"{}\"", r.pick(GO_PACKAGES)?),
        format!("func New{}", r.pick(CLASSES)?),
    ];
    let rust = [
        format!("fn {}", r.pick(METHODS)?),
        format!("pub struct {}", r.pick(CLASSES)?),
        format!("use {}", r.pick(RUST_CRATES)?),
        format!("impl {}", r.pick(CLASSES)?),
    ];
    let ts = [
        format!("class {}", r.pick(CLASSES)?),
        format!("export function {}", r.pick(METHODS)?),
        format!("import {{ {} }}", r.pick(CLASSES)?),
        format!("interface {}", r.pick(CLASSES)?),
    ];
    let patterns = match lang {
        Some(1) => go,
        Some(2) => rust,
        Some(3) => ts,
        _ => py,
    };

    // pattern = choice([*patterns, choice(_ERROR_MESSAGES)]): the error choice is
    // evaluated (drawn) while building the 5-element list, then choice indexes it.
    let err_pat = r.pick(ERRORS)?;
    let candidates = [
        patterns[0].as_str(),
        patterns[1].as_str(),
        patterns[2].as_str(),
        patterns[3].as_str(),
        err_pat,
    ];
    let pattern = candidates[r.index(candidates.len())?].to_string();

    let n = r.number(3, 6)?.min(pool.len() as i64) as usize;
    let files = r.sample(pool, n)?;
    let mut matches: Vec<String> = Vec::new();
    for f in &files {
        let line_num = r.number(1, 400)?;
        let ctx = r.pick(VARS)?;
        matches.push(format!("{f}:{line_num}:    {pattern}({ctx})"));
    }
    let content = matches.join("\n");

    Ok(format!(
        r#"<tool_name>search</tool_name>
<parameter name="pattern">{pattern}</parameter>
<result>
{content}
</result>
"#
    ))
}

pub(super) fn tool_bash(
    r: &mut TemplateRenderer,
    lang: Option<usize>,
) -> Result<String, RecordedTraceError> {
    let mod_ = r.pick(MODULES)?;
    let cls = r.pick(CLASSES)?;
    let methods = r.sample(METHODS, 4)?;
    let n_pass = r.number(10, 80)?;
    let n_fail = r.number(0, 3)?;
    let dur = r.uniform(0.5, 30.0);
    // `lang_cmds.get(language, r.choice(_CLI_COMMANDS))`: the default is always
    // evaluated (drawn), even when the language key is present.
    let fallback_cmd = r.pick(COMMANDS)?;
    let cmd = match lang {
        Some(0) => "pytest -xvs tests/",
        Some(1) => "go test -v ./...",
        Some(2) => "cargo test",
        Some(3) => "npx vitest run",
        _ => fallback_cmd,
    };
    let cls_lower = cls.to_lowercase();

    let mut test_lines: Vec<String> = Vec::new();
    for m in &methods {
        let passed = r.random() > 0.2;
        match lang {
            Some(1) => {
                let status = if passed { "ok" } else { "FAIL" };
                let t = r.uniform(0.001, 2.0);
                let mt = TemplateRenderer::title_case(m);
                test_lines.push(format!("--- {status}: Test{mt} ({t:.3}s)"));
            }
            Some(2) => {
                let status = if passed { "ok" } else { "FAILED" };
                test_lines.push(format!("test {mod_}::{cls_lower}::test_{m} ... {status}"));
            }
            Some(3) => {
                let mark = if passed { "\u{2713}" } else { "\u{2717}" };
                let ms = r.number(1, 500)?;
                test_lines.push(format!("  {mark} {cls} > {m} ({ms} ms)"));
            }
            _ => {
                let status = if passed { "PASSED" } else { "FAILED" };
                test_lines.push(format!(
                    "tests/test_{mod_}.py::Test{cls}::test_{m} {status}"
                ));
            }
        }
    }
    let test_output = test_lines.join("\n");

    Ok(format!(
        r#"<tool_name>bash</tool_name>
<parameter name="command">{cmd}</parameter>
<result>
{test_output}

{n_pass} passed, {n_fail} failed in {dur:.2}s
</result>
"#
    ))
}

fn bash_file_explore(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let find_cmd = "find . -name '*.py'";
    let glob_pat = "src/**/*.py";
    let cmds = ["ls -la", find_cmd, "tree src/", "wc -l"];
    let cmd = cmds[r.index(cmds.len())?];

    let n = r.number(4, 8)?.min(FILES.len() as i64) as usize;
    let files = r.sample(FILES, n)?;
    let mut listing: Vec<String> = Vec::new();
    for f in &files {
        let a = r.number(1, 500)?;
        let b = r.number(1, 50)?;
        listing.push(format!("  {f:<42} {a:>4} lines  {b:>3}K"));
    }
    let file_listing = listing.join("\n");
    let total_lines = r.number(500, 15000)?;
    let du = r.number(1, 500)?;
    let last = format!("  {du}M\t.");

    Ok(format!(
        r#"$ {cmd}
{file_listing}
$ wc -l {glob_pat} | tail -1
  {total_lines} total
$ du -sh .
{last}
"#
    ))
}

fn bash_build_test(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    // Python `_gen_bash_build_test` draws mod/n_pkgs/build_time even though the
    // language=None ("python") output uses none of them; the draws must fire to
    // keep the shared stream in sync.
    let _mod = r.pick(MODULES)?;
    let _n_pkgs = r.number(10, 200)?;
    let _build_time = r.uniform(0.5, 30.0);
    let n_pass = r.number(20, 150)?;
    let n_fail = r.number(0, 5)?;
    let test_time = r.uniform(1.0, 60.0);

    let build_cmd = "pip install -e '.[dev]'";
    let test_cmd =
        format!("pytest tests/ -x\n  {n_pass} passed, {n_fail} failed in {test_time:.1}s");
    let echo = if n_fail == 0 { "0" } else { "1" };

    Ok(format!(
        r#"$ {build_cmd}
$ {test_cmd}
$ echo $?
{echo}
"#
    ))
}

fn bash_git_workflow(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let b_mod = r.pick(MODULES)?;
    let b_m = r.pick(METHODS)?;
    let b_v = r.pick(VARS)?;
    let branch = format!("{b_mod}/{b_m}-{b_v}");
    let mod_ = r.pick(MODULES)?;
    let n_files = 3.min(FILES.len());
    let files = r.sample(FILES, n_files)?;
    let changed = files
        .iter()
        .map(|f| format!("  M {f}"))
        .collect::<Vec<_>>()
        .join("\n");
    let hash1 = format!("{:07x}", r.number(1_000_000, 9_999_999)?);
    let hash2 = format!("{:07x}", r.number(1_000_000, 9_999_999)?);

    let cm_m = r.pick(METHODS)?;
    let cm_v = r.pick(VARS)?;
    let lg_m = r.pick(METHODS)?;
    let lg_v = r.pick(VARS)?;
    let ins = r.number(10, 200)?;
    let del = r.number(1, 50)?;
    let log_m = r.pick(METHODS)?;
    let log_v = r.pick(VARS)?;
    let log_err = r.pick(ERRORS)?;
    let n = files.len();

    Ok(format!(
        r#"$ git checkout -b {branch}
Switched to a new branch '{branch}'
$ git status
On branch {branch}
Changes not staged for commit:
{changed}
$ git add -A && git commit -m "feat: {cm_m} {cm_v} in {mod_}"
[{branch} {hash1}] feat: {lg_m} {lg_v} in {mod_}
 {n} files changed, {ins} insertions(+), {del} deletions(-)
$ git log --oneline -3
{hash1} feat: {log_m} {log_v} in {mod_}
{hash2} fix: {log_err}
"#
    ))
}
