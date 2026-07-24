// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Long-form tool generators (read-long, search-verbose, bash-verbose).
//!
//! Each takes a `lang: Option<usize>` selector (`Some(0..=3)` =
//! Python, Go, Rust, and TypeScript. Conversation renderers pass the
//! conversation's language; the code body, file pool, and per-language draw
//! sequence is byte-exact.

use super::CodingCorpusError;
use super::templates::TemplateRenderer;
use super::vocab::*;

/// A 40-80-line file read.
pub(super) fn tool_read_long(
    r: &mut TemplateRenderer,
    lang: Option<usize>,
) -> Result<String, CodingCorpusError> {
    let f = r.choose(file_pool(lang))?;
    let start_line = r.number(1, 200)?;
    let cls = r.pick(CLASSES)?;
    let m = r.sample(METHODS, 3)?;
    let (m1, m2, m3) = (m[0], m[1], m[2]);
    let v = r.sample(VARS, 3)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);
    let mod_ = r.pick(MODULES)?;
    let err = r.pick(ERRORS)?;
    let t = r.sample(TYPES, 2)?;
    let (t1, t2) = (t[0], t[1]);
    let m1t = TemplateRenderer::title_case(m1);
    let m2t = TemplateRenderer::title_case(m2);
    let cls_lower = cls.to_lowercase();

    let code_lines: Vec<String> = match lang {
        Some(1) => vec![
            format!("type {cls} struct {{"),
            format!("\t{v1} {t1}"),
            format!("\t{v2} {t2}"),
            "\tmu   sync.RWMutex".to_string(),
            "\tlog  *zap.Logger".to_string(),
            "}".to_string(),
            String::new(),
            format!("func New{cls}({v1} {t1}, log *zap.Logger) *{cls} {{"),
            format!("\treturn &{cls}{{"),
            format!("\t\t{v1}: {v1},"),
            "\t\tlog: log,".to_string(),
            "\t}".to_string(),
            "}".to_string(),
            String::new(),
            format!("func (s *{cls}) {m1t}(ctx context.Context) error {{"),
            "\ts.mu.Lock()".to_string(),
            "\tdefer s.mu.Unlock()".to_string(),
            String::new(),
            format!("\t{v2}, err := s.{m2t}(ctx)"),
            "\tif err != nil {".to_string(),
            format!("\t\treturn fmt.Errorf(\"{err}: %w\", err)"),
            "\t}".to_string(),
            format!("\ts.{v1} = {v2}"),
            "\treturn nil".to_string(),
            "}".to_string(),
            String::new(),
            format!("func (s *{cls}) {m2t}(ctx context.Context) ({t2}, error) {{"),
            format!("\ts.log.Debug(\"{cls}.{m2t}\", zap.String(\"{v1}\", s.{v1}))"),
            format!("\tresult, err := {mod_}.{m2t}(ctx, s.{v1})"),
            "\tif err != nil {".to_string(),
            format!("\t\treturn \"\", fmt.Errorf(\"{err}: %w\", err)"),
            "\t}".to_string(),
            "\treturn result, nil".to_string(),
            "}".to_string(),
        ],
        Some(2) => vec![
            format!("pub struct {cls} {{"),
            format!("    {v1}: {t1},"),
            format!("    {v2}: Option<{t2}>,"),
            "    initialized: bool,".to_string(),
            "}".to_string(),
            String::new(),
            format!("impl {cls} {{"),
            format!("    pub fn new({v1}: {t1}) -> Self {{"),
            format!("        Self {{ {v1}, {v2}: None, initialized: false }}"),
            "    }".to_string(),
            String::new(),
            format!("    pub async fn {m1}(&mut self) -> Result<{t2}> {{"),
            format!("        anyhow::ensure!(self.initialized, \"{cls} not initialized\");"),
            format!("        let {v2} = self.{m2}().await?;"),
            format!("        if {v2}.is_empty() {{"),
            format!("            anyhow::bail!(\"{err}\");"),
            "        }".to_string(),
            format!("        Ok({v2})"),
            "    }".to_string(),
            String::new(),
            format!("    async fn {m2}(&self) -> Result<{t2}> {{"),
            format!("        let {v2} = {mod_}::{m2}(&self.{v1}).await?;"),
            format!("        tracing::debug!(\"{cls}.{m2}: {{}}\", self.{v1});"),
            format!("        Ok({v2})"),
            "    }".to_string(),
            String::new(),
            format!("    pub async fn {m3}(&mut self, {v1}: {t1}) -> Result<()> {{"),
            format!("        let existing = self.{m2}().await.ok();"),
            "        match existing {".to_string(),
            "            Some(val) if !val.is_empty() => {".to_string(),
            format!("                self.{v2} = Some(val);"),
            "            }".to_string(),
            "            _ => {".to_string(),
            format!("                {mod_}::{m3}(&{v1}).await?;"),
            "            }".to_string(),
            "        }".to_string(),
            "        Ok(())".to_string(),
            "    }".to_string(),
            "}".to_string(),
        ],
        Some(3) => vec![
            format!("export class {cls} {{"),
            format!("  private {v1}: {t1};"),
            format!("  private {v2}: {t2} | null = null;"),
            "  private initialized = false;".to_string(),
            String::new(),
            format!("  constructor({v1}: {t1}) {{"),
            format!("    this.{v1} = {v1};"),
            "  }".to_string(),
            String::new(),
            format!("  async {m1}({v1}: {t1}): Promise<{t2}> {{"),
            "    if (!this.initialized) {".to_string(),
            format!("      throw new Error('{cls} not initialized');"),
            "    }".to_string(),
            format!("    const {v2} = await this.{m2}({v1});"),
            format!("    if (!{v2}) {{"),
            format!("      throw new Error('{err}');"),
            "    }".to_string(),
            format!("    return {v2};"),
            "  }".to_string(),
            String::new(),
            format!("  private async {m2}({v1}: {t1}): Promise<{t2} | null> {{"),
            "    try {".to_string(),
            format!("      const {v2} = await {mod_}.{m2}({v1});"),
            format!("      console.debug(`{cls}.{m2}: ${{{{{v1}}}}}`);"),
            format!("      return {v2};"),
            "    } catch (err) {".to_string(),
            format!("      console.error('{err}:', err);"),
            "      throw err;".to_string(),
            "    }".to_string(),
            "  }".to_string(),
            String::new(),
            format!("  async {m3}({v1}: {t1}, {v2}: {t2}): Promise<void> {{"),
            format!("    const existing = await this.{m2}({v1}).catch(() => null);"),
            "    if (existing) {".to_string(),
            format!("      Object.assign(existing, {{ {v3}: {v2} }});"),
            "      await existing.save();".to_string(),
            "    } else {".to_string(),
            format!("      await {mod_}.{m3}({v1}, {v2});"),
            "    }".to_string(),
            "  }".to_string(),
            "}".to_string(),
        ],
        _ => vec![
            format!("class {cls}:"),
            format!("    \"\"\"{cls} handles {m1} operations for {mod_}.\"\"\""),
            String::new(),
            format!("    _default_{v3} = 64"),
            String::new(),
            format!("    def __init__(self, {v1}: {t1}, {v2}: {t2} = None):"),
            format!("        self._{v1} = {v1}"),
            format!("        self._{v2} = {v2}"),
            format!("        self._{v3} = self._default_{v3}"),
            "        self._initialized = False".to_string(),
            "        self._lock = asyncio.Lock()".to_string(),
            String::new(),
            format!("    async def {m1}(self, {v1}: {t1}) -> {t2}:"),
            "        if not self._initialized:".to_string(),
            format!("            raise RuntimeError(\"{cls} not initialized\")"),
            "        async with self._lock:".to_string(),
            format!("            {v2} = await self._{m2}({v1})"),
            format!("            if {v2} is None:"),
            format!("                raise ValueError(\"{err}\")"),
            format!("            return {v2}"),
            String::new(),
            format!("    async def _{m2}(self, {v1}: {t1}) -> {t2}:"),
            "        try:".to_string(),
            format!("            {v2} = await {mod_}.{m2}({v1})"),
            format!("            logger.debug(f\"{cls}.{m2}: {{{{{v1}}}}}\")"),
            format!("            return {v2}"),
            "        except Exception as e:".to_string(),
            format!("            logger.error(\"{err}: %s\", e)"),
            format!("            raise ValueError(\"{err}\") from e"),
            String::new(),
            format!("    async def {m3}(self, {v1}: {t1}, {v2}: {t2}) -> None:"),
            format!("        if {v1} is None:"),
            "            return".to_string(),
            format!("        existing = await self._{m2}({v1})"),
            "        if existing is not None:".to_string(),
            format!("            existing.{v3} = {v2}"),
            "            await existing.save()".to_string(),
            "        else:".to_string(),
            format!("            await {mod_}.{m3}({v1}, {v2})"),
            String::new(),
            format!("    def {m1}_sync(self) -> None:"),
            "        self._initialized = True".to_string(),
            format!("        self._{v3} = 0"),
        ],
    };
    let _ = (m1t, m2t, cls_lower);

    let mut lines: Vec<String> = Vec::new();
    for (offset, content) in code_lines.iter().enumerate() {
        let i = start_line + offset as i64;
        lines.push(format!("{i:>6}\t{content}"));
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

/// A verbose shell transcript.
pub(super) fn tool_bash_verbose(
    r: &mut TemplateRenderer,
    lang: Option<usize>,
) -> Result<String, CodingCorpusError> {
    let mod_ = r.pick(MODULES)?;
    let cls = r.pick(CLASSES)?;
    let k = r.number(8, 15)? as usize;
    let methods = r.sample(METHODS, k)?;
    let n_pass = r.number(30, 150)?;
    let n_fail = r.number(0, 3)?;
    let dur = r.uniform(2.0, 45.0);
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
        let passed = r.random() > 0.15;
        // `t` is drawn before the language branch.
        let t = r.uniform(0.001, 3.0);
        match lang {
            Some(1) => {
                let status = if passed { "ok" } else { "FAIL" };
                let mt = TemplateRenderer::title_case(m);
                test_lines.push(format!("--- {status}: Test{mt} ({t:.3}s)"));
                if !passed {
                    let v = r.pick(VARS)?;
                    let ln = r.number(20, 300)?;
                    test_lines.push(format!(
                        "        {mod_}_test.go:{ln}: expected {v} to be non-nil"
                    ));
                }
            }
            Some(2) => {
                let status = if passed { "ok" } else { "FAILED" };
                test_lines.push(format!("test {mod_}::{cls_lower}::test_{m} ... {status}"));
                if !passed {
                    let err = r.pick(ERRORS)?;
                    test_lines.push(format!("  thread '{m}' panicked at '{err}'"));
                }
            }
            Some(3) => {
                let mark = if passed { "\u{2713}" } else { "\u{2717}" };
                let ms = r.number(1, 800)?;
                test_lines.push(format!("  {mark} {cls} > {m} ({ms} ms)"));
                if !passed {
                    test_lines.push("    Expected: true\n    Received: false".to_string());
                }
            }
            _ => {
                let status = if passed { "PASSED" } else { "FAILED" };
                test_lines.push(format!(
                    "tests/test_{mod_}.py::Test{cls}::test_{m} {status}"
                ));
                if !passed {
                    let err = r.pick(ERRORS)?;
                    let v = r.pick(VARS)?;
                    test_lines.push(format!(
                        "    FAILED tests/test_{mod_}.py::Test{cls}::test_{m}"
                    ));
                    test_lines.push(format!("    AssertionError: assert {v} == expected"));
                    test_lines.push(format!("      where {v} = {cls}().{m}()"));
                    test_lines.push(format!("    {err}"));
                }
            }
        }
    }
    let test_output = test_lines.join("\n");

    let warnings = if r.random() < 0.4 {
        let w_count = r.number(1, 5)?;
        format!("\n\n{w_count} warning(s)")
    } else {
        String::new()
    };

    Ok(format!(
        r#"<tool_name>bash</tool_name>
<parameter name="command">{cmd}</parameter>
<result>
{test_output}
{warnings}
========================= {n_pass} passed, {n_fail} failed in {dur:.2}s =========================
</result>
"#
    ))
}

/// Search output with many matches across multiple files.
pub(super) fn tool_search_verbose(
    r: &mut TemplateRenderer,
    lang: Option<usize>,
) -> Result<String, CodingCorpusError> {
    let pool = file_pool(lang);
    let pattern = r.pick(CLASSES)?;

    let n = r.number(6, 12)?.min(pool.len() as i64) as usize;
    let files = r.sample(pool, n)?;
    let mut matches: Vec<String> = Vec::new();
    for f in &files {
        let n_hits = r.number(1, 4)?;
        for _ in 0..n_hits {
            let line_num = r.number(1, 500)?;
            let v = r.pick(VARS)?;
            let m = r.pick(METHODS)?;
            let c_cls = r.pick(CLASSES)?;
            let c_mod = r.pick(MODULES)?;
            let candidates = [
                format!("class {pattern}({c_cls}):"),
                format!("    {m} = {pattern}({v})"),
                format!("from {c_mod} import {pattern}"),
                format!("    self._{v} = {pattern}.{m}()"),
                format!("    result: {pattern} = await svc.{m}({v})"),
                format!("# TODO: refactor {pattern} to use async"),
            ];
            let ctx = &candidates[r.index(candidates.len())?];
            matches.push(format!("{f}:{line_num}:{ctx}"));
        }
    }
    let n_matches = matches.len();
    let n_files = files.len();
    let content = matches.join("\n");

    Ok(format!(
        r#"<tool_name>search</tool_name>
<parameter name="pattern">{pattern}</parameter>
<result>
Found {n_matches} matches in {n_files} files:

{content}
</result>
"#
    ))
}
