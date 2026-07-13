// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Long-form tool generators (read-long, search-verbose, bash-verbose).

use super::templates::TemplateRenderer;
use super::vocab::*;
use crate::graph::recorded::RecordedTraceError;

/// `_gen_tool_read_long`: like `tool_read` but with 40-80 lines for realistic large file reads.
pub(super) fn tool_read_long(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let f = r.pick(FILES)?;
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

    let code_lines: Vec<String> = vec![
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
    ];

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

/// `_gen_tool_bash_verbose`: like `tool_bash` but with longer, more realistic test output.
pub(super) fn tool_bash_verbose(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let mod_ = r.pick(MODULES)?;
    let cls = r.pick(CLASSES)?;
    let k = r.number(8, 15)? as usize;
    let methods = r.sample(METHODS, k)?;
    let n_pass = r.number(30, 150)?;
    let n_fail = r.number(0, 3)?;
    let dur = r.number(200, 4500)? as f64 / 100.0;
    let cmd = r.pick(COMMANDS)?;

    let mut test_lines: Vec<String> = Vec::new();
    for m in &methods {
        let passed = r.number(1, 100)? > 15;
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
    let test_output = test_lines.join("\n");

    let warnings = if r.number(1, 100)? <= 40 {
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

/// `_gen_tool_search_verbose`: like `tool_search` but returns many matches across multiple files.
pub(super) fn tool_search_verbose(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let pattern = r.pick(CLASSES)?;

    let n = r.number(6, 12)?.min(FILES.len() as i64) as usize;
    let files = r.sample(FILES, n)?;
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
