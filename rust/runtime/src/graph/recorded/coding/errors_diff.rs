// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error-traceback and git-diff template renderers.

use super::templates::TemplateRenderer;
use super::vocab::*;
use crate::graph::recorded::RecordedTraceError;

/// `_gen_error_traceback`: dispatch across the per-language traceback variants.
///
/// When `lang` is `Some`, the traceback kind is fixed by the language
/// (`lang_to_kind`, where typescript maps to the `node` kind) with NO random
/// draw; `None` (the pool path) draws `choice(["python","go","rust","node"])`.
pub(super) fn error_traceback(
    r: &mut TemplateRenderer,
    lang: Option<usize>,
) -> Result<String, RecordedTraceError> {
    let err = r.pick(ERRORS)?;
    let cls = r.pick(CLASSES)?;
    let ms = r.sample(METHODS, 4)?;
    // python=0, go=1, rust=2, node=3; typescript(lang 3) -> node kind.
    let kind = match lang {
        Some(0) => 0,
        Some(1) => 1,
        Some(2) => 2,
        Some(3) => 3,
        _ => r.index(4)?,
    };
    let fs = r.sample(file_pool(lang), 4)?;
    let (m1, m2, m3, m4) = (ms[0], ms[1], ms[2], ms[3]);
    let (f1, f2, f3, f4) = (fs[0], fs[1], fs[2], fs[3]);
    match kind {
        0 => traceback_python(r, err, cls, m1, m2, m3, m4, f1, f2, f3, f4),
        1 => traceback_go(r, err, cls, m1, m2, m3, m4, f1, f2, f3, f4),
        2 => traceback_rust(r, err, cls, m1, m2, m3, f1, f2, f3, f4),
        _ => traceback_node(r, err, cls, m1, m2, m3, m4, f1, f2, f3, f4),
    }
}

#[allow(clippy::too_many_arguments)]
fn traceback_python(
    r: &mut TemplateRenderer,
    err: &str,
    cls: &str,
    m1: &str,
    m2: &str,
    m3: &str,
    m4: &str,
    f1: &str,
    f2: &str,
    f3: &str,
    f4: &str,
) -> Result<String, RecordedTraceError> {
    let v = r.pick(VARS)?;
    let mod_ = r.pick(MODULES)?;
    let err2 = r.pick(ERRORS)?;
    let cls2 = r.pick(CLASSES)?;
    let l1 = r.number(10, 500)?;
    let l2 = r.number(10, 300)?;
    let l3 = r.number(10, 200)?;
    let l4 = r.number(1, 200)?;
    let l5 = r.number(10, 500)?;
    let l6 = r.number(10, 300)?;

    Ok(format!(
        r#"Traceback (most recent call last):
  File "{f1}", line {l1}, in {m1}
    result = self.{m2}(data)
  File "{f2}", line {l2}, in {m2}
    {v} = await self._{m3}()
  File "{f3}", line {l3}, in _{m3}
    return {mod_}.{m4}({v})
  File "{f4}", line {l4}, in {m4}
    raise ValueError("{err}")
ValueError: {err}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "{f1}", line {l5}, in {m1}
    self._{v} = {mod_}.{m1}()
  File "{f2}", line {l6}, in __init__
    raise RuntimeError("{err2}")
RuntimeError: {cls}.{m1}() failed: {err2}

The above exception was the direct cause of the following exception:

{cls2}Error: {cls}.{m1}() aborted after {err}: {err2}
"#
    ))
}

#[allow(clippy::too_many_arguments)]
fn traceback_go(
    r: &mut TemplateRenderer,
    err: &str,
    cls: &str,
    m1: &str,
    m2: &str,
    m3: &str,
    m4: &str,
    f1: &str,
    f2: &str,
    f3: &str,
    f4: &str,
) -> Result<String, RecordedTraceError> {
    let g1 = r.number(1, 100)?;
    let g2 = r.number(101, 200)?;
    let cls2 = r.pick(CLASSES)?;
    let m1t = TemplateRenderer::title_case(m1);
    let m2t = TemplateRenderer::title_case(m2);
    let m3t = TemplateRenderer::title_case(m3);
    let m4t = TemplateRenderer::title_case(m4);
    let s1 = r.number(10, 50)?;
    let s2 = r.number(10, 300)?;
    let h1 = r.number(10000, 99999)?;
    let s3 = r.number(10, 300)?;
    let h2 = r.number(10000, 99999)?;
    let h3 = r.number(100, 999)?;
    let s4 = r.number(10, 300)?;
    let h4 = r.number(10000, 99999)?;
    let s5 = r.number(10, 300)?;
    let h5 = r.number(100, 999)?;
    let s6 = r.number(10, 100)?;

    Ok(format!(
        r#"goroutine {g1} [running]:
runtime/debug.Stack()
    /usr/local/go/src/runtime/debug/stack.go:{s1}
main.{cls}.{m1t}(...)
    {f1}:{s2}
main.{cls}.{m2t}(0xc000{h1:05x})
    {f2}:{s3}
main.{cls}.{m3t}(0xc000{h2:05x}, 0x{h3:x})
    {f3}:{s4}
panic: {err}

goroutine {g2} [select]:
main.{cls2}.{m4t}(0xc000{h4:05x})
    {f4}:{s5} +0x{h5:x}
created by main.New{cls2}
    {f4}:{s6}
"#
    ))
}

#[allow(clippy::too_many_arguments)]
fn traceback_rust(
    r: &mut TemplateRenderer,
    err: &str,
    cls: &str,
    m1: &str,
    m2: &str,
    m3: &str,
    f1: &str,
    f2: &str,
    f3: &str,
    f4: &str,
) -> Result<String, RecordedTraceError> {
    let mods = r.sample(MODULES, 3)?;
    let (mod1, mod2, mod3) = (mods[0], mods[1], mods[2]);
    let p1 = r.number(10, 300)?;
    let p2 = r.number(10, 300)?;
    let p3 = r.number(10, 300)?;
    let p4 = r.number(10, 300)?;
    let p5 = r.number(10, 300)?;
    let r1 = r.number(50, 200)?;
    let r2 = r.number(50, 200)?;

    Ok(format!(
        r#"thread 'main' panicked at '{err}', {f1}:{p1}
stack backtrace:
   0: std::panicking::begin_panic
   1: {mod1}::{cls}::{m1}
             at {f1}:{p2}
   2: {mod2}::{cls}::{m2}
             at {f2}:{p3}
   3: {mod3}::{cls}::{m3}
             at {f3}:{p4}
   4: {mod1}::main
             at {f4}:{p5}
   5: std::rt::lang_start::{{{{closure}}}}
             at /rustc/src/rt.rs:{r1}
   6: std::rt::lang_start
             at /rustc/src/rt.rs:{r2}
note: run with `RUST_BACKTRACE=1` for a full backtrace
"#
    ))
}

#[allow(clippy::too_many_arguments)]
fn traceback_node(
    r: &mut TemplateRenderer,
    err: &str,
    cls: &str,
    m1: &str,
    m2: &str,
    m3: &str,
    m4: &str,
    f1: &str,
    f2: &str,
    f3: &str,
    f4: &str,
) -> Result<String, RecordedTraceError> {
    let async_cls = r.pick(CLASSES)?;
    let async_method = r.pick(METHODS)?;
    let cls2 = r.pick(CLASSES)?;
    let a1 = r.number(10, 300)?;
    let b1 = r.number(1, 40)?;
    let a2 = r.number(10, 300)?;
    let b2 = r.number(1, 40)?;
    let a3 = r.number(10, 300)?;
    let b3 = r.number(1, 40)?;
    let t1 = r.number(50, 100)?;
    let a4 = r.number(10, 300)?;
    let err3 = r.pick(ERRORS)?;
    let a5 = r.number(10, 300)?;
    let b5 = r.number(1, 40)?;

    Ok(format!(
        r#"Error: {err}
    at {cls}.{m1} ({f1}:{a1}:{b1})
    at {cls}.{m2} ({f2}:{a2}:{b2})
    at {cls2}.{m3} ({f3}:{a3}:{b3})
    at processTicksAndRejections (node:internal/process/task_queues:{t1})
    at async {async_cls}.{async_method} ({f4}:{a4})
Caused by: {err3}
    at {cls2}.{m4} ({f3}:{a5}:{b5})
"#
    ))
}

/// Pre-sampled randomness shared across per-language diff hunk builders.
#[derive(Clone, Copy)]
struct DiffCtx {
    cls: &'static str,
    m1: &'static str,
    m2: &'static str,
    m3: &'static str,
    v1: &'static str,
    v2: &'static str,
    v3: &'static str,
    ln: i64,
    ln2: i64,
    err: &'static str,
    mod_: &'static str,
    hunk_old: i64,
    hunk_new: i64,
}

fn diff_hunks_python(c: &DiffCtx) -> (String, String, String) {
    let DiffCtx {
        cls,
        m1,
        m2,
        m3,
        v1,
        v2,
        err,
        mod_,
        ln,
        ln2,
        hunk_old,
        hunk_new,
        ..
    } = *c;
    let hunk1 = format!(
        r#"@@ -{ln},8 +{ln},14 @@ class {cls}:
     def {m1}(self):
-        {v1} = self._{m2}()
-        return {v1}
+        try:
+            {v1} = await self._{m2}()
+            if {v1} is None:
+                raise ValueError("{err}")
+            return {v1}
+        except Exception as e:
+            logger.error(f"{cls}.{m1} failed: {{{{e}}}}")
+            raise"#
    );
    let hunk2 = format!(
        r#"@@ -{ln2},5 +{ln2},9 @@ def {m2}({v1}):
     {v2} = {mod_}.{m3}({v1})
-    return {v2}
+    if not {v2}:
+        raise RuntimeError("{err}")
+    logger.info("{m2} completed: %s", {v2})
+    return {{{{"{v1}": {v2}, "status": "ok"}}}}"#
    );
    let hunk3 = format!(
        r#"@@ -{hunk_old},3 +{hunk_new},7 @@
+import logging
+from {mod_} import {cls}
+
+logger = logging.getLogger(__name__)"#
    );
    (hunk1, hunk2, hunk3)
}

fn diff_hunks_go(c: &DiffCtx) -> (String, String, String) {
    let DiffCtx {
        cls,
        m1,
        m2,
        v1,
        v2,
        err,
        ln,
        ln2,
        hunk_old,
        hunk_new,
        ..
    } = *c;
    let m1t = TemplateRenderer::title_case(m1);
    let m2t = TemplateRenderer::title_case(m2);
    let hunk1 = format!(
        r#"@@ -{ln},6 +{ln},12 @@ func (s *{cls}) {m1t}() error {{{{
-    return nil
+    {v1}, err := s.{m2t}(ctx)
+    if err != nil {{{{
+        return fmt.Errorf("{err}: %w", err)
+    }}}}
+    s.{v2} = {v1}
+    return nil"#
    );
    let hunk2 = format!(
        r#"@@ -{ln2},4 +{ln2},8 @@ func (s *{cls}) {m2t}() (string, error) {{{{
     s.mu.RLock()
     defer s.mu.RUnlock()
-    return s.{v1}, nil
+    if s.{v1} == "" {{{{
+        return "", fmt.Errorf("{err}")
+    }}}}
+    return fmt.Sprintf("%s:%d", s.{v1}, s.{v2}), nil"#
    );
    let hunk3 = format!(
        r#"@@ -{hunk_old},3 +{hunk_new},7 @@
+import (
+    "fmt"
+    "log/slog"
+)"#
    );
    (hunk1, hunk2, hunk3)
}

fn diff_hunks_rust(c: &DiffCtx) -> (String, String, String) {
    let DiffCtx {
        cls,
        m1,
        m2,
        v1,
        v2,
        err,
        mod_,
        ln,
        ln2,
        hunk_old,
        hunk_new,
        ..
    } = *c;
    let hunk1 = format!(
        r#"@@ -{ln},5 +{ln},11 @@ impl {cls} {{{{
     pub fn {m1}(&self) -> Result<()> {{{{
-        Ok(())
+        let {v1} = self.{m2}()?;
+        if {v1}.is_empty() {{{{
+            anyhow::bail!("{err}");
+        }}}}
+        tracing::info!("{m1} completed: {{}}", {v1});
+        Ok(())"#
    );
    let hunk2 = format!(
        r#"@@ -{ln2},4 +{ln2},7 @@ impl {cls} {{{{
     fn {m2}(&self) -> Result<String> {{{{
-        Ok(self.{v1}.clone())
+        let {v2} = &self.{v1};
+        anyhow::ensure!(!{v2}.is_empty(), "{err}");
+        Ok({v2}.clone())"#
    );
    let hunk3 = format!(
        r#"@@ -{hunk_old},3 +{hunk_new},6 @@
+use anyhow::Result;
+use tracing;
+use {mod_}::{cls};"#
    );
    (hunk1, hunk2, hunk3)
}

fn diff_hunks_typescript(c: &DiffCtx) -> (String, String, String) {
    let DiffCtx {
        cls,
        m1,
        m2,
        v1,
        v2,
        v3,
        err,
        mod_,
        ln,
        ln2,
        hunk_old,
        hunk_new,
        ..
    } = *c;
    let v3t = TemplateRenderer::title_case(v3);
    let hunk1 = format!(
        r#"@@ -{ln},6 +{ln},12 @@ export class {cls} {{{{
   {m1}({v1}: string) {{{{
-    return this.{m2}({v1});
+    try {{{{
+      const {v2} = await this.{m2}({v1});
+      if (!{v2}) throw new Error('{err}');
+      return {{ status: 'ok', data: {v2} }};
+    }}}} catch (err) {{{{
+      console.error(`{cls}.{m1} failed: ${{{{err}}}}`);
+      throw err;
+    }}}}"#
    );
    let hunk2 = format!(
        r#"@@ -{ln2},4 +{ln2},7 @@ export class {cls} {{{{
   private {m2}({v1}: string): {v2} {{{{
-    return this.#{v1};
+    if (!this.#{v1}) {{{{
+      throw new Error('{err}');
+    }}}}
+    return this.#{v1};"#
    );
    let hunk3 = format!(
        r#"@@ -{hunk_old},3 +{hunk_new},6 @@
+import {{ {cls} }} from './{mod_}';
+import type {{ {v3t} }} from './types';
+"#
    );
    (hunk1, hunk2, hunk3)
}

/// `_gen_git_diff`: a commit diff assembled from per-language hunk builders.
///
/// The hunk language is chosen by the `lang` parameter (Python's `language`
/// argument selects `lang_hunks[language]`), NOT by a random draw — `None`
/// (the pool path) resolves to the python hunks.
pub(super) fn git_diff(
    r: &mut TemplateRenderer,
    lang: Option<usize>,
) -> Result<String, RecordedTraceError> {
    let fs = r.sample(file_pool(lang), 3)?;
    let (f1, f2, f3) = (fs[0], fs[1], fs[2]);
    let ms = r.sample(METHODS, 3)?;
    let (m1, m2, m3) = (ms[0], ms[1], ms[2]);
    let vs = r.sample(VARS, 3)?;
    let (v1, v2, v3) = (vs[0], vs[1], vs[2]);
    let cls = r.pick(CLASSES)?;
    let ln = r.number(10, 200)?;
    let ln2 = r.number(50, 300)?;
    let err = r.pick(ERRORS)?;
    let mod_ = r.pick(MODULES)?;
    let hunk_old = r.number(1, 50)?;
    let hunk_new = r.number(1, 50)?;
    let commit_hash_n = r.number(1_000_000, 9_999_999)?;
    let commit_hash = format!("{commit_hash_n:07x}");

    let ctx = DiffCtx {
        cls,
        m1,
        m2,
        m3,
        v1,
        v2,
        v3,
        ln,
        ln2,
        err,
        mod_,
        hunk_old,
        hunk_new,
    };
    let (hunk1, hunk2, hunk3) = match lang {
        Some(1) => diff_hunks_go(&ctx),
        Some(2) => diff_hunks_rust(&ctx),
        Some(3) => diff_hunks_typescript(&ctx),
        _ => diff_hunks_python(&ctx),
    };

    let i1 = r.number(1_000_000, 9_999_999)?;
    let i2 = r.number(1_000_000, 9_999_999)?;
    let i3 = r.number(1_000_000, 9_999_999)?;
    let i4 = r.number(1_000_000, 9_999_999)?;
    let i5 = r.number(1_000_000, 9_999_999)?;
    let i6 = r.number(1_000_000, 9_999_999)?;

    Ok(format!(
        r#"commit {commit_hash}
Author: dev <dev@example.com>
Date:   Mon Jan 15 14:32:00 2025 +0000

    feat({mod_}): add async {m1} with error handling

diff --git a/{f1} b/{f1}
index {i1:07x}..{i2:07x} 100644
--- a/{f1}
+++ b/{f1}
{hunk1}
diff --git a/{f2} b/{f2}
index {i3:07x}..{i4:07x} 100644
--- a/{f2}
+++ b/{f2}
{hunk2}
diff --git a/{f3} b/{f3}
index {i5:07x}..{i6:07x} 100644
--- a/{f3}
+++ b/{f3}
{hunk3}
"#
    ))
}
