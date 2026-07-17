# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error-traceback and git-diff generators (mixin for CodingContentGenerator).

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiperf.dataset.generator._coding_vocab import (
    _CLASSES,
    _ERROR_MESSAGES,
    _METHODS,
    _MODULES,
    _VARS,
)


@dataclass(frozen=True, slots=True)
class _DiffCtx:
    """Pre-sampled randomness shared across per-language diff hunk builders."""

    cls: str
    m1: str
    m2: str
    m3: str
    v1: str
    v2: str
    v3: str
    ln: int
    ln2: int
    err: str
    mod: str
    hunk_old: int
    hunk_new: int


def _diff_hunks_python(c: _DiffCtx) -> tuple[str, str, str]:
    return (
        f"""\
@@ -{c.ln},8 +{c.ln},14 @@ class {c.cls}:
     def {c.m1}(self):
-        {c.v1} = self._{c.m2}()
-        return {c.v1}
+        try:
+            {c.v1} = await self._{c.m2}()
+            if {c.v1} is None:
+                raise ValueError("{c.err}")
+            return {c.v1}
+        except Exception as e:
+            logger.error(f"{c.cls}.{c.m1} failed: {{{{e}}}}")
+            raise""",
        f"""\
@@ -{c.ln2},5 +{c.ln2},9 @@ def {c.m2}({c.v1}):
     {c.v2} = {c.mod}.{c.m3}({c.v1})
-    return {c.v2}
+    if not {c.v2}:
+        raise RuntimeError("{c.err}")
+    logger.info("{c.m2} completed: %s", {c.v2})
+    return {{{{"{c.v1}": {c.v2}, "status": "ok"}}}}""",
        f"""\
@@ -{c.hunk_old},3 +{c.hunk_new},7 @@
+import logging
+from {c.mod} import {c.cls}
+
+logger = logging.getLogger(__name__)""",
    )


def _diff_hunks_go(c: _DiffCtx) -> tuple[str, str, str]:
    return (
        f"""\
@@ -{c.ln},6 +{c.ln},12 @@ func (s *{c.cls}) {c.m1.title()}() error {{{{
-    return nil
+    {c.v1}, err := s.{c.m2.title()}(ctx)
+    if err != nil {{{{
+        return fmt.Errorf("{c.err}: %w", err)
+    }}}}
+    s.{c.v2} = {c.v1}
+    return nil""",
        f"""\
@@ -{c.ln2},4 +{c.ln2},8 @@ func (s *{c.cls}) {c.m2.title()}() (string, error) {{{{
     s.mu.RLock()
     defer s.mu.RUnlock()
-    return s.{c.v1}, nil
+    if s.{c.v1} == "" {{{{
+        return "", fmt.Errorf("{c.err}")
+    }}}}
+    return fmt.Sprintf("%s:%d", s.{c.v1}, s.{c.v2}), nil""",
        f"""\
@@ -{c.hunk_old},3 +{c.hunk_new},7 @@
+import (
+    "fmt"
+    "log/slog"
+)""",
    )


def _diff_hunks_rust(c: _DiffCtx) -> tuple[str, str, str]:
    return (
        f"""\
@@ -{c.ln},5 +{c.ln},11 @@ impl {c.cls} {{{{
     pub fn {c.m1}(&self) -> Result<()> {{{{
-        Ok(())
+        let {c.v1} = self.{c.m2}()?;
+        if {c.v1}.is_empty() {{{{
+            anyhow::bail!("{c.err}");
+        }}}}
+        tracing::info!("{c.m1} completed: {{}}", {c.v1});
+        Ok(())""",
        f"""\
@@ -{c.ln2},4 +{c.ln2},7 @@ impl {c.cls} {{{{
     fn {c.m2}(&self) -> Result<String> {{{{
-        Ok(self.{c.v1}.clone())
+        let {c.v2} = &self.{c.v1};
+        anyhow::ensure!(!{c.v2}.is_empty(), "{c.err}");
+        Ok({c.v2}.clone())""",
        f"""\
@@ -{c.hunk_old},3 +{c.hunk_new},6 @@
+use anyhow::Result;
+use tracing;
+use {c.mod}::{c.cls};""",
    )


def _diff_hunks_typescript(c: _DiffCtx) -> tuple[str, str, str]:
    return (
        f"""\
@@ -{c.ln},6 +{c.ln},12 @@ export class {c.cls} {{{{
   {c.m1}({c.v1}: string) {{{{
-    return this.{c.m2}({c.v1});
+    try {{{{
+      const {c.v2} = await this.{c.m2}({c.v1});
+      if (!{c.v2}) throw new Error('{c.err}');
+      return {{ status: 'ok', data: {c.v2} }};
+    }}}} catch (err) {{{{
+      console.error(`{c.cls}.{c.m1} failed: ${{{{err}}}}`);
+      throw err;
+    }}}}""",
        f"""\
@@ -{c.ln2},4 +{c.ln2},7 @@ export class {c.cls} {{{{
   private {c.m2}({c.v1}: string): {c.v2} {{{{
-    return this.#{c.v1};
+    if (!this.#{c.v1}) {{{{
+      throw new Error('{c.err}');
+    }}}}
+    return this.#{c.v1};""",
        f"""\
@@ -{c.hunk_old},3 +{c.hunk_new},6 @@
+import {{ {c.cls} }} from './{c.mod}';
+import type {{ {c.v3.title()} }} from './types';
+""",
    )


class _ErrorsDiffMixin:
    def _gen_error_traceback(self, language: str | None = None) -> str:
        r = self._template_rng
        err = r.choice(_ERROR_MESSAGES)
        cls = r.choice(_CLASSES)
        m1, m2, m3, m4 = r.sample(_METHODS, 4)

        lang_to_kind = {
            "python": "python",
            "go": "go",
            "rust": "rust",
            "typescript": "node",
        }
        kind = (
            lang_to_kind[language]
            if language in lang_to_kind
            else r.choice(["python", "go", "rust", "node"])
        )
        file_pool = self._file_pool(language)
        f1, f2, f3, f4 = r.sample(list(file_pool), 4)
        ms = (m1, m2, m3, m4)
        fs = (f1, f2, f3, f4)
        if kind == "python":
            return self._error_traceback_python(r, err=err, cls=cls, ms=ms, fs=fs)
        elif kind == "go":
            return self._error_traceback_go(r, err=err, cls=cls, ms=ms, fs=fs)
        elif kind == "rust":
            return self._error_traceback_rust(r, err=err, cls=cls, ms=ms, fs=fs)
        else:
            return self._error_traceback_node(r, err=err, cls=cls, ms=ms, fs=fs)

    def _error_traceback_python(
        self,
        r,
        *,
        err: str,
        cls: str,
        ms: tuple[str, str, str, str],
        fs: tuple[str, str, str, str],
    ) -> str:
        m1, m2, m3, m4 = ms
        f1, f2, f3, f4 = fs
        v = r.choice(_VARS)
        mod = r.choice(_MODULES)
        err2 = r.choice(_ERROR_MESSAGES)
        cls2 = r.choice(_CLASSES)
        return f"""\
Traceback (most recent call last):
  File "{f1}", line {r.randint(10, 500)}, in {m1}
    result = self.{m2}(data)
  File "{f2}", line {r.randint(10, 300)}, in {m2}
    {v} = await self._{m3}()
  File "{f3}", line {r.randint(10, 200)}, in _{m3}
    return {mod}.{m4}({v})
  File "{f4}", line {r.randint(1, 200)}, in {m4}
    raise ValueError("{err}")
ValueError: {err}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "{f1}", line {r.randint(10, 500)}, in {m1}
    self._{v} = {mod}.{m1}()
  File "{f2}", line {r.randint(10, 300)}, in __init__
    raise RuntimeError("{err2}")
RuntimeError: {cls}.{m1}() failed: {err2}

The above exception was the direct cause of the following exception:

{cls2}Error: {cls}.{m1}() aborted after {err}: {err2}
"""

    def _error_traceback_go(
        self,
        r,
        *,
        err: str,
        cls: str,
        ms: tuple[str, str, str, str],
        fs: tuple[str, str, str, str],
    ) -> str:
        m1, m2, m3, m4 = ms
        f1, f2, f3, f4 = fs
        g1 = r.randint(1, 100)
        g2 = r.randint(101, 200)
        cls2 = r.choice(_CLASSES)
        return f"""\
goroutine {g1} [running]:
runtime/debug.Stack()
    /usr/local/go/src/runtime/debug/stack.go:{r.randint(10, 50)}
main.{cls}.{m1.title()}(...)
    {f1}:{r.randint(10, 300)}
main.{cls}.{m2.title()}(0xc000{r.randint(10000, 99999):05x})
    {f2}:{r.randint(10, 300)}
main.{cls}.{m3.title()}(0xc000{r.randint(10000, 99999):05x}, 0x{r.randint(100, 999):x})
    {f3}:{r.randint(10, 300)}
panic: {err}

goroutine {g2} [select]:
main.{cls2}.{m4.title()}(0xc000{r.randint(10000, 99999):05x})
    {f4}:{r.randint(10, 300)} +0x{r.randint(100, 999):x}
created by main.New{cls2}
    {f4}:{r.randint(10, 100)}
"""

    def _error_traceback_rust(
        self,
        r,
        *,
        err: str,
        cls: str,
        ms: tuple[str, str, str, str],
        fs: tuple[str, str, str, str],
    ) -> str:
        m1, m2, m3, _m4 = ms
        f1, f2, f3, f4 = fs
        mod1, mod2, mod3 = r.sample(list(_MODULES), 3)
        return f"""\
thread 'main' panicked at '{err}', {f1}:{r.randint(10, 300)}
stack backtrace:
   0: std::panicking::begin_panic
   1: {mod1}::{cls}::{m1}
             at {f1}:{r.randint(10, 300)}
   2: {mod2}::{cls}::{m2}
             at {f2}:{r.randint(10, 300)}
   3: {mod3}::{cls}::{m3}
             at {f3}:{r.randint(10, 300)}
   4: {mod1}::main
             at {f4}:{r.randint(10, 300)}
   5: std::rt::lang_start::{{{{closure}}}}
             at /rustc/src/rt.rs:{r.randint(50, 200)}
   6: std::rt::lang_start
             at /rustc/src/rt.rs:{r.randint(50, 200)}
note: run with `RUST_BACKTRACE=1` for a full backtrace
"""

    def _error_traceback_node(
        self,
        r,
        *,
        err: str,
        cls: str,
        ms: tuple[str, str, str, str],
        fs: tuple[str, str, str, str],
    ) -> str:
        m1, m2, m3, m4 = ms
        f1, f2, f3, f4 = fs
        async_cls = r.choice(_CLASSES)
        async_method = r.choice(_METHODS)
        cls2 = r.choice(_CLASSES)
        return f"""\
Error: {err}
    at {cls}.{m1} ({f1}:{r.randint(10, 300)}:{r.randint(1, 40)})
    at {cls}.{m2} ({f2}:{r.randint(10, 300)}:{r.randint(1, 40)})
    at {cls2}.{m3} ({f3}:{r.randint(10, 300)}:{r.randint(1, 40)})
    at processTicksAndRejections (node:internal/process/task_queues:{r.randint(50, 100)})
    at async {async_cls}.{async_method} ({f4}:{r.randint(10, 300)})
Caused by: {r.choice(_ERROR_MESSAGES)}
    at {cls2}.{m4} ({f3}:{r.randint(10, 300)}:{r.randint(1, 40)})
"""

    def _gen_git_diff(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        f1, f2, f3 = r.sample(list(file_pool), 3)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2, v3 = r.sample(_VARS, 3)
        cls = r.choice(_CLASSES)
        ln = r.randint(10, 200)
        ln2 = r.randint(50, 300)
        err = r.choice(_ERROR_MESSAGES)
        mod = r.choice(_MODULES)
        idx = lambda: f"{r.randint(1000000, 9999999):07x}"  # noqa: E731
        hunk_old, hunk_new = r.randint(1, 50), r.randint(1, 50)
        commit_hash = f"{r.randint(1000000, 9999999):07x}"

        ctx = _DiffCtx(
            cls=cls,
            m1=m1,
            m2=m2,
            m3=m3,
            v1=v1,
            v2=v2,
            v3=v3,
            ln=ln,
            ln2=ln2,
            err=err,
            mod=mod,
            hunk_old=hunk_old,
            hunk_new=hunk_new,
        )
        builder = {
            "python": _diff_hunks_python,
            "go": _diff_hunks_go,
            "rust": _diff_hunks_rust,
            "typescript": _diff_hunks_typescript,
        }.get(language, _diff_hunks_python)
        hunk1, hunk2, hunk3 = builder(ctx)

        return f"""\
commit {commit_hash}
Author: dev <dev@example.com>
Date:   Mon Jan 15 14:32:00 2025 +0000

    feat({mod}): add async {m1} with error handling

diff --git a/{f1} b/{f1}
index {idx()}..{idx()} 100644
--- a/{f1}
+++ b/{f1}
{hunk1}
diff --git a/{f2} b/{f2}
index {idx()}..{idx()} 100644
--- a/{f2}
+++ b/{f2}
{hunk2}
diff --git a/{f3} b/{f3}
index {idx()}..{idx()} 100644
--- a/{f3}
+++ b/{f3}
{hunk3}
"""
