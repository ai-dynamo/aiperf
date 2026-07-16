// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Go code-template renderers.

use super::templates::TemplateRenderer;
use super::vocab::*;
use crate::graph::recorded::RecordedTraceError;

/// `_gen_go_code`: dispatch across the Go structural variants.
pub(super) fn render(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    match r.index(4)? {
        0 => go_struct(r),
        1 => go_http_handler(r),
        2 => go_errors(r),
        _ => go_test(r),
    }
}

fn go_struct(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let pkg = r.sample(GO_PACKAGES, 2)?;
    let (pkg1, pkg2) = (pkg[0], pkg[1]);
    let cls = r.pick(CLASSES)?;
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let v = r.sample(VARS, 3)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);
    let pkg_name = r.pick(MODULES)?;
    let err = r.pick(ERRORS)?;
    let m1_title = TemplateRenderer::title_case(m1);
    let m2_title = TemplateRenderer::title_case(m2);

    Ok(format!(
        r#"package {pkg_name}

import (
    "{pkg1}"
    "{pkg2}"
)

type {cls} struct {{{{
    {v1} string `json:"{v1}"`
    {v2} int    `json:"{v2},omitempty"`
    {v3} bool   `json:"-"`
    mu  sync.RWMutex
}}}}

func New{cls}({v1} string) *{cls} {{{{
    return &{cls}{{{{{v1}: {v1}}}}}
}}}}

func (s *{cls}) {m1_title}(ctx context.Context) error {{{{
    s.mu.Lock()
    defer s.mu.Unlock()
    if s.{v1} == "" {{{{
        return {pkg1}.Errorf("{err}")
    }}}}
    s.{v2}++
    return nil
}}}}

func (s *{cls}) {m2_title}() (string, error) {{{{
    s.mu.RLock()
    defer s.mu.RUnlock()
    if !s.{v3} {{{{
        return "", {pkg1}.Errorf("%w: not initialized", Err{cls})
    }}}}
    return {pkg2}.Sprintf("%s:%d", s.{v1}, s.{v2}), nil
}}}}
"#
    ))
}

fn go_http_handler(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    const STATUS: &[&str] = &["http.StatusOK", "http.StatusCreated", "http.StatusAccepted"];

    let cls = r.pick(CLASSES)?;
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let v = r.sample(VARS, 2)?;
    let (v1, v2) = (v[0], v[1]);
    let pkg_name = r.pick(MODULES)?;
    let table = r.pick(TABLES)?;
    let err = r.pick(ERRORS)?;
    let status_code = r.pick(STATUS)?;
    let m1_title = TemplateRenderer::title_case(m1);
    let m2_title = TemplateRenderer::title_case(m2);
    let v1_title = TemplateRenderer::title_case(v1);
    let v2_title = TemplateRenderer::title_case(v2);

    Ok(format!(
        r#"package {pkg_name}

import (
    "encoding/json"
    "net/http"
    "log/slog"
)

type {m1_title}Request struct {{{{
    {v1_title} string `json:"{v1}" binding:"required"`
    {v2_title} int    `json:"{v2}" binding:"gte=0"`
}}}}

type {m1_title}Response struct {{{{
    Items []map[string]any `json:"items"`
    Total int              `json:"total"`
}}}}

func (h *{cls}) {m1_title}Handler(w http.ResponseWriter, r *http.Request) {{{{
    var req {m1_title}Request
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {{{{
        slog.Error("{err}", "handler", "{m1}")
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }}}}

    items, err := h.svc.{m2_title}(r.Context(), req.{v1_title})
    if err != nil {{{{
        slog.Error("{err}", "table", "{table}")
        http.Error(w, "{err}", http.StatusInternalServerError)
        return
    }}}}

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader({status_code})
    json.NewEncoder(w).Encode({m1_title}Response{{{{Items: items, Total: len(items)}}}})
}}}}
"#
    ))
}

fn go_errors(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let cls = r.pick(CLASSES)?;
    let pkg_name = r.pick(MODULES)?;
    let e = r.sample(ERRORS, 3)?;
    let (e1, e2, e3) = (e[0], e[1], e[2]);
    let m1 = r.pick(METHODS)?;
    let v1 = r.pick(VARS)?;
    let m1_title = TemplateRenderer::title_case(m1);
    let v1_title = TemplateRenderer::title_case(v1);

    Ok(format!(
        r#"package {pkg_name}

import (
    "errors"
    "fmt"
)

var (
    Err{cls}         = errors.New("{e1}")
    ErrNot{m1_title} = errors.New("{e2}")
    ErrInvalid{v1_title} = errors.New("{e3}")
)

type {cls}Error struct {{{{
    Op      string
    {v1_title} string
    Err     error
}}}}

func (e *{cls}Error) Error() string {{{{
    return fmt.Sprintf("%s %s: %v", e.Op, e.{v1_title}, e.Err)
}}}}

func (e *{cls}Error) Unwrap() error {{{{
    return e.Err
}}}}

func Wrap{cls}Error(op, {v1} string, err error) error {{{{
    return &{cls}Error{{{{Op: op, {v1_title}: {v1}, Err: err}}}}
}}}}
"#
    ))
}

fn go_test(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let cls = r.pick(CLASSES)?;
    let pkg_name = r.pick(MODULES)?;
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let v = r.sample(VARS, 2)?;
    let (v1, v2) = (v[0], v[1]);
    let m1_title = TemplateRenderer::title_case(m1);
    let m2_title = TemplateRenderer::title_case(m2);

    Ok(format!(
        r#"package {pkg_name}_test

import (
    "context"
    "testing"
)

func Test{cls}_{m1_title}(t *testing.T) {{{{
    tests := []struct {{{{
        name    string
        {v1}    string
        want    int
        wantErr bool
    }}}}{{{{
        {{{{"valid {v1}", "test_value", 42, false}}}},
        {{{{"empty {v1}", "", 0, true}}}},
        {{{{"long {v1}", "a]very_long_value_that_exceeds_limit", 0, true}}}},
    }}}}

    for _, tt := range tests {{{{
        t.Run(tt.name, func(t *testing.T) {{{{
            s := New{cls}(tt.{v1})
            got, err := s.{m1_title}(context.Background())
            if (err != nil) != tt.wantErr {{{{
                t.Errorf("{m1_title}() error = %v, wantErr %v", err, tt.wantErr)
                return
            }}}}
            if got != tt.want {{{{
                t.Errorf("{m1_title}() = %v, want %v", got, tt.want)
            }}}}
        }}}})
    }}}}
}}}}

func Test{cls}_{m2_title}_Concurrent(t *testing.T) {{{{
    s := New{cls}("{v2}")
    ctx := context.Background()
    errs := make(chan error, 10)
    for i := 0; i < 10; i++ {{{{
        go func() {{{{ errs <- s.{m2_title}(ctx) }}}}()
    }}}}
    for i := 0; i < 10; i++ {{{{
        if err := <-errs; err != nil {{{{
            t.Errorf("concurrent {m2}: %v", err)
        }}}}
    }}}}
}}}}
"#
    ))
}
