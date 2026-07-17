# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Go code-template generators (mixin for CodingContentGenerator).

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from aiperf.dataset.generator._coding_vocab import (
    _CLASSES,
    _DB_TABLES,
    _ERROR_MESSAGES,
    _GO_PACKAGES,
    _METHODS,
    _MODULES,
    _VARS,
)


class _GoMixin:
    def _gen_go_code(self) -> str:
        return self._template_rng.choice(
            [
                self._gen_go_struct,
                self._gen_go_http_handler,
                self._gen_go_errors,
                self._gen_go_test,
            ]
        )()

    def _gen_go_struct(self) -> str:
        r = self._template_rng
        pkg1, pkg2 = r.sample(list(_GO_PACKAGES), 2)
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2, v3 = r.sample(_VARS, 3)
        pkg_name = r.choice(_MODULES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
package {pkg_name}

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

func (s *{cls}) {m1.title()}(ctx context.Context) error {{{{
    s.mu.Lock()
    defer s.mu.Unlock()
    if s.{v1} == "" {{{{
        return {pkg1}.Errorf("{err}")
    }}}}
    s.{v2}++
    return nil
}}}}

func (s *{cls}) {m2.title()}() (string, error) {{{{
    s.mu.RLock()
    defer s.mu.RUnlock()
    if !s.{v3} {{{{
        return "", {pkg1}.Errorf("%w: not initialized", Err{cls})
    }}}}
    return {pkg2}.Sprintf("%s:%d", s.{v1}, s.{v2}), nil
}}}}
"""

    def _gen_go_http_handler(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        pkg_name = r.choice(_MODULES)
        table = r.choice(_DB_TABLES)
        err = r.choice(_ERROR_MESSAGES)
        status_code = r.choice(
            ["http.StatusOK", "http.StatusCreated", "http.StatusAccepted"]
        )

        return f"""\
package {pkg_name}

import (
    "encoding/json"
    "net/http"
    "log/slog"
)

type {m1.title()}Request struct {{{{
    {v1.title()} string `json:"{v1}" binding:"required"`
    {v2.title()} int    `json:"{v2}" binding:"gte=0"`
}}}}

type {m1.title()}Response struct {{{{
    Items []map[string]any `json:"items"`
    Total int              `json:"total"`
}}}}

func (h *{cls}) {m1.title()}Handler(w http.ResponseWriter, r *http.Request) {{{{
    var req {m1.title()}Request
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {{{{
        slog.Error("{err}", "handler", "{m1}")
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }}}}

    items, err := h.svc.{m2.title()}(r.Context(), req.{v1.title()})
    if err != nil {{{{
        slog.Error("{err}", "table", "{table}")
        http.Error(w, "{err}", http.StatusInternalServerError)
        return
    }}}}

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader({status_code})
    json.NewEncoder(w).Encode({m1.title()}Response{{{{Items: items, Total: len(items)}}}})
}}}}
"""

    def _gen_go_errors(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        pkg_name = r.choice(_MODULES)
        e1, e2, e3 = r.sample(_ERROR_MESSAGES, 3)
        m1 = r.choice(_METHODS)
        v1 = r.choice(_VARS)

        return f"""\
package {pkg_name}

import (
    "errors"
    "fmt"
)

var (
    Err{cls}         = errors.New("{e1}")
    ErrNot{m1.title()} = errors.New("{e2}")
    ErrInvalid{v1.title()} = errors.New("{e3}")
)

type {cls}Error struct {{{{
    Op      string
    {v1.title()} string
    Err     error
}}}}

func (e *{cls}Error) Error() string {{{{
    return fmt.Sprintf("%s %s: %v", e.Op, e.{v1.title()}, e.Err)
}}}}

func (e *{cls}Error) Unwrap() error {{{{
    return e.Err
}}}}

func Wrap{cls}Error(op, {v1} string, err error) error {{{{
    return &{cls}Error{{{{Op: op, {v1.title()}: {v1}, Err: err}}}}
}}}}
"""

    def _gen_go_test(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        pkg_name = r.choice(_MODULES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)

        return f"""\
package {pkg_name}_test

import (
    "context"
    "testing"
)

func Test{cls}_{m1.title()}(t *testing.T) {{{{
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
            got, err := s.{m1.title()}(context.Background())
            if (err != nil) != tt.wantErr {{{{
                t.Errorf("{m1.title()}() error = %v, wantErr %v", err, tt.wantErr)
                return
            }}}}
            if got != tt.want {{{{
                t.Errorf("{m1.title()}() = %v, want %v", got, tt.want)
            }}}}
        }}}})
    }}}}
}}}}

func Test{cls}_{m2.title()}_Concurrent(t *testing.T) {{{{
    s := New{cls}("{v2}")
    ctx := context.Background()
    errs := make(chan error, 10)
    for i := 0; i < 10; i++ {{{{
        go func() {{{{ errs <- s.{m2.title()}(ctx) }}}}()
    }}}}
    for i := 0; i < 10; i++ {{{{
        if err := <-errs; err != nil {{{{
            t.Errorf("concurrent {m2}: %v", err)
        }}}}
    }}}}
}}}}
"""
