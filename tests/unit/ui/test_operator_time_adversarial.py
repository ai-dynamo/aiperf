# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial timestamp, timezone, duration, and relative-time UI tests."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
TIME_PATH = UI_ROOT / "components" / "time.js"


def _time_probe() -> dict[str, object]:
    script = f"""
        import fs from 'node:fs';

        const html = (strings, ...values) => ({{ strings: [...strings], values }});
        const intervalCalls = [];
        const cleanupCalls = [];
        const originalNow = Date.now;
        const originalSetInterval = globalThis.setInterval;
        const originalClearInterval = globalThis.clearInterval;

        Date.now = () => Date.parse('2026-05-18T12:00:00Z');
        globalThis.setInterval = (_fn, ms) => {{
          const id = `timer-${{intervalCalls.length + 1}}`;
          intervalCalls.push(ms);
          return id;
        }};
        globalThis.clearInterval = (id) => cleanupCalls.push(id);

        const useState = () => [0, () => undefined];
        const useEffect = (effect) => {{
          const cleanup = effect();
          if (typeof cleanup === 'function') cleanup();
        }};

        let src = fs.readFileSync({str(TIME_PATH)!r}, 'utf8');
        src = src.replace(/^import .*;$/gm, '');
        src = src.replace(/export function /g, 'function ');
        const time = Function(
          'html', 'useState', 'useEffect',
          `${{src}}\nreturn {{ fmtRelativeSeconds, fmtElapsedSeconds, fmtAbsolute, RelativeTime }};`,
        )(html, useState, useEffect);

        const noTimezone = '2026-05-18T12:34:56';
        const invalidMonth = '2026-13-01T00:00:00Z';
        const leapSecond = '2016-12-31T23:59:60Z';
        const hugeSeconds = 1_000_000_000_000;
        const completionBeforeStartSeconds =
          (Date.parse('2026-05-18T11:55:00Z') - Date.parse('2026-05-18T12:00:00Z')) / 1000;

        const components = {{
          noTimezone: time.RelativeTime({{ ts: noTimezone, suffix: 'ago' }}),
          invalidMonth: time.RelativeTime({{ ts: invalidMonth, className: 'bad-date' }}),
          leapSecond: time.RelativeTime({{ ts: leapSecond, className: 'leap-second' }}),
          negativeDuration: time.RelativeTime({{ seconds: -12, mode: 'elapsed' }}),
          completionBeforeStart: time.RelativeTime({{
            seconds: completionBeforeStartSeconds,
            mode: 'elapsed',
          }}),
          hugeAge: time.RelativeTime({{ ts: '1970-01-01T00:00:00Z' }}),
          rawSecondsNoTimer: time.RelativeTime({{ seconds: 30 }}),
        }};

        // Exercise each live interval band after the non-live cases above.
        time.RelativeTime({{ ts: '2026-05-18T11:59:30Z' }});
        time.RelativeTime({{ ts: '2026-05-18T11:58:55Z' }});
        time.RelativeTime({{ ts: '2026-05-18T10:00:00Z' }});
        time.RelativeTime({{ ts: '2026-05-17T10:00:00Z' }});

        Date.now = originalNow;
        globalThis.setInterval = originalSetInterval;
        globalThis.clearInterval = originalClearInterval;

        console.log(JSON.stringify({{
          absolute: {{
            noTimezone: time.fmtAbsolute(noTimezone),
            invalidMonth: time.fmtAbsolute(invalidMonth),
            leapSecond: time.fmtAbsolute(leapSecond),
          }},
          durations: {{
            negativeRelative: time.fmtRelativeSeconds(-1),
            negativeElapsed: time.fmtElapsedSeconds(-1),
            hugeRelative: time.fmtRelativeSeconds(hugeSeconds),
            hugeElapsed: time.fmtElapsedSeconds(hugeSeconds),
          }},
          components: {{
            noTimezone: {{ text: components.noTimezone.values[2], title: components.noTimezone.values[1] }},
            invalidMonth: components.invalidMonth.values,
            leapSecond: components.leapSecond.values,
            negativeDuration: components.negativeDuration.values[2],
            completionBeforeStart: components.completionBeforeStart.values[2],
            hugeAge: {{ text: components.hugeAge.values[2], title: components.hugeAge.values[1] }},
            rawSecondsNoTimer: components.rawSecondsNoTimer.values[2],
          }},
          intervalCalls,
          cleanupCalls,
        }}));
    """
    return json.loads(run_node(script))


def test_absolute_time_formats_timezone_less_and_rejects_unparseable_calendar_values() -> (
    None
):
    out = _time_probe()

    assert "2026" in out["absolute"]["noTimezone"]
    assert out["absolute"]["noTimezone"] != "2026-05-18T12:34:56"
    assert out["absolute"]["invalidMonth"] == "2026-13-01T00:00:00Z"
    assert out["absolute"]["leapSecond"] == "2016-12-31T23:59:60Z"
    assert out["components"]["invalidMonth"] == ["bad-date"]
    assert out["components"]["leapSecond"] == ["leap-second"]


def test_negative_durations_and_reversed_start_completion_order_clamp_to_zero() -> None:
    out = _time_probe()

    assert out["durations"]["negativeRelative"] == "0s"
    assert out["durations"]["negativeElapsed"] == "0s"
    assert out["components"]["negativeDuration"] == "0s"
    assert out["components"]["completionBeforeStart"] == "0s"


def test_huge_ages_format_as_days_without_live_refresh_timers() -> None:
    out = _time_probe()

    assert out["durations"]["hugeRelative"] == "11574074d"
    assert out["durations"]["hugeElapsed"] == "11574074d"
    assert out["components"]["hugeAge"]["text"].endswith("d")
    assert out["components"]["hugeAge"]["title"]


def test_relative_time_update_intervals_scale_with_age_and_ignore_raw_seconds() -> None:
    out = _time_probe()

    assert out["components"]["rawSecondsNoTimer"] == "30s"
    # First entry is the timezone-less `2026-05-18T12:34:56`, now read as UTC:
    # 35 minutes from the frozen clock, so the <1h band. Before archived
    # timestamps were interpreted as UTC this value depended on the machine's
    # timezone offset, which is the bug that fix removed.
    assert out["intervalCalls"] == [30_000, 5_000, 30_000, 300_000]
    assert out["cleanupCalls"] == ["timer-1", "timer-2", "timer-3", "timer-4"]
