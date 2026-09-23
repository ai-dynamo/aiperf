# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.dataset.loader.h_cua_perf_processing import (
    IMAGE_OMITTED_TEXT,
    IMAGE_PLACEHOLDER,
    HCuaPerfFilters,
    apply_screenshot_window,
    drop_screenshot,
    fit_lengths,
    iter_selected_records,
    screenshot_slots,
    select_trace_lengths,
)
from tests.unit.dataset.loader._h_cua_perf_records import (
    image,
    image_slots,
    record,
    trajectory,
)

TURNS = {"traj-a": 3, "traj-b": 1, "traj-c": 2, "traj-d": 4}
RECORDS = [r for sid, n in TURNS.items() for r in trajectory(sid, n)]


def _image(slot: int) -> dict[str, Any]:
    return image("traj", slot)


def _record(step: int, *, image_slots: set[int] | None = None) -> dict[str, Any]:
    return record("traj", step, image_slots=image_slots)


def _image_slots(rec: dict[str, Any]) -> list[int]:
    return image_slots(rec["messages"])


class TestSelectTraceLengths:
    def test_defaults_keep_everything_in_file_order(self) -> None:
        assert select_trace_lengths(HCuaPerfFilters(), TURNS) == TURNS

    def test_min_length_drops_short_trajectories(self) -> None:
        plan = select_trace_lengths(HCuaPerfFilters(min_trace_length=2), TURNS)
        assert plan == {"traj-a": 3, "traj-c": 2, "traj-d": 4}

    def test_max_length_truncates(self) -> None:
        plan = select_trace_lengths(HCuaPerfFilters(max_trace_length=2), TURNS)
        assert plan == {"traj-a": 2, "traj-b": 1, "traj-c": 2, "traj-d": 2}

    def test_first_n_applies_after_eligibility(self) -> None:
        plan = select_trace_lengths(
            HCuaPerfFilters(min_trace_length=2), TURNS, first_n=2
        )
        assert plan == {"traj-a": 3, "traj-c": 2}

    def test_nothing_eligible_raises(self) -> None:
        with pytest.raises(ValueError, match="No trajectory has 10\\+"):
            select_trace_lengths(HCuaPerfFilters(min_trace_length=10), TURNS)

    def test_avg_length_scales_every_trajectory_by_one_factor(self) -> None:
        # Lengths [3, 1, 2, 4] scaled by 0.75 round to [2, 1, 2, 3] (mean 2.0);
        # any smaller factor rounds 2 -> 1 and drops the mean below 2.
        plan = select_trace_lengths(HCuaPerfFilters(avg_trace_length=2), TURNS)
        assert plan == {"traj-a": 2, "traj-b": 1, "traj-c": 2, "traj-d": 3}


class TestFitLengths:
    def test_mean_already_below_target_is_untouched(self) -> None:
        assert fit_lengths([1, 2, 3], avg=5, floor=1) == [1, 2, 3]

    def test_floor_is_never_crossed(self) -> None:
        lengths = fit_lengths([10, 10, 10], avg=1, floor=3)
        assert lengths == [3, 3, 3]


class TestIterSelectedRecords:
    def test_yields_planned_prefixes_in_file_order(self) -> None:
        plan = {"traj-a": 2, "traj-c": 2}
        out = list(iter_selected_records(iter(RECORDS), plan, None))
        assert [(r["session_id"], r["output_length"]) for r in out] == [
            ("traj-a", 10), ("traj-a", 11), ("traj-c", 10), ("traj-c", 11),
        ]  # fmt: skip

    def test_stops_reading_after_the_last_planned_trajectory(self) -> None:
        consumed: list[str] = []

        def records():
            for r in RECORDS:
                consumed.append(r["session_id"])
                yield r

        list(iter_selected_records(records(), {"traj-a": 3, "traj-b": 1}, None))
        assert consumed == ["traj-a"] * 3 + ["traj-b"]

    def test_non_contiguous_trajectory_is_rejected(self) -> None:
        records = RECORDS[:4] + [RECORDS[0]] + RECORDS[4:]
        with pytest.raises(ValueError, match="not contiguous"):
            list(iter_selected_records(iter(records), TURNS, None))

    def test_short_trajectory_reports_truncation(self) -> None:
        with pytest.raises(
            ValueError, match="has 2 requests but the manifest lists at least 3"
        ):
            list(iter_selected_records(iter(RECORDS[:2]), {"traj-a": 3}, None))

    def test_missing_trajectory_reports_truncation(self) -> None:
        with pytest.raises(ValueError, match="1 selected trajectories are missing"):
            list(
                iter_selected_records(
                    iter(RECORDS[:4]), {"traj-a": 3, "traj-d": 1}, None
                )
            )


class TestScreenshotWindow:
    def test_slots_span_user_and_tool_observations(self) -> None:
        messages = _record(2)["messages"]
        by_content = {id(m["content"]): m["role"] for m in messages}
        slots = screenshot_slots(messages)
        assert [by_content[id(parts)] for parts, _ in slots] == ["user", "tool", "tool"]

    def test_window_of_one_is_a_no_op_on_the_source(self) -> None:
        records = [_record(0), _record(1), _record(2)]
        untouched = copy.deepcopy(records)
        apply_screenshot_window(records, 1)
        assert records == untouched

    def test_wider_window_restores_earlier_screenshots(self) -> None:
        records = [_record(0), _record(1), _record(2)]
        apply_screenshot_window(records, 2)

        assert [_image_slots(r) for r in records] == [[0], [0, 1], [1, 2]]
        slots = screenshot_slots(records[2]["messages"])
        parts, idx = slots[1]
        assert parts[idx] == _image(1)
        assert parts[idx - 1] == {"type": "text", "text": "obs 1 before"}
        assert parts[idx + 1] == {"type": "text", "text": "obs 1 after"}

    def test_window_does_not_restore_across_a_history_reset(self) -> None:
        before_reset = [_record(0), _record(1), _record(2)]
        after_reset = [record("traj-reset", 0), record("traj-reset", 1)]
        records = before_reset + after_reset
        apply_screenshot_window(records, 3)

        assert [_image_slots(r) for r in records] == [
            [0],
            [0, 1],
            [0, 1, 2],
            [0],
            [0, 1],
        ]
        parts, idx = screenshot_slots(records[4]["messages"])[0]
        assert parts[idx] == image("traj-reset", 0)

    def test_restored_screenshot_leaves_no_empty_text_parts(self) -> None:
        system = {"role": "system", "content": "sys"}
        placeholder = {"type": "text", "text": IMAGE_OMITTED_TEXT}
        first = {"messages": [system, {"role": "user", "content": [_image(0)]}]}
        second = {
            "messages": [
                system,
                {"role": "user", "content": [placeholder]},
                {"role": "user", "content": [_image(1)]},
            ]
        }
        apply_screenshot_window([first, second], 2)

        assert second["messages"][1]["content"] == [_image(0)]

    def test_narrower_window_drops_screenshots(self) -> None:
        records = [_record(2, image_slots={0, 1, 2})]
        apply_screenshot_window(records, 1)
        assert _image_slots(records[0]) == [2]
        first_user = records[0]["messages"][1]["content"]
        assert first_user == [
            {"type": "text", "text": f"obs 0 before{IMAGE_OMITTED_TEXT}obs 0 after"}
        ]

    @pytest.mark.parametrize(
        "parts, expected_text",
        [
            param(
                [{"type": "text", "text": "a"}, _image(0), {"type": "text", "text": "b"}],
                f"a{IMAGE_OMITTED_TEXT}b",
                id="text_both_sides",
            ),
            param([_image(0)], IMAGE_OMITTED_TEXT, id="image_alone"),
        ],
    )  # fmt: skip
    def test_drop_screenshot_merges_neighbouring_text(
        self, parts: list[dict[str, Any]], expected_text: str
    ) -> None:
        drop_screenshot(parts, next(i for i, p in enumerate(parts) if "uuid" in p))
        assert parts == [{"type": "text", "text": expected_text}]
        assert IMAGE_PLACEHOLDER in parts[0]["text"]


class TestFilters:
    def test_string_values_from_the_cli_are_coerced(self) -> None:
        filters = HCuaPerfFilters.model_validate(
            {"n_screenshots": "3", "avg_trace_length": "20"}
        )
        assert (filters.n_screenshots, filters.avg_trace_length) == (3, 20.0)

    def test_max_length_below_min_length_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="below min_trace_length"):
            HCuaPerfFilters(min_trace_length=3, max_trace_length=2, avg_trace_length=1)

    def test_defaults_match_trace_processor(self) -> None:
        assert HCuaPerfFilters().min_trace_length == 1
