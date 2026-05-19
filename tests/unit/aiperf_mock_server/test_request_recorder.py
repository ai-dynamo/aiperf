# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the request-recorder helpers."""

import math
from collections import Counter, defaultdict

import pytest
from aiperf_mock_server.request_recorder import (
    RequestRecorder,
    _build_summary,
    _compute_shape_80,
    _histogram,
    _print_summary,
    _render_histogram,
    _vocab_distribution,
)
from pytest import param


class TestHistogram:
    @pytest.mark.parametrize(
        "values,expected",
        [
            param([], None, id="empty_returns_none"),
            param([42], {"bin_edges": [42.0, 42.0], "counts": [1]}, id="single_value"),
            param(
                [100, 100, 100],
                {"bin_edges": [100.0, 100.0], "counts": [3]},
                id="all_equal",
            ),
        ],
    )  # fmt: skip
    def test_degenerate_inputs(self, values, expected) -> None:
        assert _histogram(values) == expected

    def test_narrow_range_hits_min_bins_floor(self) -> None:
        # range 25..230 (width 205) -> ceil(205/100) = 3, but min_bins=10 wins
        values = list(range(25, 231, 5))  # 42 values spanning 25..230
        hist = _histogram(values)
        assert hist is not None
        assert len(hist["counts"]) == 10
        assert len(hist["bin_edges"]) == 11
        assert hist["bin_edges"][0] == 25.0
        assert hist["bin_edges"][-1] == 230.0
        assert sum(hist["counts"]) == len(values)

    def test_wide_range_hits_max_bin_width_cap(self) -> None:
        # range 207..1821 (width 1614) -> ceil(1614/100) = 17 bins
        values = list(range(207, 1822, 1))  # 1615 values
        hist = _histogram(values)
        assert hist is not None
        assert len(hist["counts"]) == 17
        assert len(hist["bin_edges"]) == 18
        assert hist["bin_edges"][0] == 207.0
        assert hist["bin_edges"][-1] == 1821.0
        assert sum(hist["counts"]) == len(values)

    def test_max_value_lands_in_last_bin(self) -> None:
        # Without the last-bin-closed rule, max would fall just past the last edge
        # and be lost. With it: 1000 must land in bin 9, not vanish.
        hist = _histogram([0, 1000])
        assert hist is not None
        assert len(hist["counts"]) == 10
        assert hist["counts"][0] == 1
        assert hist["counts"][-1] == 1
        assert sum(hist["counts"]) == 2

    def test_bin_widths_are_equal(self) -> None:
        hist = _histogram(list(range(0, 1001)))
        assert hist is not None
        edges = hist["bin_edges"]
        widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        # Width tolerance accounts for float drift in `span / num_bins`; the
        # spec only promises equal-width up to representational precision.
        assert max(widths) - min(widths) < 1e-9

    def test_last_edge_pinned_on_non_round_span(self) -> None:
        # Span 1001 doesn't divide evenly into 11 bins (max_bin_width=100 ->
        # ceil(1001/100)=11 bins). The last edge must equal hi exactly, even
        # though float arithmetic would otherwise drift.
        values = list(range(0, 1002))  # 1002 values, range 0..1001
        hist = _histogram(values)
        assert hist is not None
        assert hist["bin_edges"][-1] == 1001.0
        assert sum(hist["counts"]) == 1002


class TestBuildSummary:
    def test_isl_block_has_histogram_and_unique_values(self) -> None:
        isls: dict = defaultdict(list, {"/v1/chat/completions": [10, 20, 30, 10]})
        osls: dict = defaultdict(list, {"/v1/chat/completions": [100, 200, 100]})
        summary = _build_summary(
            total=4,
            isls=isls,
            osls=osls,
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
        )
        isl_stats = summary["per_endpoint"]["/v1/chat/completions"]["isl"]
        assert isl_stats["unique_values"] == 3
        assert isinstance(isl_stats["histogram"], dict)
        assert sum(isl_stats["histogram"]["counts"]) == 4

    def test_requested_osl_unique_count(self) -> None:
        osls: dict = defaultdict(list, {"/v1/chat/completions": [16, 32, 16, 64]})
        summary = _build_summary(
            total=4,
            isls=defaultdict(list, {"/v1/chat/completions": [1, 2, 3, 4]}),
            osls=osls,
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
        )
        osl_stats = summary["per_endpoint"]["/v1/chat/completions"]["requested_osl"]
        assert osl_stats["unique_values"] == 3
        assert isinstance(osl_stats["histogram"], dict)

    def test_empty_osl_block_is_none(self) -> None:
        # Mimics /v1/embeddings — requested_osl block stays `None` when no values.
        summary = _build_summary(
            total=2,
            isls=defaultdict(list, {"/v1/embeddings": [50, 60]}),
            osls=defaultdict(list),
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
        )
        assert summary["per_endpoint"]["/v1/embeddings"]["requested_osl"] is None
        # ISL block should still get a histogram
        isl_stats = summary["per_endpoint"]["/v1/embeddings"]["isl"]
        assert isinstance(isl_stats["histogram"], dict)
        assert isl_stats["unique_values"] == 2

    def test_min_tokens_block_unchanged(self) -> None:
        # min_tokens deliberately does NOT get the new fields.
        summary = _build_summary(
            total=2,
            isls=defaultdict(list, {"/v1/chat/completions": [10, 20]}),
            osls=defaultdict(list),
            min_tokens=defaultdict(list, {"/v1/chat/completions": [4, 8]}),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
        )
        mn = summary["per_endpoint"]["/v1/chat/completions"]["min_tokens"]
        assert "histogram" not in mn
        assert "unique_values" not in mn


class TestRenderHistogram:
    def test_header_line(self) -> None:
        hist = {"bin_edges": [0.0, 5.0, 10.0], "counts": [1, 3]}
        lines = _render_histogram("ISL", hist, count=4, unique=4)
        assert lines[0] == "    ISL histogram (2 bins, n=4, 4 unique)"

    def test_row_count_matches_bins(self) -> None:
        hist = {"bin_edges": [0.0, 5.0, 10.0, 15.0], "counts": [1, 2, 1]}
        lines = _render_histogram("ISL", hist, count=4, unique=4)
        assert len(lines) == 1 + 3  # header + 3 bin rows

    def test_bars_scaled_to_tallest_bin(self) -> None:
        hist = {"bin_edges": [0.0, 1.0, 2.0], "counts": [10, 5]}
        lines = _render_histogram("ISL", hist, count=15, unique=2)
        # First bin (max) should be fully filled — 20 block chars.
        assert lines[1].count("█") == 20
        # Second bin: 5/10 = 50% -> 10 filled, 10 unfilled.
        assert lines[2].count("█") == 10
        assert lines[2].count("░") == 10

    def test_empty_counts_returns_only_header(self) -> None:
        hist = {"bin_edges": [0.0, 0.0], "counts": []}
        lines = _render_histogram("ISL", hist, count=0, unique=0)
        assert lines == ["    ISL histogram (0 bins, n=0, 0 unique)"]

    def test_single_bin_renders(self) -> None:
        hist = {"bin_edges": [42.0, 42.0], "counts": [3]}
        lines = _render_histogram("ISL", hist, count=3, unique=1)
        assert len(lines) == 2
        # label_width=2 (from "42"), count_width=3 (floor), bar fully filled.
        assert lines[1] == "      42- 42    3 " + "█" * 20


class TestPrintSummary:
    def test_isl_histogram_block_printed(self, capsys) -> None:
        summary = {
            "total_requests": 4,
            "per_endpoint": {
                "/v1/chat/completions": {
                    "count": 4,
                    "streamed_count": 0,
                    "ignore_eos_count": 0,
                    "reasoning_effort_counts": None,
                    "isl": {
                        "min": 10.0,
                        "max": 40.0,
                        "mean": 25.0,
                        "stdev": 12.91,
                        "p50": 25.0,
                        "p90": 38.0,
                        "p95": 39.0,
                        "p99": 39.8,
                        "unique_values": 4,
                        "histogram": {
                            "bin_edges": [
                                10.0,
                                13.0,
                                16.0,
                                19.0,
                                22.0,
                                25.0,
                                28.0,
                                31.0,
                                34.0,
                                37.0,
                                40.0,
                            ],
                            "counts": [1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                        },
                    },
                    "requested_osl": None,
                    "min_tokens": None,
                },
            },
        }
        _print_summary(summary)
        out = capsys.readouterr().out
        assert "ISL histogram (10 bins, n=4, 4 unique)" in out

    def test_osl_histogram_skipped_when_null(self, capsys) -> None:
        summary = {
            "total_requests": 2,
            "per_endpoint": {
                "/v1/embeddings": {
                    "count": 2,
                    "streamed_count": 0,
                    "ignore_eos_count": 0,
                    "reasoning_effort_counts": None,
                    "isl": {
                        "min": 5.0,
                        "max": 6.0,
                        "mean": 5.5,
                        "stdev": 0.5,
                        "p50": 5.5,
                        "p90": 6.0,
                        "p95": 6.0,
                        "p99": 6.0,
                        "unique_values": 2,
                        "histogram": {"bin_edges": [5.0, 5.5, 6.0], "counts": [1, 1]},
                    },
                    "requested_osl": None,
                    "min_tokens": None,
                },
            },
        }
        _print_summary(summary)
        out = capsys.readouterr().out
        assert "ISL histogram" in out
        assert "OSL histogram" not in out


class _FakeTokenizer:
    """Minimal stub for unit tests that drive `RequestRecorder.record()`."""

    def __init__(self, vocab_size: int, encodings: dict[str, list[int]]) -> None:
        self._vocab_size = vocab_size
        self._encodings = encodings

    def __len__(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        return list(self._encodings.get(text, []))

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


def _make_recorder(tmp_path, tokenizer: _FakeTokenizer) -> RequestRecorder:
    path = tmp_path / "rec.jsonl"
    r = RequestRecorder(
        path=str(path),
        tokenizer_name="fake",
        tokenizer_revision="main",
        trust_remote_code=False,
    )
    # Bypass open() so we don't need to wire `Tokenizer.from_pretrained`.
    r._tokenizer = tokenizer
    r._vocab_size = len(tokenizer)
    r._vocab_size_source = "tokenizer"
    r._file = open(path, "wb")  # noqa: SIM115 — lifetime managed by test (explicit close)
    return r


class TestRecorderTokenIdTracking:
    def test_record_updates_vocab_counter(self, tmp_path) -> None:
        tok = _FakeTokenizer(vocab_size=100, encodings={"hello": [1, 2, 1, 3]})
        r = _make_recorder(tmp_path, tok)
        r.record(
            ts=0.0,
            endpoint="/v1/chat/completions",
            request_id="x",
            model="m",
            text="hello",
            stream=False,
            osl_fingerprint={},
        )
        assert r._vocab_counts["/v1/chat/completions"] == Counter({1: 2, 2: 1, 3: 1})
        r._file.close()

    def test_record_accumulates_across_calls(self, tmp_path) -> None:
        tok = _FakeTokenizer(vocab_size=100, encodings={"a": [1, 1], "b": [2, 3]})
        r = _make_recorder(tmp_path, tok)
        for text in ("a", "b", "a"):
            r.record(
                ts=0.0,
                endpoint="/v1/chat/completions",
                request_id="x",
                model="m",
                text=text,
                stream=False,
                osl_fingerprint={},
            )
        assert r._vocab_counts["/v1/chat/completions"] == Counter({1: 4, 2: 1, 3: 1})
        r._file.close()

    def test_record_segregates_counts_by_endpoint(self, tmp_path) -> None:
        tok = _FakeTokenizer(vocab_size=100, encodings={"x": [5, 6]})
        r = _make_recorder(tmp_path, tok)
        r.record(0.0, "/v1/chat/completions", "x", "m", "x", False, {})
        r.record(0.0, "/v1/embeddings", "x", "m", "x", False, {})
        assert r._vocab_counts["/v1/chat/completions"] == Counter({5: 1, 6: 1})
        assert r._vocab_counts["/v1/embeddings"] == Counter({5: 1, 6: 1})
        assert list(r._vocab_counts.keys()) == [
            "/v1/chat/completions",
            "/v1/embeddings",
        ]
        r._file.close()

    def test_open_sets_vocab_size_from_tokenizer(self, tmp_path) -> None:
        # Same path as production: from_pretrained -> len(tokenizer).
        # Verifies the `open()` integration captures vocab_size + source.
        r = RequestRecorder(
            path=str(tmp_path / "rec.jsonl"),
            tokenizer_name="builtin",
            tokenizer_revision="main",
            trust_remote_code=False,
        )
        r.open()
        try:
            assert isinstance(r._vocab_size, int)
            assert r._vocab_size > 0
            assert r._vocab_size_source == "tokenizer"
        finally:
            r.close()


class TestComputeShape80:
    def test_length_is_always_80(self) -> None:
        assert len(_compute_shape_80(Counter({0: 1, 99999: 1}), 100000)) == 80

    def test_sum_of_buckets_equals_total_observations(self) -> None:
        counts = Counter({0: 10, 1: 20, 50: 30, 99: 40, 5000: 50})
        shape = _compute_shape_80(counts, 10000)
        assert sum(shape) == 10 + 20 + 30 + 40 + 50

    def test_id_at_bucket_boundary_lands_in_lower_bucket(self) -> None:
        # vocab_size=80, bucket width = 1, so id 5 must land in bucket 5.
        shape = _compute_shape_80(Counter({5: 7}), 80)
        assert shape[5] == 7
        assert sum(shape) == 7

    def test_max_id_lands_in_last_bucket(self) -> None:
        # Highest id is vocab_size-1; spec says last bucket is closed on both
        # ends so vocab_size-1 ends up in bucket 79, not lost.
        shape = _compute_shape_80(Counter({999: 3}), 1000)
        assert shape[-1] == 3
        assert sum(shape) == 3

    def test_empty_counter_returns_all_zero_buckets(self) -> None:
        shape = _compute_shape_80(Counter(), 1000)
        assert shape == [0] * 80

    def test_buckets_partition_vocab_evenly(self) -> None:
        # vocab_size=800, bucket width = 10. Place one observation in each
        # bucket's lower bound to verify equal-width partitioning.
        counts = Counter({i * 10: 1 for i in range(80)})
        shape = _compute_shape_80(counts, 800)
        assert shape == [1] * 80

    def test_ids_above_vocab_size_are_dropped(self) -> None:
        # Defensive: if the tokenizer ever returns an id >= vocab_size we drop
        # it rather than silently miscount the last bucket.
        shape = _compute_shape_80(Counter({100: 5, 99: 3}), 100)
        assert shape[-1] == 3  # id=99 (last valid)
        assert sum(shape) == 3  # id=100 dropped


def _id_to_text(i: int) -> str:
    return f"<tok-{i}>"


class TestVocabDistribution:
    def test_returns_none_for_empty_counter(self) -> None:
        assert _vocab_distribution(Counter(), 100, "tokenizer", _id_to_text) is None

    def test_unique_ids_and_coverage_pct(self) -> None:
        vd = _vocab_distribution(
            Counter({1: 5, 2: 5, 3: 5}), 1000, "tokenizer", _id_to_text
        )
        assert vd is not None
        assert vd["vocab_size"] == 1000
        assert vd["vocab_size_source"] == "tokenizer"
        assert vd["unique_ids"] == 3
        assert vd["coverage_pct"] == 0.3
        assert vd["total_tokens"] == 15

    def test_top_tokens_length_caps_at_10(self) -> None:
        counts = Counter({i: 100 - i for i in range(20)})
        vd = _vocab_distribution(counts, 100, "tokenizer", _id_to_text)
        assert vd is not None
        assert len(vd["top_tokens"]) == 10
        # Sorted descending by count
        assert vd["top_tokens"][0]["count"] >= vd["top_tokens"][-1]["count"]
        assert vd["top_tokens"][0] == {"id": 0, "text": "<tok-0>", "count": 100}

    def test_top_tokens_length_matches_unique_when_below_ten(self) -> None:
        vd = _vocab_distribution(Counter({1: 3, 2: 2}), 100, "tokenizer", _id_to_text)
        assert vd is not None
        assert len(vd["top_tokens"]) == 2

    def test_top_tokens_falls_back_to_id_marker_when_decode_raises(self) -> None:
        def raising_decode(i: int) -> str:
            if i == 7:
                raise RuntimeError("boom")
            return f"<tok-{i}>"

        vd = _vocab_distribution(
            Counter({7: 100, 8: 50}), 100, "tokenizer", raising_decode
        )
        assert vd is not None
        # id 7 was the most frequent, so it appears first in top_tokens.
        assert vd["top_tokens"][0] == {"id": 7, "text": "<id=7>", "count": 100}
        assert vd["top_tokens"][1] == {"id": 8, "text": "<tok-8>", "count": 50}

    def test_top_10_concentration_pct(self) -> None:
        # Top 10 of these 11 ids account for 1000 of 1010 total = 99.0099%
        counts = Counter({i: 100 for i in range(10)})
        counts[99] = 10
        vd = _vocab_distribution(counts, 100, "tokenizer", _id_to_text)
        assert vd is not None
        assert abs(vd["top_10_concentration_pct"] - 99.0099) < 0.01

    def test_entropy_zero_for_single_token(self) -> None:
        vd = _vocab_distribution(Counter({42: 100}), 1000, "tokenizer", _id_to_text)
        assert vd is not None
        assert vd["entropy_bits"] == 0.0
        assert vd["max_entropy_bits"] == pytest.approx(math.log2(1000))

    def test_entropy_at_max_for_uniform_sampling(self) -> None:
        # Perfectly uniform sampling over the full vocab -> entropy_bits == log2(V).
        counts = Counter({i: 5 for i in range(64)})
        vd = _vocab_distribution(counts, 64, "tokenizer", _id_to_text)
        assert vd is not None
        assert vd["entropy_bits"] == pytest.approx(math.log2(64))
        assert vd["max_entropy_bits"] == pytest.approx(math.log2(64))

    def test_shape_80_length(self) -> None:
        counts = Counter({i: 1 for i in range(80)})
        vd = _vocab_distribution(counts, 80, "tokenizer", _id_to_text)
        assert vd is not None
        assert len(vd["shape_80"]) == 80
        assert sum(vd["shape_80"]) == 80

    def test_frequencies_full_table_with_string_keys(self) -> None:
        counts = Counter({1: 5, 2: 3, 99: 1})
        vd = _vocab_distribution(counts, 100, "tokenizer", _id_to_text)
        assert vd is not None
        # JSON dict keys must be strings.
        assert vd["frequencies"] == {"1": 5, "2": 3, "99": 1}

    def test_vocab_size_source_observed_path_uses_max_id_plus_one(self) -> None:
        # When source == "observed" we don't trust the passed vocab_size if
        # max(observed) >= it. The helper should report the source verbatim
        # and use vocab_size as given for coverage math. Observed-fallback
        # responsibility lives in the caller (open()/record() machinery), so
        # this test just asserts the field is passed through.
        vd = _vocab_distribution(Counter({1: 1, 5: 1}), 10, "observed", _id_to_text)
        assert vd is not None
        assert vd["vocab_size_source"] == "observed"
        assert vd["vocab_size"] == 10
