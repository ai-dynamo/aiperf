# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the request-recorder helpers."""

import math
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import pytest
from aiperf_mock_server.models import (
    ChatCompletionRequest,
    CompletionRequest,
    Message,
)
from aiperf_mock_server.request_recorder import (
    RequestRecorder,
    _build_summary,
    _compute_shape_80,
    _histogram,
    _print_summary,
    _render_histogram,
    _render_vocab_lines,
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
                    "requested_osl": {
                        "min": 100.0,
                        "max": 400.0,
                        "mean": 250.0,
                        "stdev": 129.1,
                        "p50": 250.0,
                        "p90": 380.0,
                        "p95": 390.0,
                        "p99": 398.0,
                        "unique_values": 4,
                        "histogram": {
                            "bin_edges": [100.0, 250.0, 400.0],
                            "counts": [2, 2],
                        },
                    },
                    "min_tokens": None,
                },
            },
        }
        _print_summary(summary)
        out = capsys.readouterr().out
        assert (
            "ISL            mean    25.0   min    10   max    40   p50    25   p99    40"
            in out
        )
        assert (
            "Requested OSL  mean   250.0   min   100   max   400   p50   250   p99   398"
            in out
        )
        assert "ISL histogram (10 bins, n=4, 4 unique)" in out
        assert "Requested OSL histogram (2 bins, n=4, 4 unique)" in out

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
        assert "Requested OSL histogram" not in out


class _FakeTokenizer:
    """Minimal stub for unit tests that drive `RequestRecorder.record()`."""

    def __init__(self, vocab_size: int, encodings: dict[str, list[int]]) -> None:
        self._vocab_size = vocab_size
        self._encodings = encodings
        self.called_texts: list[str] = []
        self.encoded_texts: list[str] = []

    def __len__(self) -> int:
        return self._vocab_size

    def __call__(self, text: str) -> dict[str, list[int]]:
        self.called_texts.append(text)
        return {"input_ids": self.encode(text)}

    def encode(self, text: str, **_: object) -> list[int]:
        self.encoded_texts.append(text)
        return list(self._encodings.get(text, []))

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


class _ChatTemplateTokenizer(_FakeTokenizer):
    def __init__(self) -> None:
        super().__init__(vocab_size=1000, encodings={})
        self.template_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict] | None = None,
        *,
        conversation: list[dict] | None = None,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool,
    ) -> list[int]:
        rendered_messages = messages if messages is not None else conversation
        assert rendered_messages is not None
        self.template_calls.append(
            {
                "messages": rendered_messages,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "return_dict": return_dict,
            }
        )
        return [101, len(rendered_messages), 201 if add_generation_prompt else 200]


class _CountingTokenizer(_FakeTokenizer):
    def __init__(self) -> None:
        super().__init__(vocab_size=1000, encodings={})

    def encode(self, text: str, **_: object) -> list[int]:
        self.encoded_texts.append(text)
        return list(range(max(1, len(text.split()))))


class _KeywordOnlyChatTemplateTokenizer(_FakeTokenizer):
    """Mirrors older HF where `apply_chat_template`'s first parameter was
    keyword-only: positional calls raise TypeError, the `conversation=` form
    succeeds. Exercises the retry branch in `_encode_chat_prompt_ids`.
    """

    def __init__(self) -> None:
        super().__init__(vocab_size=1000, encodings={})
        self.positional_call_count: int = 0
        self.kwarg_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        *args: object,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool,
        conversation: list[dict] | None = None,
    ) -> list[int]:
        if args:
            self.positional_call_count += 1
            raise TypeError(
                "apply_chat_template() does not accept positional arguments"
            )
        assert conversation is not None
        self.kwarg_calls.append(
            {
                "conversation": conversation,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "return_dict": return_dict,
            }
        )
        return [501, len(conversation), 502]


class _WrapperWithSpecialTokenAddingBackend:
    """Mimics AIPerf's `Tokenizer` wrapper: the wrapper's own `__call__`
    suppresses backend special tokens (returning 3 ids), but the unwrapped
    `_tokenizer` backend adds two BOS/EOS sentinels (returning 5 ids).

    The recorder must prefer the wrapper's `__call__` so ISL counts stay
    consistent with `add_special_tokens=False` semantics.
    """

    class _Backend:
        def __init__(self, call_log: list[str]) -> None:
            self._log = call_log

        def __call__(self, text: str) -> dict[str, list[int]]:
            self._log.append(text)
            return {"input_ids": [101, 999, 999, 999, 102]}

        def encode(self, text: str, **_: object) -> list[int]:
            return [101, 999, 999, 999, 102]

    def __init__(self) -> None:
        self.wrapper_calls: list[str] = []
        self.backend_calls: list[str] = []
        self._tokenizer = self._Backend(self.backend_calls)

    def __len__(self) -> int:
        return 32000

    def __call__(self, text: str) -> dict[str, list[int]]:
        self.wrapper_calls.append(text)
        return {"input_ids": [999, 999, 999]}

    def encode(self, text: str, **_: object) -> list[int]:
        return [999, 999, 999]

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


def _read_jsonl(path: str) -> list[dict]:
    return [orjson.loads(line) for line in Path(path).read_bytes().splitlines()]


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

    def test_record_request_chat_uses_chat_template(self, tmp_path) -> None:
        tok = _ChatTemplateTokenizer()
        r = _make_recorder(tmp_path, tok)
        req = ChatCompletionRequest(
            model="m",
            messages=[
                Message(role="system", content="policy"),
                Message(role="user", content="hello"),
                Message(role="assistant", content="prior answer"),
            ],
            max_tokens=8,
        )

        r.record_request(
            ts=0.0,
            endpoint="/v1/chat/completions",
            request_id="x",
            model="m",
            request=req,
            stream=False,
            osl_fingerprint={"max_tokens": 8},
        )
        r._file.flush()

        assert tok.template_calls
        call = tok.template_calls[0]
        assert [m["role"] for m in call["messages"]] == [
            "system",
            "user",
            "assistant",
        ]
        assert call["add_generation_prompt"] is True
        assert call["tokenize"] is True
        assert r._vocab_counts["/v1/chat/completions"] == Counter(
            {101: 1, 3: 1, 201: 1}
        )
        row = _read_jsonl(r.path)[0]
        assert row["isl"] == 3
        assert row["tokenization_mode"] == "chat_template"
        r._file.close()

    def test_record_request_chat_typerror_retry_uses_conversation_kwarg(
        self, tmp_path
    ) -> None:
        """When `apply_chat_template` rejects positional `messages` with a
        TypeError, the recorder must retry with `conversation=` AND use the
        retry's token IDs — not silently fall through to the ChatML fallback
        (regression test for the try/except/else bug)."""
        tok = _KeywordOnlyChatTemplateTokenizer()
        r = _make_recorder(tmp_path, tok)
        req = ChatCompletionRequest(
            model="m",
            messages=[
                Message(role="system", content="policy"),
                Message(role="user", content="hello"),
            ],
        )

        r.record_request(
            ts=0.0,
            endpoint="/v1/chat/completions",
            request_id="x",
            model="m",
            request=req,
            stream=False,
            osl_fingerprint={"max_tokens": 8},
        )
        r._file.flush()

        # Positional call must have been attempted (and rejected), then retry
        # must have been invoked via the conversation= kwarg.
        assert tok.positional_call_count == 1
        assert len(tok.kwarg_calls) == 1
        assert [m["role"] for m in tok.kwarg_calls[0]["conversation"]] == [
            "system",
            "user",
        ]
        assert tok.kwarg_calls[0]["add_generation_prompt"] is True
        assert tok.kwarg_calls[0]["tokenize"] is True

        # Critical: the recorded row must reflect the RETRY's tokens
        # ([501, 2, 502]), not the ChatML fallback's tokenization of the
        # rendered string. This is what the bug got wrong.
        assert r._vocab_counts["/v1/chat/completions"] == Counter(
            {501: 1, 2: 1, 502: 1}
        )
        row = _read_jsonl(r.path)[0]
        assert row["isl"] == 3
        assert row["tokenization_mode"] == "chat_template"
        r._file.close()

    def test_record_request_chat_prompt_token_ids_skip_template(self, tmp_path) -> None:
        tok = _ChatTemplateTokenizer()
        r = _make_recorder(tmp_path, tok)
        req = ChatCompletionRequest(
            model="m",
            messages=[Message(role="user", content="hello")],
        )
        req.prompt_token_ids = [42, 43, 44]

        r.record_request(
            ts=0.0,
            endpoint="/v1/chat/completions",
            request_id="x",
            model="m",
            request=req,
            stream=False,
            osl_fingerprint={},
        )
        r._file.flush()

        assert tok.template_calls == []
        assert r._vocab_counts["/v1/chat/completions"] == Counter({42: 1, 43: 1, 44: 1})
        row = _read_jsonl(r.path)[0]
        assert row["isl"] == 3
        assert row["tokenization_mode"] == "prompt_token_ids"
        r._file.close()

    def test_record_request_completion_tokenizes_each_prompt(self, tmp_path) -> None:
        tok = _FakeTokenizer(
            vocab_size=100,
            encodings={"alpha": [1, 2], "beta": [3], "alpha\nbeta": [99]},
        )
        r = _make_recorder(tmp_path, tok)
        req = CompletionRequest(model="m", prompt=["alpha", "beta"], max_tokens=4)

        r.record_request(
            ts=0.0,
            endpoint="/v1/completions",
            request_id="x",
            model="m",
            request=req,
            stream=False,
            osl_fingerprint={"max_tokens": 4},
        )
        r._file.flush()

        assert tok.called_texts == ["alpha", "beta"]
        assert "alpha\nbeta" not in tok.called_texts
        assert r._vocab_counts["/v1/completions"] == Counter({1: 1, 2: 1, 3: 1})
        row = _read_jsonl(r.path)[0]
        assert row["isl"] == 3
        assert row["tokenization_mode"] == "tokenizer_call"
        r._file.close()

    def test_record_request_completion_prefers_wrapper_call_over_backend(
        self, tmp_path
    ) -> None:
        """Regression: `_tokenizer_call_ids` must call the wrapper's `__call__`
        (which configures `add_special_tokens=False`) rather than unwrapping to
        the raw backend (which adds them). Otherwise HF completion / TGI /
        embedding ISL counts include spurious BOS/EOS tokens.
        """
        tok = _WrapperWithSpecialTokenAddingBackend()
        r = _make_recorder(tmp_path, tok)
        req = CompletionRequest(model="m", prompt="alpha", max_tokens=4)

        r.record_request(
            ts=0.0,
            endpoint="/v1/completions",
            request_id="x",
            model="m",
            request=req,
            stream=False,
            osl_fingerprint={"max_tokens": 4},
        )
        r._file.flush()

        # Wrapper must have handled the call; the backend must NOT have been
        # bypassed by an unwrap step.
        assert tok.wrapper_calls == ["alpha"]
        assert tok.backend_calls == []
        row = _read_jsonl(r.path)[0]
        # 3 ids from the wrapper's __call__, not 5 from the backend.
        assert row["isl"] == 3
        assert row["tokenization_mode"] == "tokenizer_call"
        r._file.close()

    @pytest.mark.parametrize(
        "prompt,expected_ids",
        [
            ([11, 22, 33], [11, 22, 33]),
            ([[11, 22], [33, 44]], [11, 22, 33, 44]),
        ],
    )
    def test_record_request_completion_prompt_token_ids_skip_tokenizer(
        self, tmp_path, prompt, expected_ids
    ) -> None:
        tok = _FakeTokenizer(vocab_size=100, encodings={"11 22 33": [99]})
        r = _make_recorder(tmp_path, tok)
        req = CompletionRequest(model="m", prompt=prompt, max_tokens=4)

        r.record_request(
            ts=0.0,
            endpoint="/v1/completions",
            request_id="x",
            model="m",
            request=req,
            stream=False,
            osl_fingerprint={"max_tokens": 4},
        )
        r._file.flush()

        assert tok.called_texts == []
        assert r._vocab_counts["/v1/completions"] == Counter(expected_ids)
        row = _read_jsonl(r.path)[0]
        assert row["isl"] == len(expected_ids)
        assert row["tokenization_mode"] == "prompt_token_ids"
        r._file.close()

    def test_record_request_chat_fallback_preserves_roles(self, tmp_path) -> None:
        tok = _CountingTokenizer()
        r = _make_recorder(tmp_path, tok)
        req = ChatCompletionRequest(
            model="m",
            messages=[
                Message(role="system", content="policy"),
                Message(role="user", content="hello"),
            ],
        )

        r.record_request(
            ts=0.0,
            endpoint="/v1/chat/completions",
            request_id="x",
            model="m",
            request=req,
            stream=False,
            osl_fingerprint={},
        )
        r._file.flush()

        assert tok.encoded_texts
        rendered = tok.encoded_texts[0]
        assert "<|im_start|>system\npolicy<|im_end|>" in rendered
        assert "<|im_start|>user\nhello<|im_end|>" in rendered
        assert rendered.endswith("<|im_start|>assistant\n")
        row = _read_jsonl(r.path)[0]
        assert row["tokenization_mode"] == "chat_template_fallback"
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

    def test_observed_path_when_tokenizer_has_no_vocab_size(self, tmp_path) -> None:
        """When `len(tokenizer)` raises AND no `vocab_size`/`n_vocab` attrs
        are available, the recorder must fall back to deriving vocab_size
        from observed ids — not silently emit `vocab_distribution: null`."""

        class _SilentVocabTokenizer:
            # Looks like a tokenizer but exposes no vocab_size signal.
            def encode(self, text: str) -> list[int]:
                return [10, 20, 30] if text == "hello" else []

            def decode(self, ids: list[int]) -> str:
                return " ".join(str(i) for i in ids)

        import orjson

        tok = _SilentVocabTokenizer()
        path = tmp_path / "rec.jsonl"
        r = RequestRecorder(
            path=str(path),
            tokenizer_name="silent",
            tokenizer_revision="main",
            trust_remote_code=False,
        )
        # Bypass open()'s `from_pretrained` call by injecting the tokenizer
        # directly, then run the same vocab-size probe `open()` would run.
        r._tokenizer = tok
        try:
            r._vocab_size = len(tok)
            r._vocab_size_source = "tokenizer"
        except TypeError:
            r._vocab_size = None
            r._vocab_size_source = "observed"
        r._file = open(path, "wb")  # noqa: SIM115 — lifetime managed by test (explicit close via r.close())

        r.record(0.0, "/v1/chat/completions", "x", "m", "hello", False, {})
        r.close()

        summary = orjson.loads((tmp_path / "rec.jsonl.summary.json").read_bytes())
        vd = summary["per_endpoint"]["/v1/chat/completions"]["vocab_distribution"]
        assert vd is not None
        assert vd["vocab_size_source"] == "observed"
        # Observed vocab_size is max_observed + 1 = 31.
        assert vd["vocab_size"] == 31
        assert vd["unique_ids"] == 3
        assert vd["total_tokens"] == 3


class TestMaybeRecordRequest:
    """`_maybe_record_request` derives `stream` from the request body, but
    `TGIGenerateRequest` has no `stream` field — so streaming TGI runs would
    record `stream: null` and never increment `streamed_count` unless the
    helper handles the `/generate_stream` endpoint explicitly.
    """

    def test_generate_stream_endpoint_records_stream_true(self, tmp_path) -> None:
        from aiperf_mock_server.models import TGIGenerateRequest
        from aiperf_mock_server.request_recorder import set_global_recorder
        from aiperf_mock_server.utils import _maybe_record_request

        tok = _FakeTokenizer(vocab_size=100, encodings={"hi": [1, 2, 3]})
        r = _make_recorder(tmp_path, tok)
        set_global_recorder(r)
        try:
            req = TGIGenerateRequest(inputs="hi")
            _maybe_record_request(req, "/generate_stream", "cmpl-x", "tgi")
        finally:
            set_global_recorder(None)
        r._file.flush()

        row = _read_jsonl(r.path)[0]
        assert row["endpoint"] == "/generate_stream"
        assert row["stream"] is True
        r.close()

    def test_generate_endpoint_records_stream_falsy(self, tmp_path) -> None:
        """Sanity check: the non-streaming `/generate` endpoint stays falsy
        (TGIGenerateRequest has no `stream` field so it records as `None`).
        """
        from aiperf_mock_server.models import TGIGenerateRequest
        from aiperf_mock_server.request_recorder import set_global_recorder
        from aiperf_mock_server.utils import _maybe_record_request

        tok = _FakeTokenizer(vocab_size=100, encodings={"hi": [1, 2, 3]})
        r = _make_recorder(tmp_path, tok)
        set_global_recorder(r)
        try:
            req = TGIGenerateRequest(inputs="hi")
            _maybe_record_request(req, "/generate", "cmpl-x", "tgi")
        finally:
            set_global_recorder(None)
        r._file.flush()

        row = _read_jsonl(r.path)[0]
        assert row["endpoint"] == "/generate"
        assert not row["stream"]
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
        assert vd["max_entropy_bits"] == pytest.approx(math.log2(1000), abs=5e-5)

    def test_entropy_at_max_for_uniform_sampling(self) -> None:
        # Perfectly uniform sampling over the full vocab -> entropy_bits == log2(V).
        counts = Counter({i: 5 for i in range(64)})
        vd = _vocab_distribution(counts, 64, "tokenizer", _id_to_text)
        assert vd is not None
        assert vd["entropy_bits"] == pytest.approx(math.log2(64), abs=5e-5)
        assert vd["max_entropy_bits"] == pytest.approx(math.log2(64), abs=5e-5)

    def test_shape_80_length(self) -> None:
        counts = Counter({i: 1 for i in range(80)})
        vd = _vocab_distribution(counts, 80, "tokenizer", _id_to_text)
        assert vd is not None
        assert len(vd["shape_80"]) == 80
        assert sum(vd["shape_80"]) == 80

    def test_shape_80_stats_include_bucket_quantiles(self) -> None:
        vd = _vocab_distribution(Counter({0: 10, 1: 20}), 160, "tokenizer", _id_to_text)
        assert vd is not None
        stats = vd["shape_80_stats"]
        assert stats["min"] == 0.0
        assert stats["max"] == 30.0
        assert stats["mean"] == pytest.approx(30 / 80)
        assert stats["p50"] == 0.0

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


class TestRenderVocabShape:
    def _make_vd(
        self,
        unique_ids: int = 5,
        vocab_size: int = 1000,
        coverage_pct: float = 0.5,
        top_10_concentration_pct: float = 50.0,
        entropy_bits: float = 4.0,
        max_entropy_bits: float = 10.0,
        top_tokens: list | None = None,
        shape_80: list | None = None,
    ) -> dict:
        if top_tokens is None:
            top_tokens = [
                {"id": i, "text": f"<t{i}>", "count": 100 - i * 10} for i in range(5)
            ]
        if shape_80 is None:
            shape_80 = [10, 5, 2] + [0] * 77
        return {
            "vocab_size": vocab_size,
            "vocab_size_source": "tokenizer",
            "unique_ids": unique_ids,
            "coverage_pct": coverage_pct,
            "total_tokens": sum(shape_80),
            "top_10_concentration_pct": top_10_concentration_pct,
            "entropy_bits": entropy_bits,
            "max_entropy_bits": max_entropy_bits,
            "top_tokens": top_tokens,
            "shape_80": shape_80,
            "frequencies": {},
        }

    def test_headline_line_format(self) -> None:
        lines = _render_vocab_lines(
            self._make_vd(
                unique_ids=5234,
                vocab_size=151936,
                coverage_pct=3.4438,
                top_10_concentration_pct=47.2,
                entropy_bits=8.23,
                max_entropy_bits=17.21,
            )
        )
        assert lines[0] == (
            "    Vocab  used 5234/151936 (3.4%)  top-10 cover 47%"
            "  entropy 8.2/17.2 bits"
        )

    def test_top_line_format(self) -> None:
        vd = self._make_vd(
            top_tokens=[
                {"id": 1, "text": " the", "count": 3201},
                {"id": 2, "text": " a", "count": 2890},
            ]
        )
        lines = _render_vocab_lines(vd)
        assert lines[1] == '      top decoded tokens: " the" 3201, " a" 2890'

    def test_top_line_caps_at_5(self) -> None:
        vd = self._make_vd(
            top_tokens=[
                {"id": i, "text": f"<t{i}>", "count": 100 - i} for i in range(10)
            ]
        )
        lines = _render_vocab_lines(vd)
        # Only first 5 entries appear in the stdout line.
        assert lines[1].count(",") == 4

    def test_top_line_falls_back_to_unquoted_id_marker(self) -> None:
        vd = self._make_vd(
            top_tokens=[
                {"id": 7, "text": "<id=7>", "count": 100},
                {"id": 8, "text": " ok", "count": 50},
            ]
        )
        lines = _render_vocab_lines(vd)
        assert "<id=7> 100" in lines[1]
        assert '" ok" 50' in lines[1]

    def test_blank_line_before_shape(self) -> None:
        lines = _render_vocab_lines(self._make_vd())
        # lines[0] = Vocab headline, lines[1] = top, lines[2] = blank
        assert lines[2] == ""

    def test_shape_header_line(self) -> None:
        lines = _render_vocab_lines(self._make_vd(vocab_size=151936))
        assert lines[3] == "    vocab shape  (80 buckets over id 0..151935, log-y)"

    def test_blank_lines_around_shape_stats(self) -> None:
        lines = _render_vocab_lines(self._make_vd())
        assert lines[4] == ""
        assert lines[6] == ""

    def test_shape_stats_line(self) -> None:
        lines = _render_vocab_lines(self._make_vd(shape_80=[1] * 80))
        assert lines[5] == (
            "      bucket tokens mean     1.0   p50     1"
            "   p90     1   p95     1   p99     1"
        )

    def test_sparkline_is_80_chars(self) -> None:
        lines = _render_vocab_lines(
            self._make_vd(
                shape_80=[10, 5, 2, 1] + [0] * 76,
            )
        )
        # lines[7] is the sparkline, indented 4 spaces.
        sparkline = lines[7][4:]
        assert len(sparkline) == 80

    def test_zero_bucket_renders_as_space(self) -> None:
        shape = [10] + [0] * 79
        lines = _render_vocab_lines(self._make_vd(shape_80=shape))
        sparkline = lines[7][4:]
        # First bucket is the tallest (█); the rest are zero (space).
        assert sparkline[0] == "█"
        assert sparkline[1:] == " " * 79

    def test_log_y_makes_small_bars_visible(self) -> None:
        # One huge bucket and several small ones — linear scaling would render
        # all the small bars at ▁ or below. Log-y must lift them into visible
        # block characters.
        shape = [1000] + [1] * 79
        lines = _render_vocab_lines(self._make_vd(shape_80=shape))
        sparkline = lines[7][4:]
        assert sparkline[0] == "█"
        # The small buckets must render as non-space (i.e. a visible block).
        # log1p(1)/log1p(1000) ≈ 0.10, which maps to idx 0 (▁) under our
        # ratio*8 quantization — the smallest visible block, not space.
        block_chars = set("▁▂▃▄▅▆▇█")
        for ch in sparkline[1:]:
            assert ch in block_chars

    def test_axis_tick_line(self) -> None:
        lines = _render_vocab_lines(self._make_vd(vocab_size=151936))
        # lines[8] is the axis tick line. The leftmost label '0' starts at
        # column 4 (after the indent); the rightmost ('152K') ends at column
        # 4 + 80 = 84.
        ticks = lines[8]
        assert ticks.startswith("    0")
        assert ticks.rstrip().endswith("152K")
        # Includes the three middle ticks at 25%/50%/75% positions.
        assert "38K" in ticks
        assert "76K" in ticks
        assert "114K" in ticks


class TestBuildSummaryVocab:
    def test_endpoint_block_contains_vocab_distribution(self) -> None:
        summary = _build_summary(
            total=2,
            isls=defaultdict(list, {"/v1/chat/completions": [10, 20]}),
            osls=defaultdict(list, {"/v1/chat/completions": [5, 5]}),
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
            vocab_counts={"/v1/chat/completions": Counter({1: 3, 2: 2})},
            vocab_size=100,
            vocab_size_source="tokenizer",
            decode_fn=_id_to_text,
        )
        ep = summary["per_endpoint"]["/v1/chat/completions"]
        assert ep["vocab_distribution"] is not None
        assert ep["vocab_distribution"]["unique_ids"] == 2
        assert ep["vocab_distribution"]["total_tokens"] == 5

    def test_vocab_distribution_is_none_for_endpoint_with_no_observations(self) -> None:
        summary = _build_summary(
            total=2,
            isls=defaultdict(list, {"/v1/embeddings": [10, 20]}),
            osls=defaultdict(list),
            min_tokens=defaultdict(list),
            streamed=defaultdict(int),
            ignore_eos=defaultdict(int),
            reasoning_efforts=defaultdict(Counter),
            vocab_counts={"/v1/embeddings": Counter()},
            vocab_size=100,
            vocab_size_source="tokenizer",
            decode_fn=_id_to_text,
        )
        ep = summary["per_endpoint"]["/v1/embeddings"]
        assert ep["vocab_distribution"] is None


class TestPrintSummaryVocab:
    def _vd(self, shape: list[int] | None = None) -> dict:
        if shape is None:
            shape = [10] + [0] * 79
        return {
            "vocab_size": 1000,
            "vocab_size_source": "tokenizer",
            "unique_ids": 5,
            "coverage_pct": 0.5,
            "total_tokens": sum(shape),
            "top_10_concentration_pct": 99.0,
            "entropy_bits": 1.2,
            "max_entropy_bits": 9.97,
            "top_tokens": [
                {"id": 1, "text": " the", "count": 6},
                {"id": 2, "text": " a", "count": 2},
            ],
            "shape_80": shape,
            "frequencies": {},
        }

    def _summary(self, vd: dict | None) -> dict:
        return {
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
                            "bin_edges": [10.0, 25.0, 40.0],
                            "counts": [2, 2],
                        },
                    },
                    "requested_osl": None,
                    "min_tokens": None,
                    "vocab_distribution": vd,
                },
            },
        }

    def test_vocab_block_prints_after_histograms(self, capsys) -> None:
        _print_summary(self._summary(self._vd()))
        out = capsys.readouterr().out
        idx_isl_hist = out.index("ISL histogram")
        idx_vocab_headline = out.index("Vocab  used")
        idx_shape = out.index("    vocab shape  (")
        assert idx_isl_hist < idx_vocab_headline < idx_shape

    def test_no_vocab_lines_when_distribution_is_none(self, capsys) -> None:
        _print_summary(self._summary(None))
        out = capsys.readouterr().out
        assert "Definitions" not in out
        assert "Vocab  used" not in out
        assert "vocab shape" not in out

    def test_description_box_prints_when_vocab_distribution_exists(
        self, capsys
    ) -> None:
        _print_summary(self._summary(self._vd()))
        out = capsys.readouterr().out
        assert "Definitions" in out
        assert "OSL is the request cap" in out
        assert "entropy: token-id diversity" in out
        assert "top decoded tokens: most frequent token IDs" in out
        assert "vocab shape stats: mean/percentiles" in out

    def test_blank_lines_between_blocks(self, capsys) -> None:
        _print_summary(self._summary(self._vd()))
        out = capsys.readouterr().out
        lines = out.splitlines()
        isl_hist_idx = next(i for i, ln in enumerate(lines) if "ISL histogram" in ln)
        vocab_idx = next(i for i, ln in enumerate(lines) if "Vocab  used" in ln)
        # The line immediately before each block start should be blank, and
        # vocab gets an extra visual gap after rendered histograms.
        assert lines[isl_hist_idx - 1] == ""
        assert lines[vocab_idx - 1] == ""
        assert lines[vocab_idx - 2] == ""
