# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure unit tests for mock-server accuracy-dataset mode."""

from __future__ import annotations

from dataclasses import replace

import pytest
from aiperf_mock_server.accuracy import (
    AccuracyDataset,
    AccuracyDecision,
    AccuracyFormat,
    AccuracyLive,
    AccuracyMatch,
    AccuracySettings,
    Adversarial,
    bump_number,
    format_correct,
)
from pytest import param

pytestmark = pytest.mark.server_unit


def _settings(**kwargs: object) -> AccuracySettings:
    return replace(AccuracySettings(random_seed=1234), **kwargs)  # type: ignore[arg-type]


def _dataset(body: str, **kwargs: object) -> AccuracyDataset:
    return AccuracyDataset.from_jsonl(body, _settings(**kwargs))


def test_parses_aliases_and_normalizes() -> None:
    body = (
        '{"question": "What is 2+2?", "answer": "4", "task": "math"}\n'
        '{"text": "Capital of France?", "ground_truth": " B ", "subject": "geo"}'
    )
    ds = _dataset(body)
    assert ds.len() == 2
    entry = ds.lookup("What is 2+2?")
    assert entry is not None
    assert entry.gold == "4"
    assert entry.task == "math"


def test_lookup_substring_fallback_handles_wrapped_prompt() -> None:
    body = '{"prompt": "Capital of France?", "answer": "Paris"}'
    ds = _dataset(body)
    assert ds.lookup("You are an expert.\n\nCapital of France?") is not None


def test_exact_mode_rejects_wrapped_prompt() -> None:
    body = '{"prompt": "Capital of France?", "answer": "Paris"}'
    ds = _dataset(body, match_mode=AccuracyMatch.EXACT)
    assert ds.lookup("Capital of France?") is not None
    assert ds.lookup("You are an expert. Capital of France?") is None
    assert ds.lookup("  Capital   of   France?  ") is not None


def test_case_insensitive_modes_fold_case() -> None:
    body = '{"prompt": "Capital of France?", "answer": "Paris"}'
    assert _dataset(body).lookup("CAPITAL OF FRANCE?") is None
    assert (
        _dataset(body, match_mode=AccuracyMatch.EXACT_CI).lookup("CAPITAL OF FRANCE?")
        is not None
    )
    assert (
        _dataset(body, match_mode=AccuracyMatch.SUBSTRING_CI).lookup(
            "Note: CAPITAL OF FRANCE?"
        )
        is not None
    )


def test_dedicated_match_key_matches_a_stable_fragment() -> None:
    body = (
        '{"prompt": "irrelevant", "match_key": "q_id_4217", "answer": "C", "task": "t"}'
    )
    ds = _dataset(body, match_mode=AccuracyMatch.SUBSTRING)
    wire = "Few-shot examples...\nQuestion [q_id_4217]: pick one.\nAnswer:"
    entry = ds.lookup(wire)
    assert entry is not None
    assert entry.gold == "C"
    decision = ds.decide(entry)
    assert decision.correct
    assert decision.content == "C"


def test_verdict_is_stable_across_prompt_wrappings() -> None:
    body = '{"prompt": "the q", "answer": "B"}'
    ds = _dataset(
        body,
        match_mode=AccuracyMatch.SUBSTRING,
        default_format=AccuracyFormat.MMLU,
        correct_rate=0.5,
    )
    a = ds.decide(ds.lookup("prefix one — the q"))  # type: ignore[arg-type]
    b = ds.decide(
        ds.lookup("a totally different prefix the q suffix")  # type: ignore[arg-type]
    )
    assert a.content == b.content
    assert a.correct == b.correct


def test_empty_dataset_is_an_error() -> None:
    settings = _settings()
    with pytest.raises(ValueError, match="no usable rows"):
        AccuracyDataset.from_jsonl('{"foo":1}\n', settings)
    with pytest.raises(ValueError, match="no usable rows"):
        AccuracyDataset.from_jsonl("", settings)


@pytest.mark.parametrize(
    ("fmt", "gold", "expected"),
    [
        param(AccuracyFormat.MMLU, "B", "The answer is (B)", id="mmlu"),
        param(AccuracyFormat.MMLU_PRO, "J", "The answer is (J)", id="mmlu_pro"),
        param(AccuracyFormat.GSM8K, "42", "#### 42", id="gsm8k"),
        param(AccuracyFormat.MATH, "42", r"\boxed{42}", id="math"),
        param(AccuracyFormat.EXACT_MATCH, "True", "True", id="exact"),
        param(AccuracyFormat.PASSTHROUGH, "hi", "hi", id="passthrough"),
    ],
)  # fmt: skip
def test_correct_answers_are_grader_formatted(
    fmt: AccuracyFormat, gold: str, expected: str
) -> None:
    assert format_correct(fmt, gold) == expected
    body = '{"prompt":"q","answer":"B"}'
    ds = _dataset(body, default_format=AccuracyFormat.MMLU, correct_rate=1.0)
    decision = ds.decide(ds.lookup("q"))  # type: ignore[arg-type]
    assert decision.correct
    assert decision.content == "The answer is (B)"
    assert decision.reasoning_content is None


def test_wrong_answers_differ_from_gold() -> None:
    body = '{"prompt":"q","answer":"B"}'
    ds = _dataset(body, default_format=AccuracyFormat.MMLU, correct_rate=0.0)
    decision = ds.decide(ds.lookup("q"))  # type: ignore[arg-type]
    assert not decision.correct
    assert decision.content.startswith("The answer is (")
    assert decision.content != "The answer is (B)"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        param("42", "43", id="int"),
        param("1,000", "1001", id="comma"),
        param("3.5", "4.5", id="float"),
        param("abc", "abc1", id="non_numeric"),
    ],
)  # fmt: skip
def test_gsm8k_wrong_is_a_different_number(raw: str, expected: str) -> None:
    assert bump_number(raw) == expected


def test_decision_is_deterministic_per_prompt() -> None:
    body = '{"prompt":"q","answer":"B"}'
    ds = _dataset(body, default_format=AccuracyFormat.MMLU, correct_rate=0.5)
    entry = ds.lookup("q")
    assert entry is not None
    a = ds.decide(entry)
    b = ds.decide(entry)
    assert a.content == b.content
    assert a.correct == b.correct


def test_correct_rate_is_honored_across_prompts() -> None:
    lines = [f'{{"prompt":"q{i}","answer":"B"}}' for i in range(400)]
    ds = _dataset(
        "\n".join(lines),
        default_format=AccuracyFormat.MMLU,
        correct_rate=0.25,
    )
    correct = sum(
        1
        for i in range(400)
        if ds.decide(ds.lookup(f"q{i}")).correct  # type: ignore[union-attr]
    )
    assert 70 <= correct <= 130


def test_cot_uses_reasoning_field_by_default() -> None:
    body = '{"prompt":"q","answer":"B"}'
    ds = _dataset(
        body,
        default_format=AccuracyFormat.MMLU,
        correct_rate=1.0,
        cot_rate=1.0,
        reasoning_field=True,
    )
    decision = ds.decide(ds.lookup("q"))  # type: ignore[arg-type]
    assert decision.cot
    assert decision.content == "The answer is (B)"
    assert decision.reasoning_content is not None
    assert "The answer is (B)" in decision.reasoning_content


def test_cot_inline_prefixes_answer_when_field_disabled() -> None:
    body = '{"prompt":"q","answer":"42"}'
    ds = _dataset(
        body,
        default_format=AccuracyFormat.GSM8K,
        correct_rate=1.0,
        cot_rate=1.0,
        reasoning_field=False,
    )
    decision = ds.decide(ds.lookup("q"))  # type: ignore[arg-type]
    assert decision.reasoning_content is None
    assert decision.content.endswith("#### 42")
    assert "Therefore" in decision.content


def test_exact_match_cot_never_pollutes_content_inline() -> None:
    body = '{"prompt":"q","answer":"True"}'
    ds = _dataset(
        body,
        default_format=AccuracyFormat.EXACT_MATCH,
        correct_rate=1.0,
        cot_rate=1.0,
        reasoning_field=False,
    )
    decision = ds.decide(ds.lookup("q"))  # type: ignore[arg-type]
    assert decision.content == "True"
    assert decision.reasoning_content is not None


def test_adversarial_reasoning_only_empties_content() -> None:
    found = False
    for i in range(64):
        body = f'{{"prompt":"q{i}","answer":"B"}}'
        ds = _dataset(
            body,
            default_format=AccuracyFormat.MMLU,
            correct_rate=1.0,
            adversarial_rate=1.0,
        )
        decision = ds.decide(ds.lookup(f"q{i}"))  # type: ignore[arg-type]
        if decision.adversarial == Adversarial.REASONING_ONLY:
            assert decision.content == ""
            assert decision.reasoning_content is not None
            assert "The answer is (B)" in decision.reasoning_content
            found = True
            break
    assert found


def test_adversarial_null_object_chunk_sets_flag() -> None:
    found = False
    for i in range(64):
        body = f'{{"prompt":"q{i}","answer":"B"}}'
        ds = _dataset(
            body,
            default_format=AccuracyFormat.MMLU,
            correct_rate=1.0,
            adversarial_rate=1.0,
        )
        decision = ds.decide(ds.lookup(f"q{i}"))  # type: ignore[arg-type]
        if decision.adversarial == Adversarial.NULL_OBJECT_CHUNK:
            assert decision.null_object_chunk
            found = True
            break
    assert found


def test_live_tally_counts_correct_incorrect_and_tasks() -> None:
    live = AccuracyLive()
    live.record(
        AccuracyDecision(
            content="ok",
            reasoning_content=None,
            null_object_chunk=False,
            correct=True,
            cot=False,
            adversarial=None,
        ),
        task="demo",
    )
    live.record(
        AccuracyDecision(
            content="bad",
            reasoning_content="cot",
            null_object_chunk=False,
            correct=False,
            cot=True,
            adversarial=Adversarial.WRONG_CASE,
        ),
        task="demo",
    )
    live.record(
        AccuracyDecision(
            content="ok",
            reasoning_content=None,
            null_object_chunk=False,
            correct=True,
            cot=False,
            adversarial=None,
        ),
        task="other",
    )
    live.record_unmatched()
    snap = live.snapshot()
    assert snap.matched == 3
    assert snap.correct == 2
    assert snap.incorrect == 1
    assert snap.accuracy == pytest.approx(2 / 3)
    assert snap.unmatched == 1
    assert snap.adversarial == 1
    assert snap.cot == 1
    assert snap.tasks["demo"].matched == 2
    assert snap.tasks["demo"].correct == 1
    assert snap.tasks["demo"].accuracy == pytest.approx(0.5)
    assert snap.tasks["other"].correct == 1
    assert snap.tasks["other"].accuracy == pytest.approx(1.0)


def test_live_tally_empty_snapshot_is_zeroed() -> None:
    snap = AccuracyLive().snapshot()
    assert snap.matched == 0
    assert snap.accuracy == 0.0
    assert snap.tasks == {}


def test_per_entry_format_overrides_default() -> None:
    body = '{"prompt":"q","answer":"42","format":"gsm8k"}'
    ds = _dataset(body, default_format=AccuracyFormat.MMLU, correct_rate=1.0)
    entry = ds.lookup("q")
    assert entry is not None
    assert entry.format == AccuracyFormat.GSM8K
    assert ds.decide(entry).content == "#### 42"
