# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy-dataset response mode for the AIPerf mock server.

AIPerf does not send ground truth to inference servers, so the mock loads a
JSONL ``{prompt, ground_truth}`` dataset and matches requests by prompt. Seeded
decisions produce grader-compatible correct, incorrect, chain-of-thought, and
adversarial responses. Seeding uses ``random.Random`` derived from
``(random_seed, key_norm)``.
"""

from __future__ import annotations

import hashlib
import random
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import orjson


class AccuracyFormat(StrEnum):
    """Grader answer wrap selected globally or per dataset row."""

    PASSTHROUGH = "passthrough"
    MMLU = "mmlu"
    MMLU_PRO = "mmlu_pro"
    GSM8K = "gsm8k"
    MATH = "math"
    EXACT_MATCH = "exact_match"

    def tolerates_inline_prefix(self) -> bool:
        """Whether inline CoT before the answer is safe for the grader."""
        return self in {
            AccuracyFormat.MMLU,
            AccuracyFormat.MMLU_PRO,
            AccuracyFormat.GSM8K,
            AccuracyFormat.MATH,
        }


class AccuracyMatch(StrEnum):
    """How request user text is matched to a dataset row."""

    EXACT = "exact"
    EXACT_CI = "exact_ci"
    SUBSTRING = "substring"
    SUBSTRING_CI = "substring_ci"

    def case_insensitive(self) -> bool:
        return self in {AccuracyMatch.EXACT_CI, AccuracyMatch.SUBSTRING_CI}

    def substring(self) -> bool:
        return self in {AccuracyMatch.SUBSTRING, AccuracyMatch.SUBSTRING_CI}


class Adversarial(StrEnum):
    """Parser-choke shapes used to stress-test client robustness."""

    LEADING_WHITESPACE = "leading_whitespace"
    TRAILING_PROSE = "trailing_prose"
    WRONG_CASE = "wrong_case"
    REASONING_ONLY = "reasoning_only"
    BOXED_WRAP = "boxed_wrap"
    MULTIPLE_CONFLICTING = "multiple_conflicting"
    UNICODE = "unicode"
    NULL_OBJECT_CHUNK = "null_object_chunk"


# Fixed catalog order — do not reorder; it defines the seeded choice stream.
_ADVERSARIAL_ALL: tuple[Adversarial, ...] = (
    Adversarial.LEADING_WHITESPACE,
    Adversarial.TRAILING_PROSE,
    Adversarial.WRONG_CASE,
    Adversarial.REASONING_ONLY,
    Adversarial.BOXED_WRAP,
    Adversarial.MULTIPLE_CONFLICTING,
    Adversarial.UNICODE,
    Adversarial.NULL_OBJECT_CHUNK,
)

_COT_OPENERS: tuple[str, ...] = (
    "Let me work through this step by step.",
    "First, let me analyze the problem carefully.",
    "Breaking this down into parts.",
    "I'll reason about each option in turn.",
)

_COT_MIDDLES: tuple[str, ...] = (
    "Considering the constraints, one path stands out.",
    "Eliminating the implausible cases narrows it down.",
    "The key relationship makes the conclusion clear.",
    "Cross-checking against the given facts confirms the direction.",
)


@dataclass(frozen=True, slots=True)
class AccuracySettings:
    """Knobs used when loading / deciding accuracy responses."""

    match_mode: AccuracyMatch = AccuracyMatch.SUBSTRING
    default_format: AccuracyFormat = AccuracyFormat.PASSTHROUGH
    correct_rate: float = 1.0
    cot_rate: float = 0.0
    adversarial_rate: float = 0.0
    reasoning_field: bool = True
    random_seed: int = 0


@dataclass(slots=True)
class Entry:
    """A normalized dataset row and its grader metadata."""

    key_norm: str
    """Stable normalized identity used for matching and seeded verdicts."""

    gold: str
    """Clean gold answer (not trimmed at load time)."""

    task: str | None = None
    """Reporting task/subject, if present."""

    format: AccuracyFormat | None = None
    """Per-row format override (falls back to the dataset default)."""

    choices: list[str] = field(default_factory=list)
    """Multiple-choice option letters for wrong-answer selection."""


@dataclass(slots=True)
class AccuracyDecision:
    """The rendered response for one matched request."""

    content: str
    reasoning_content: str | None
    null_object_chunk: bool
    correct: bool
    cot: bool
    adversarial: Adversarial | None


@dataclass(slots=True)
class TaskAccuracy:
    """Per-task rollup for the live oracle."""

    matched: int
    correct: int
    accuracy: float


@dataclass(slots=True)
class AccuracyLiveSnapshot:
    """Point-in-time copy of the live tally."""

    matched: int
    correct: int
    incorrect: int
    accuracy: float
    unmatched: int
    adversarial: int
    cot: int
    tasks: dict[str, TaskAccuracy]


def normalize(s: str) -> str:
    """Collapse whitespace runs and trim ends."""
    return " ".join(s.split())


def norm_key(s: str, ci: bool) -> str:
    """Apply the same normalization rules to dataset and request keys."""
    n = normalize(s)
    return n.lower() if ci else n


def parse_format(s: str) -> AccuracyFormat | None:
    """Parse a format / benchmark alias string into ``AccuracyFormat``."""
    key = s.lower().replace("-", "_")
    mapping: dict[str, AccuracyFormat] = {
        "passthrough": AccuracyFormat.PASSTHROUGH,
        "mmlu": AccuracyFormat.MMLU,
        "mmlu_pro": AccuracyFormat.MMLU_PRO,
        "gsm8k": AccuracyFormat.GSM8K,
        "math": AccuracyFormat.MATH,
        "aime": AccuracyFormat.MATH,
        "exact_match": AccuracyFormat.EXACT_MATCH,
        "exact": AccuracyFormat.EXACT_MATCH,
        "hellaswag": AccuracyFormat.EXACT_MATCH,
        "bigbench": AccuracyFormat.EXACT_MATCH,
    }
    return mapping.get(key)


def format_correct(fmt: AccuracyFormat, gold: str) -> str:
    """Emit a grader-compatible correct answer for ``fmt``."""
    g = gold.strip()
    if fmt in {AccuracyFormat.PASSTHROUGH, AccuracyFormat.EXACT_MATCH}:
        return g
    if fmt in {AccuracyFormat.MMLU, AccuracyFormat.MMLU_PRO}:
        return f"The answer is ({g})"
    if fmt is AccuracyFormat.GSM8K:
        return f"#### {g}"
    return rf"\boxed{{{g}}}"


def bump_number(s: str) -> str:
    """Produce a plausible wrong numeric answer for gsm8k/math."""
    cleaned = s.replace(",", "")
    try:
        return str(int(cleaned) + 1)
    except ValueError:
        pass
    try:
        return str(float(cleaned) + 1.0)
    except ValueError:
        return f"{s}1"


def _value_to_string(v: Any) -> str | None:
    if isinstance(v, str):
        return v
    # bool is a subclass of int — handle before numeric branches.
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return str(v)
    return None


def _field(obj: dict[str, Any], aliases: tuple[str, ...]) -> Any | None:
    for key in aliases:
        if key in obj:
            return obj[key]
    return None


def _derive_seed(random_seed: int, key_norm: str) -> int:
    """Stable seed from ``(random_seed, key_norm)``."""
    digest = hashlib.blake2b(
        random_seed.to_bytes(8, "little", signed=False)
        + key_norm.encode()
        + b"mock.accuracy",
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big", signed=False)


class AccuracyDataset:
    """Parsed JSONL dataset plus seeded-decision knobs."""

    def __init__(
        self,
        exact: dict[str, Entry],
        entries: list[Entry],
        settings: AccuracySettings,
    ) -> None:
        self._exact = exact
        self._entries = entries
        self._match_mode = settings.match_mode
        self._default_format = settings.default_format
        self._correct_rate = max(0.0, min(1.0, settings.correct_rate))
        self._cot_rate = max(0.0, min(1.0, settings.cot_rate))
        self._adversarial_rate = max(0.0, min(1.0, settings.adversarial_rate))
        self._reasoning_field = settings.reasoning_field
        self._seed = settings.random_seed

    @classmethod
    def load(cls, path: Path | str, settings: AccuracySettings) -> AccuracyDataset:
        """Load from a JSONL file path."""
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError(f"accuracy dataset {p}: {e}") from e
        return cls.from_jsonl(text, settings)

    @classmethod
    def from_jsonl(cls, body: str, settings: AccuracySettings) -> AccuracyDataset:
        """Parse a JSONL body into an ``AccuracyDataset``."""
        ci = settings.match_mode.case_insensitive()
        exact: dict[str, Entry] = {}
        for lineno, raw_line in enumerate(body.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = orjson.loads(line)
            except orjson.JSONDecodeError as e:
                raise ValueError(f"accuracy dataset line {lineno}: {e}") from e
            if not isinstance(value, dict):
                continue
            prompt = _value_to_string(
                _field(value, ("prompt", "question", "input", "text"))
            )
            gold = _value_to_string(
                _field(value, ("ground_truth", "answer", "gold", "target"))
            )
            if prompt is None or gold is None:
                continue
            task = _value_to_string(_field(value, ("task", "subject", "category")))
            fmt_raw = value.get("format")
            if fmt_raw is None:
                fmt_raw = value.get("benchmark")
            fmt = parse_format(fmt_raw) if isinstance(fmt_raw, str) else None
            choices_raw = value.get("choices")
            choices: list[str] = []
            if isinstance(choices_raw, list):
                for item in choices_raw:
                    s = _value_to_string(item)
                    if s is not None:
                        choices.append(s)
            match_base = _value_to_string(
                _field(value, ("match_key", "match", "key", "id"))
            )
            if match_base is None:
                match_base = prompt
            key = norm_key(match_base, ci)
            exact[key] = Entry(
                key_norm=key,
                gold=gold,
                task=task,
                format=fmt,
                choices=choices,
            )
        if not exact:
            raise ValueError(
                "accuracy dataset has no usable rows (need a prompt field "
                "[prompt/question/input/text] and a gold field "
                "[ground_truth/answer/gold/target])"
            )
        entries = sorted(exact.values(), key=lambda e: len(e.key_norm), reverse=True)
        return cls(exact, entries, settings)

    def len(self) -> int:
        return len(self._exact)

    def __len__(self) -> int:
        return len(self._exact)

    def is_empty(self) -> bool:
        return not self._exact

    def lookup(self, request_text: str) -> Entry | None:
        """Find the entry matching the request's user text."""
        nk = norm_key(request_text, self._match_mode.case_insensitive())
        hit = self._exact.get(nk)
        if hit is not None:
            return hit
        if self._match_mode.substring():
            for entry in self._entries:
                if entry.key_norm and entry.key_norm in nk:
                    return entry
        return None

    def decide(self, entry: Entry) -> AccuracyDecision:
        """Render the response for a matched request.

        Deterministic in ``(random_seed, entry.key_norm)`` — independent of
        arrival order and prompt wrapping.
        """
        rng = random.Random(_derive_seed(self._seed, entry.key_norm))
        # Fixed draw order — do NOT reorder; it defines the seeded stream.
        correct = rng.random() < self._correct_rate
        cot = rng.random() < self._cot_rate
        adversarial_on = rng.random() < self._adversarial_rate

        fmt = entry.format if entry.format is not None else self._default_format
        answer = (
            format_correct(fmt, entry.gold)
            if correct
            else _format_wrong(fmt, entry, rng)
        )
        reasoning_prose = _generate_cot(rng, answer) if cot else None

        if reasoning_prose is None:
            content = answer
            reasoning_content: str | None = None
        elif self._reasoning_field:
            content = answer
            reasoning_content = reasoning_prose
        elif fmt.tolerates_inline_prefix():
            content = f"{reasoning_prose}\n{answer}"
            reasoning_content = None
        else:
            content = answer
            reasoning_content = reasoning_prose

        null_object_chunk = False
        adversarial: Adversarial | None = None
        if adversarial_on:
            variant = rng.choice(_ADVERSARIAL_ALL)
            content, reasoning_content, null_object_chunk = _apply_adversarial(
                variant,
                answer,
                reasoning_prose,
                content,
                reasoning_content,
            )
            adversarial = variant

        return AccuracyDecision(
            content=content,
            reasoning_content=reasoning_content,
            null_object_chunk=null_object_chunk,
            correct=correct,
            cot=cot,
            adversarial=adversarial,
        )


def _format_wrong(fmt: AccuracyFormat, entry: Entry, rng: random.Random) -> str:
    g = entry.gold.strip()
    if fmt in {AccuracyFormat.MMLU, AccuracyFormat.MMLU_PRO}:
        return f"The answer is ({_wrong_letter(fmt, entry, rng)})"
    if fmt is AccuracyFormat.GSM8K:
        return f"#### {bump_number(g)}"
    if fmt is AccuracyFormat.MATH:
        return rf"\boxed{{{bump_number(g)}}}"
    return f"{g}_wrong"


def _wrong_letter(fmt: AccuracyFormat, entry: Entry, rng: random.Random) -> str:
    gold = entry.gold.strip().upper()
    if entry.choices:
        pool = [c.strip() for c in entry.choices]
    elif fmt is AccuracyFormat.MMLU_PRO:
        pool = list("ABCDEFGHIJ")
    else:
        pool = list("ABCD")
    alternatives = [c for c in pool if c.upper() != gold]
    if not alternatives:
        first = gold[0] if gold else "Z"
        return chr(ord(first) + 1) if first.isalpha() else "Z"
    return rng.choice(alternatives)


def _generate_cot(rng: random.Random, answer: str) -> str:
    opener = rng.choice(_COT_OPENERS)
    middle = rng.choice(_COT_MIDDLES)
    return f"{opener} {middle} Therefore, {answer}"


def _apply_adversarial(
    variant: Adversarial,
    answer: str,
    reasoning_prose: str | None,
    content: str,
    reasoning_content: str | None,
) -> tuple[str, str | None, bool]:
    null_object_chunk = False
    if variant is Adversarial.LEADING_WHITESPACE:
        content = f"\n\n{content}"
    elif variant is Adversarial.TRAILING_PROSE:
        content = f"{content}\n\nActually, on reflection I am not fully certain."
    elif variant is Adversarial.WRONG_CASE:
        content = content.lower()
    elif variant is Adversarial.REASONING_ONLY:
        prose = reasoning_prose or ""
        reasoning_content = answer if not prose else f"{prose}\n{answer}"
        content = ""
    elif variant is Adversarial.BOXED_WRAP:
        content = rf"\boxed{{{content}}}"
    elif variant is Adversarial.MULTIPLE_CONFLICTING:
        content = f"The answer is (Z). Wait, reconsidering — {content}"
    elif variant is Adversarial.UNICODE:
        content = f"{content} ✓🎯—naïve"
    elif variant is Adversarial.NULL_OBJECT_CHUNK:
        null_object_chunk = True
    return content, reasoning_content, null_object_chunk


def _ratio(correct: int, matched: int) -> float:
    return 0.0 if matched == 0 else correct / matched


class AccuracyLive:
    """Live tally of served responses for comparison with reported accuracy."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._matched = 0
        self._correct = 0
        self._adversarial = 0
        self._cot = 0
        self._unmatched = 0
        self._per_task: dict[str, list[int]] = {}

    def record(self, decision: AccuracyDecision, task: str | None) -> None:
        """Record one served, prompt-matched response."""
        with self._lock:
            self._matched += 1
            if decision.correct:
                self._correct += 1
            if decision.adversarial is not None:
                self._adversarial += 1
            if decision.cot:
                self._cot += 1
            key = task if task is not None else "unknown"
            counts = self._per_task.setdefault(key, [0, 0])
            counts[0] += 1
            if decision.correct:
                counts[1] += 1

    def record_unmatched(self) -> None:
        """Record one accuracy-enabled request whose prompt matched no row."""
        with self._lock:
            self._unmatched += 1

    def snapshot(self) -> AccuracyLiveSnapshot:
        """Take a consistent snapshot of the current tally."""
        with self._lock:
            matched = self._matched
            correct = self._correct
            tasks = {
                k: TaskAccuracy(
                    matched=v[0],
                    correct=v[1],
                    accuracy=_ratio(v[1], v[0]),
                )
                for k, v in sorted(self._per_task.items())
            }
            return AccuracyLiveSnapshot(
                matched=matched,
                correct=correct,
                incorrect=max(0, matched - correct),
                accuracy=_ratio(correct, matched),
                unmatched=self._unmatched,
                adversarial=self._adversarial,
                cot=self._cot,
                tasks=tasks,
            )

    def reset(self) -> None:
        """Clear all counters (used on lifespan startup)."""
        with self._lock:
            self._matched = 0
            self._correct = 0
            self._adversarial = 0
            self._cot = 0
            self._unmatched = 0
            self._per_task.clear()


# Process-local holders wired by app lifespan / make_ctx.
_accuracy_dataset: AccuracyDataset | None = None
_accuracy_live: AccuracyLive = AccuracyLive()
_accuracy_settings: AccuracySettings | None = None


def get_accuracy_dataset() -> AccuracyDataset | None:
    """Return the loaded dataset, or ``None`` when accuracy mode is off."""
    return _accuracy_dataset


def get_accuracy_live() -> AccuracyLive:
    """Return the process-local live tally."""
    return _accuracy_live


def get_accuracy_settings() -> AccuracySettings | None:
    """Return the settings used to load the dataset, if any."""
    return _accuracy_settings


def set_accuracy_state(
    dataset: AccuracyDataset | None,
    settings: AccuracySettings | None = None,
) -> None:
    """Install (or clear) the process-local accuracy dataset and reset live tallies."""
    global _accuracy_dataset, _accuracy_settings
    _accuracy_dataset = dataset
    _accuracy_settings = settings
    _accuracy_live.reset()


def settings_from_config(config: Any) -> AccuracySettings:
    """Build ``AccuracySettings`` from a ``MockServerConfig``-like object."""
    seed = config.random_seed
    return AccuracySettings(
        match_mode=AccuracyMatch(config.accuracy_match),
        default_format=AccuracyFormat(config.accuracy_format),
        correct_rate=float(config.accuracy_correct_rate),
        cot_rate=float(config.accuracy_cot_rate),
        adversarial_rate=float(config.accuracy_adversarial_rate),
        reasoning_field=bool(config.accuracy_reasoning_field),
        random_seed=int(seed) if seed is not None else 0,
    )
