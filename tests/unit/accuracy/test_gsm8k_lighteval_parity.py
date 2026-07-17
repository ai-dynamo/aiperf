# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity tests: our GSM8K port vs the real lighteval implementation.

``LightevalGSM8KGrader`` is a hand port of lighteval's
``quasi_exact_match_gsm8k`` (``gsm8k_normalizer`` gold extraction +
``ExactMatches`` full-string comparison). These tests lock that port to
the *real* lighteval so it can never silently drift:

- the gold ``gsm8k_normalizer`` must be byte-for-byte identical, and
- in the regime our port claims parity for — a prediction containing the
  ``#### <number>`` marker — our correct/incorrect decision must match
  lighteval's ``ExactMatches.compute_one_item`` exactly.

We also pin the one *intentional* divergence: a chat-style prediction
with no ``####`` marker (lighteval scores 0; we rescue the last number).

Unlike the rest of the accuracy unit tests, this file uses the real
lighteval dependency on purpose — a fake harness cannot serve as a
reference oracle. It is skipped when lighteval (the ``[accuracy]``
extra) is not installed.

Reference:
    lighteval.metrics.normalizations.gsm8k_normalizer
    lighteval.metrics.metrics_sample.ExactMatches.compute_one_item
"""

from __future__ import annotations

import pytest
from pytest import param

# This file is a parity oracle against the real dependency; skip cleanly
# when lighteval isn't installed rather than faking it.
pytest.importorskip("lighteval")

from lighteval.metrics.metrics_sample import ExactMatches  # noqa: E402
from lighteval.metrics.normalizations import (  # noqa: E402
    gsm8k_normalizer as lighteval_gsm8k_normalizer,
)

from aiperf.accuracy.graders.gsm8k_grader import (  # noqa: E402
    _extract_prediction,
    _numbers_match,
    gsm8k_normalizer,
)

# Exactly the metric the recipe builds: quasi_exact_match_gsm8k.
_LIGHTEVAL_QEM = ExactMatches(
    strip_strings=True,
    normalize_pred=lighteval_gsm8k_normalizer,
    normalize_gold=lighteval_gsm8k_normalizer,
)


def _our_decision(gold_raw: str, pred_raw: str) -> bool:
    """Reproduce ``LightevalGSM8KGrader.grade``'s correctness decision."""
    gold = gsm8k_normalizer(gold_raw.strip())
    pred, _ = _extract_prediction(pred_raw.strip())
    return pred != "[invalid]" and gold != "[invalid]" and _numbers_match(gold, pred)


class TestGoldNormalizerParity:
    """Our ``gsm8k_normalizer`` must match lighteval's byte-for-byte."""

    @pytest.mark.parametrize(
        "text",
        [
            param("Natalia sold clips.\n#### 24", id="simple"),
            param("#### -5", id="negative"),
            param("#### 1,234", id="thousands-comma"),
            param("#### 1,000,000", id="millions"),
            param("#### 18.5", id="decimal"),
            param("long text ending #### 0", id="zero"),
            param("no marker present", id="no-marker"),
            param("", id="empty"),
            param("#### 24\nbut also #### 99", id="first-marker-wins"),
        ],
    )  # fmt: skip
    def test_matches_lighteval(self, text: str) -> None:
        assert gsm8k_normalizer(text) == lighteval_gsm8k_normalizer(text)


class TestMarkerRegimeFullParity:
    """When the prediction carries ``####``, our extraction equals
    lighteval's, so the whole decision must match
    ``ExactMatches.compute_one_item``."""

    @pytest.mark.parametrize(
        "gold,pred",
        [
            param("gold\n#### 24", "work\n#### 24", id="correct"),
            param("gold\n#### 24", "work\n#### 25", id="wrong"),
            param("gold\n#### 1,024", "steps\n#### 1024", id="comma-vs-bare"),
            param("gold\n#### -7", "steps\n#### -7", id="negative"),
            param("gold\n#### 18", "steps\n#### 0", id="off-by-a-lot"),
        ],
    )  # fmt: skip
    def test_decision_matches_lighteval(self, gold: str, pred: str) -> None:
        ours = _our_decision(gold, pred)
        theirs = bool(_LIGHTEVAL_QEM.compute_one_item(gold=gold, pred=pred))
        assert ours == theirs


class TestIntentionalDivergence:
    """Chat predictions without ``####``: lighteval scores 0, we rescue
    the last number. This divergence is deliberate and pinned here."""

    @pytest.mark.parametrize(
        "pred",
        [
            param("The answer is 24.", id="trailing-sentence"),
            param("... so 48/2 = 24", id="last-number"),
            param("I get 24.0", id="decimal-equiv"),
        ],
    )  # fmt: skip
    def test_we_rescue_chat_answers_lighteval_rejects(self, pred: str) -> None:
        gold = "gold\n#### 24"
        assert _our_decision(gold, pred) is True
        assert bool(_LIGHTEVAL_QEM.compute_one_item(gold=gold, pred=pred)) is False
