# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed SamplingDistribution shapes must actually sample at runtime.

Regression tests for the flattening bug where lognormal/multimodal/
empirical/percentile ISL/OSL configs silently collapsed to a constant at
the distribution mean (only Normal's stddev survived).
"""

import statistics
from typing import Any

import pytest

from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer
from tests.harness.fake_tokenizer import FakeTokenizer
from tests.unit.conftest import make_benchmark_run


def _make_composer(
    isl: Any = None,
    osl: Any = None,
    entries: int = 1000,
    sequence_distribution: Any = None,
    first_turn_isl: Any = None,
    **dataset_overrides: Any,
) -> SyntheticDatasetComposer:
    """Build a SyntheticDatasetComposer whose prompts.isl/osl are the given
    distribution dicts (or fixed scalars).

    Uses the native BenchmarkConfig path (rather than CLIConfig, which is flat
    and cannot express typed distributions): the top-level ``isl``/``osl``
    shortcuts hoist into ``prompts.{isl,osl}``. A ``sequence_distribution`` list
    and ``first_turn_isl`` are nested under ``prompts`` (they have no top-level
    shorthand). A FakeTokenizer keeps prompt generation cheap.

    The composer clears its per-turn sequence cache inside ``_finalize_turn``
    to free memory; the tests need to observe every per-turn sample, so the
    clear is disabled here. Sampling still happens exactly once per turn (the
    cache lookup guarantees it), so this does not change what is drawn.
    """
    dataset: dict[str, Any] = {
        "name": "default",
        "type": "synthetic",
        "entries": entries,
        **dataset_overrides,
    }
    if isl is not None:
        dataset["isl"] = isl
    if osl is not None:
        dataset["osl"] = osl
    if sequence_distribution is not None:
        dataset.setdefault("prompts", {})["sequence_distribution"] = (
            sequence_distribution
        )
    if first_turn_isl is not None:
        dataset.setdefault("prompts", {})["first_turn_isl"] = first_turn_isl

    run = make_benchmark_run(extra={"datasets": [dataset]})
    composer = SyntheticDatasetComposer(run=run, tokenizer=FakeTokenizer())
    composer._clear_turn_cache = lambda turn_id: None
    return composer


class TestPlainIslOslSampling:
    def test_lognormal_isl_varies_per_turn(self):
        composer = _make_composer(isl={"mean": 1000, "median": 600})
        composer.create_dataset()
        isls = [pair[0] for pair in composer._turn_sequence_cache.values()]
        assert len(set(isls)) > 10  # was: constant 1000 for every turn
        assert statistics.median(isls) < statistics.fmean(isls)  # right skew

    def test_multimodal_isl_produces_both_modes(self):
        composer = _make_composer(
            isl={
                "peaks": [
                    {"mean": 100, "stddev": 5, "weight": 50},
                    {"mean": 10000, "stddev": 50, "weight": 50},
                ]
            }
        )
        composer.create_dataset()
        isls = [pair[0] for pair in composer._turn_sequence_cache.values()]
        low = [v for v in isls if v < 1000]
        high = [v for v in isls if v > 5000]
        assert len(low) + len(high) == len(isls)
        assert 0.3 < len(low) / len(isls) < 0.7

    def test_empirical_isl_only_configured_values(self):
        composer = _make_composer(
            isl={
                "points": [{"value": 128, "weight": 50}, {"value": 4096, "weight": 50}]
            }
        )
        composer.create_dataset()
        isls = {pair[0] for pair in composer._turn_sequence_cache.values()}
        assert isls <= {128, 4096}
        assert isls == {128, 4096}

    def test_percentile_isl_hits_targets(self):
        composer = _make_composer(
            isl={"p50": 5000, "p99": 40000, "mean": 6000}, entries=4000
        )
        composer.create_dataset()
        isls = sorted(pair[0] for pair in composer._turn_sequence_cache.values())
        assert isls[len(isls) // 2] == pytest.approx(5000, rel=0.10)
        assert statistics.fmean(isls) == pytest.approx(6000, rel=0.08)


class TestNoDoubleSampling:
    def test_normal_isl_observed_stddev_matches_configured(self, monkeypatch):
        """Configured stddev 50 must yield observed ~50, not ~71 (sqrt(2)x).

        Capture the mean/stddev PromptGenerator.generate receives: variance
        must come from the per-turn sample (stddev arg == 0)."""
        captured = []
        composer = _make_composer(isl={"mean": 550, "stddev": 50}, entries=500)

        original = composer.prompt_generator.generate

        def spy(*, mean, stddev):
            captured.append((mean, stddev))
            return original(mean=mean, stddev=stddev)

        monkeypatch.setattr(composer.prompt_generator, "generate", spy)
        composer.create_dataset()
        assert all(s == 0 for _, s in captured)
        means = [m for m, _ in captured]
        assert 35 < statistics.stdev(means) < 65


class TestMaxTokensPairing:
    def test_osl_distribution_sets_varied_max_tokens(self):
        composer = _make_composer(isl=512, osl={"mean": 256, "stddev": 64})
        conversations = composer.create_dataset()
        max_tokens = [t.max_tokens for c in conversations for t in c.turns]
        assert len(set(max_tokens)) > 5
        assert statistics.fmean(max_tokens) == pytest.approx(256, rel=0.10)

    def test_osl_and_isl_come_from_same_turn_sample(self):
        """max_tokens must equal the OSL cached for that turn (single draw)."""
        composer = _make_composer(isl=512, osl={"mean": 256, "stddev": 64})
        conversations = composer.create_dataset()
        # every max_tokens value must appear as an OSL in the cache
        cached_osls = sorted(p[1] for p in composer._turn_sequence_cache.values())
        turn_max = sorted(t.max_tokens for c in conversations for t in c.turns)
        assert turn_max == cached_osls

    def test_osl_zero_mean_disables_max_tokens(self):
        composer = _make_composer(isl=512, osl={"mean": 0})
        conversations = composer.create_dataset()
        assert all(t.max_tokens is None for c in conversations for t in c.turns)


class TestTurnsAndDelayWiring:
    def test_lognormal_turns_varies(self):
        composer = _make_composer(isl=128, turns={"mean": 6, "median": 4}, entries=300)
        conversations = composer.create_dataset()
        counts = [len(c.turns) for c in conversations]
        assert len(set(counts)) > 3  # was: constant 6
        assert statistics.median(counts) <= statistics.fmean(counts)  # right skew

    def test_empirical_turn_delay_only_configured_values(self):
        composer = _make_composer(
            isl=128,
            turns=3,
            turn_delay={
                "points": [{"value": 100, "weight": 50}, {"value": 5000, "weight": 50}]
            },
            entries=100,
        )
        conversations = composer.create_dataset()
        delays = {
            t.delay for c in conversations for t in c.turns if t.delay is not None
        }
        assert delays <= {100, 5000, 100.0, 5000.0}
        assert len(delays) == 2  # was: constant 2550 (empirical mean)


class TestTypedSequenceDistributionBuckets:
    def test_lognormal_bucket_actually_skews(self):
        composer = _make_composer(
            sequence_distribution=[
                {"isl": {"mean": 2000, "median": 1000}, "osl": 100, "probability": 100},
            ]
        )
        composer.create_dataset()
        isls = [p[0] for p in composer._turn_sequence_cache.values()]
        assert len(set(isls)) > 10  # was: constant 2000 (lognormal flattened)
        assert statistics.median(isls) < statistics.fmean(isls)

    def test_bucket_pairing_never_crosses(self):
        composer = _make_composer(
            sequence_distribution=[
                {
                    "isl": {"mean": 100, "stddev": 5},
                    "osl": {"mean": 10, "stddev": 1},
                    "probability": 50,
                },
                {
                    "isl": {"mean": 10000, "stddev": 50},
                    "osl": {"mean": 1000, "stddev": 10},
                    "probability": 50,
                },
            ]
        )
        composer.create_dataset()
        for isl, osl in composer._turn_sequence_cache.values():
            assert (isl < 1000) == (osl < 100)  # small isl with small osl only

    def test_bucket_weights_respected(self):
        composer = _make_composer(
            entries=2000,
            sequence_distribution=[
                {"isl": 100, "osl": 10, "probability": 80},
                {"isl": 10000, "osl": 1000, "probability": 20},
            ],
        )
        composer.create_dataset()
        isls = [p[0] for p in composer._turn_sequence_cache.values()]
        small_frac = sum(1 for v in isls if v == 100) / len(isls)
        assert 0.72 < small_frac < 0.88

    def test_percentile_inside_bucket(self):
        composer = _make_composer(
            entries=4000,
            sequence_distribution=[
                {
                    "isl": {"p50": 5000, "p99": 40000, "mean": 6000},
                    "osl": 100,
                    "probability": 100,
                },
            ],
        )
        composer.create_dataset()
        isls = sorted(p[0] for p in composer._turn_sequence_cache.values())
        assert isls[len(isls) // 2] == pytest.approx(5000, rel=0.10)


class TestFirstTurnIsl:
    def test_first_turn_uses_starting_distribution_subsequent_use_isl(self):
        composer = _make_composer(
            isl={"mean": 200, "stddev": 5},
            first_turn_isl={"mean": 20000, "stddev": 10},
            turns=4,
            entries=50,
        )
        conversations = composer.create_dataset()
        cache = composer._turn_sequence_cache
        for conv in conversations:
            assert cache[id(conv.turns[0])][0] > 10000  # starting context size
            for turn in conv.turns[1:]:
                assert cache[id(turn)][0] < 1000  # per-turn new input

    def test_first_turn_isl_unset_isl_applies_to_all_turns(self):
        composer = _make_composer(isl={"mean": 200, "stddev": 5}, turns=3, entries=30)
        conversations = composer.create_dataset()
        cache = composer._turn_sequence_cache
        for conv in conversations:
            for turn in conv.turns:
                assert cache[id(turn)][0] < 1000


class TestRelativeBucketWeights:
    def test_weights_not_summing_to_100_normalized(self):
        composer = _make_composer(
            entries=2000,
            sequence_distribution=[
                {"isl": 100, "osl": 10, "probability": 50},
                {"isl": 10000, "osl": 1000, "probability": 1},
            ],
        )
        composer.create_dataset()
        isls = [p[0] for p in composer._turn_sequence_cache.values()]
        small_frac = sum(1 for v in isls if v == 100) / len(isls)
        assert small_frac > 0.94  # 50:1 ~ 98%


class TestZeroWeightBuckets:
    def test_zero_weight_bucket_validates_and_never_samples(self):
        """A probability: 0 bucket is valid config and is never sampled."""
        from aiperf.common import random_generator as rng
        from aiperf.config.types import SequenceDistributionEntry
        from aiperf.dataset.composer.base import _TypedSequenceDistribution

        entries = [
            SequenceDistributionEntry.model_validate(
                {"isl": 100, "osl": 10, "probability": 0}
            ),
            SequenceDistributionEntry.model_validate(
                {"isl": 5000, "osl": 500, "probability": 100}
            ),
        ]
        dist = _TypedSequenceDistribution(entries, rng.derive("test.zero.bucket"))
        draws = [dist.sample_lengths(dist.sample_bucket()) for _ in range(500)]
        assert all(isl == 5000 and osl == 500 for isl, osl in draws)

    def test_zero_weight_bucket_end_to_end_via_composer(self):
        composer = _make_composer(
            entries=500,
            sequence_distribution=[
                {"isl": 100, "osl": 10, "probability": 0},
                {"isl": 5000, "osl": 500, "probability": 100},
            ],
        )
        composer.create_dataset()
        isls = {p[0] for p in composer._turn_sequence_cache.values()}
        assert isls == {5000}  # zero-weight bucket never contributes

    def test_all_zero_list_rejected_by_validator(self):
        from aiperf.config.types import (
            SequenceDistributionEntry,
            validate_probability_distribution,
        )

        entries = [
            SequenceDistributionEntry.model_validate(
                {"isl": 100, "osl": 10, "probability": 0}
            ),
            SequenceDistributionEntry.model_validate(
                {"isl": 5000, "osl": 500, "probability": 0}
            ),
        ]
        with pytest.raises(ValueError, match="positive total"):
            validate_probability_distribution(entries)

    def test_all_zero_list_rejected_by_typed_distribution(self):
        from aiperf.common import random_generator as rng
        from aiperf.config.types import SequenceDistributionEntry
        from aiperf.dataset.composer.base import _TypedSequenceDistribution

        entries = [
            SequenceDistributionEntry.model_validate(
                {"isl": 100, "osl": 10, "probability": 0}
            ),
        ]
        with pytest.raises(ValueError, match="positive-weight entry"):
            _TypedSequenceDistribution(entries, rng.derive("test.all.zero"))
