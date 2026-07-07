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
    **dataset_overrides: Any,
) -> SyntheticDatasetComposer:
    """Build a SyntheticDatasetComposer whose prompts.isl/osl are the given
    distribution dicts (or fixed scalars).

    Uses the native BenchmarkConfig path (rather than CLIConfig, which is flat
    and cannot express typed distributions): the top-level ``isl``/``osl``
    shortcuts hoist into ``prompts.{isl,osl}``. A FakeTokenizer keeps prompt
    generation cheap.

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
