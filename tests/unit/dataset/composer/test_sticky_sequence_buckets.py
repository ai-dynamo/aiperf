# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sticky sequence_distribution buckets: each conversation draws ONE bucket
at creation and keeps it for every turn; per-bucket first_turn_isl sizes the
seed context while isl sizes subsequent turns."""

import bisect

from aiperf.common import random_generator as rng
from aiperf.config.dataset.content import PromptConfig
from tests.unit.dataset.composer.test_distribution_wiring import _make_composer

# Disjoint ranges so a single sample proves bucket membership.
SMALL_BUCKET = {
    "isl": {"mean": 150, "stddev": 10},
    "osl": {"mean": 50, "stddev": 5},
    "probability": 50,
}
LARGE_BUCKET = {
    "isl": {"mean": 150000, "stddev": 1000},
    "osl": {"mean": 5000, "stddev": 100},
    "probability": 50,
}


class TestStickyBuckets:
    def test_conversations_never_mix_buckets(self):
        composer = _make_composer(
            sequence_distribution=[SMALL_BUCKET, LARGE_BUCKET],
            entries=300,
            turns=4,
        )
        conversations = composer.create_dataset()
        small = large = 0
        for conv in conversations:
            pairs = [composer._turn_sequence_cache[id(t)] for t in conv.turns]
            if pairs[0][0] < 1000:
                small += 1
                assert all(isl < 1000 and osl < 1000 for isl, osl in pairs)
            else:
                large += 1
                assert all(isl > 100000 and osl > 1000 for isl, osl in pairs)
        # Both classes must actually appear (weights are 50/50).
        assert small > 60 and large > 60

    def test_single_turn_workloads_still_sample_both_buckets(self):
        composer = _make_composer(
            sequence_distribution=[SMALL_BUCKET, LARGE_BUCKET], entries=400
        )
        composer.create_dataset()
        isls = [p[0] for p in composer._turn_sequence_cache.values()]
        low = sum(1 for v in isls if v < 1000)
        assert 0.3 < low / len(isls) < 0.7


class TestFirstTurnIslRouting:
    def test_first_turn_uses_seed_context_later_turns_use_isl(self):
        composer = _make_composer(
            sequence_distribution=[
                {
                    "first_turn_isl": {"mean": 20000, "stddev": 500},
                    "isl": {"mean": 300, "stddev": 50},
                    "osl": 128,
                    "probability": 100,
                }
            ],
            entries=100,
            turns=3,
        )
        conversations = composer.create_dataset()
        for conv in conversations:
            isls = [composer._turn_sequence_cache[id(t)][0] for t in conv.turns]
            assert isls[0] > 10000  # seed context
            assert all(v < 5000 for v in isls[1:])  # per-turn growth

    def test_bucket_without_first_turn_isl_falls_back_to_isl(self):
        composer = _make_composer(
            sequence_distribution=[
                {"isl": {"mean": 300, "stddev": 50}, "osl": 128, "probability": 100}
            ],
            entries=100,
            turns=3,
        )
        conversations = composer.create_dataset()
        for conv in conversations:
            isls = [composer._turn_sequence_cache[id(t)][0] for t in conv.turns]
            assert all(v < 5000 for v in isls)


class TestSingleTurnDrawStreamCompat:
    def test_single_turn_draws_match_legacy_ordering(self):
        """Per single-turn conversation the RNG stream must see exactly the
        legacy draw order: bucket pick (one random()), then ISL, then OSL.
        Replays the pre-sticky algorithm on an identically-derived stream.
        """
        bucket_dicts = [SMALL_BUCKET, LARGE_BUCKET]
        # Sanity: derive() must be reproducible for the replay to be a valid oracle.
        assert (
            rng.derive("compat.check").random() == rng.derive("compat.check").random()
        )

        composer = _make_composer(sequence_distribution=bucket_dicts, entries=50)
        composer.create_dataset()
        actual = list(composer._turn_sequence_cache.values())

        parsed = PromptConfig.model_validate(
            {"sequence_distribution": bucket_dicts}
        ).sequence_distribution
        replay_rng = rng.derive("composer.sequence.distribution")
        total = sum(e.probability for e in parsed)
        cumulative: list[float] = []
        acc = 0.0
        for e in parsed:
            acc += e.probability / total
            cumulative.append(acc)
        expected = []
        for _ in range(50):
            r = replay_rng.random()
            idx = min(bisect.bisect_right(cumulative, r), len(parsed) - 1)
            entry = parsed[idx]
            expected.append(
                (entry.isl.sample_int(replay_rng), entry.osl.sample_int(replay_rng))
            )
        assert actual == expected
