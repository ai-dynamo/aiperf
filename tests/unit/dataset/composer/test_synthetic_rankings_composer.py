# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.models import Conversation, Turn
from aiperf.config import AIPerfConfig
from aiperf.dataset.composer.synthetic_rankings import SyntheticRankingsDatasetComposer
from tests.unit.dataset.composer.conftest import _make_run

_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
)


def _rankings_config(**rankings_overrides) -> AIPerfConfig:
    """Build an AIPerfConfig with a synthetic dataset that includes rankings config."""
    dataset = {
        "type": "synthetic",
        "entries": 5,
        "prompts": {"isl": {"mean": 10, "stddev": 2}, "osl": 64},
        "rankings": {
            "passages": {"mean": 10, "stddev": 0},
            "passage_tokens": {"mean": 128, "stddev": 0},
            "query_tokens": {"mean": 32, "stddev": 0},
        },
    }
    if rankings_overrides:
        dataset["rankings"].update(rankings_overrides)
    return AIPerfConfig(**_BASE, datasets={"default": dataset})


def test_initialization_basic(mock_tokenizer):
    """Ensure SyntheticRankingsDatasetComposer initializes correctly."""
    run = _make_run(_rankings_config())
    composer = SyntheticRankingsDatasetComposer(run, mock_tokenizer)
    assert composer.session_id_generator is not None


def test_create_dataset_structure(mock_tokenizer):
    """Test structure and content of generated synthetic ranking dataset."""
    config = _rankings_config(passages={"mean": 5, "stddev": 1})
    run = _make_run(config)
    composer = SyntheticRankingsDatasetComposer(run, mock_tokenizer)

    dataset = composer.create_dataset()
    assert len(dataset) == config.get_default_dataset().entries

    for conv in dataset:
        assert isinstance(conv, Conversation)
        assert len(conv.turns) == 1
        turn = conv.turns[0]
        assert isinstance(turn, Turn)

        assert len(turn.texts) == 2  # query + passages
        query, passages = turn.texts
        assert query.name == "query"
        assert passages.name == "passages"
        assert len(query.contents) == 1
        assert len(passages.contents) >= 1
        assert all(isinstance(x, str) for x in passages.contents)


def test_passage_count_distribution(mock_tokenizer):
    """Test passages are generated following mean/stddev distribution."""
    run = _make_run(_rankings_config(passages={"mean": 5, "stddev": 2}))
    composer = SyntheticRankingsDatasetComposer(run, mock_tokenizer)

    dataset = composer.create_dataset()
    passage_counts = [len(conv.turns[0].texts[1].contents) for conv in dataset]

    assert all(1 <= c <= 10 for c in passage_counts)
    assert len(set(passage_counts)) > 1  # variation expected


def test_reproducibility_fixed_seed(mock_tokenizer):
    """Dataset generation should be deterministic given a fixed random seed."""
    run = _make_run(_rankings_config(passages={"mean": 4, "stddev": 1}))

    composer1 = SyntheticRankingsDatasetComposer(run, mock_tokenizer)
    data1 = composer1.create_dataset()

    composer2 = SyntheticRankingsDatasetComposer(run, mock_tokenizer)
    data2 = composer2.create_dataset()

    # Session IDs differ (fresh), but text contents should match
    for c1, c2 in zip(data1, data2, strict=True):
        t1, t2 = c1.turns[0], c2.turns[0]
        assert t1.texts[0].contents == t2.texts[0].contents
        assert t1.texts[1].contents == t2.texts[1].contents


def test_rankings_specific_token_options(mock_tokenizer):
    """Test that rankings-specific token options are used for query and passages."""
    run = _make_run(
        _rankings_config(
            passages={"mean": 3, "stddev": 0},
            passage_tokens={"mean": 100, "stddev": 10},
            query_tokens={"mean": 50, "stddev": 5},
        )
    )

    composer = SyntheticRankingsDatasetComposer(run, mock_tokenizer)
    dataset = composer.create_dataset()

    # Verify that data was generated
    assert len(dataset) > 0

    # Check that each conversation has the expected structure
    for conv in dataset:
        assert len(conv.turns) == 1
        turn = conv.turns[0]
        assert len(turn.texts) == 2
        query, passages = turn.texts
        assert query.name == "query"
        assert passages.name == "passages"
        # Query and passages should have content
        assert len(query.contents) == 1
        assert len(passages.contents) >= 1
