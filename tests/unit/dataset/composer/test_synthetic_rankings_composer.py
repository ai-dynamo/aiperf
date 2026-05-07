# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

from aiperf.common.models import Conversation, Turn
from aiperf.dataset.composer.synthetic_rankings import SyntheticRankingsDatasetComposer


def test_initialization_basic(synthetic_config, mock_tokenizer):
    """Ensure SyntheticRankingsDatasetComposer initializes correctly."""
    composer = SyntheticRankingsDatasetComposer(synthetic_config, mock_tokenizer)
    assert composer.session_id_generator is not None


def test_create_dataset_structure(synthetic_config, mock_tokenizer):
    """Test structure and content of generated synthetic ranking dataset."""
    synthetic_config.input.rankings.passages.mean = 5
    synthetic_config.input.rankings.passages.stddev = 1
    composer = SyntheticRankingsDatasetComposer(synthetic_config, mock_tokenizer)

    dataset = composer.create_dataset()
    assert len(dataset) == synthetic_config.input.conversation.num_dataset_entries

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


def test_passage_count_distribution(synthetic_config, mock_tokenizer):
    """Test passages are generated following mean/stddev distribution."""
    synthetic_config.input.rankings.passages.mean = 5
    synthetic_config.input.rankings.passages.stddev = 2
    composer = SyntheticRankingsDatasetComposer(synthetic_config, mock_tokenizer)

    dataset = composer.create_dataset()
    passage_counts = [len(conv.turns[0].texts[1].contents) for conv in dataset]

    assert all(1 <= c <= 10 for c in passage_counts)
    assert len(set(passage_counts)) > 1  # variation expected


def test_reproducibility_fixed_seed(synthetic_config, mock_tokenizer):
    """Dataset generation should be deterministic given a fixed random seed."""
    synthetic_config.input.rankings.passages.mean = 4
    synthetic_config.input.rankings.passages.stddev = 1
    synthetic_config.input.random_seed = 42

    composer1 = SyntheticRankingsDatasetComposer(synthetic_config, mock_tokenizer)
    data1 = composer1.create_dataset()

    composer2 = SyntheticRankingsDatasetComposer(synthetic_config, mock_tokenizer)
    data2 = composer2.create_dataset()

    # Session IDs differ (fresh), but text contents should match
    for c1, c2 in zip(data1, data2, strict=True):
        t1, t2 = c1.turns[0], c2.turns[0]
        assert t1.texts[0].contents == t2.texts[0].contents
        assert t1.texts[1].contents == t2.texts[1].contents


def test_rankings_specific_token_options(synthetic_config, mock_tokenizer):
    """Test that rankings-specific token options are used for query and passages."""
    synthetic_config.input.rankings.passages.mean = 3
    synthetic_config.input.rankings.passages.prompt_token_mean = 100
    synthetic_config.input.rankings.passages.prompt_token_stddev = 10
    synthetic_config.input.rankings.query.prompt_token_mean = 50
    synthetic_config.input.rankings.query.prompt_token_stddev = 5
    synthetic_config.input.random_seed = 42

    composer = SyntheticRankingsDatasetComposer(synthetic_config, mock_tokenizer)
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


def test_synthetic_images_and_videos_match_passage_count(
    synthetic_config, mock_tokenizer
):
    """Synthetic rankings media should be generated per passage for paired docs."""
    synthetic_config.input.rankings.passages.mean = 3
    synthetic_config.input.rankings.passages.stddev = 0
    synthetic_config.input.image.width.mean = 8
    synthetic_config.input.image.height.mean = 8
    synthetic_config.input.image.batch_size = 1
    synthetic_config.input.video.width = 8
    synthetic_config.input.video.height = 8
    synthetic_config.input.video.batch_size = 1

    composer = SyntheticRankingsDatasetComposer(synthetic_config, mock_tokenizer)

    with (
        patch.object(
            composer.image_generator,
            "generate",
            side_effect=lambda: "data:image/png;base64,test",
        ) as generate_image,
        patch.object(
            composer.video_generator,
            "generate",
            side_effect=lambda: "data:video/webm;base64,test",
        ) as generate_video,
    ):
        dataset = composer.create_dataset()

    expected_media_count = synthetic_config.input.conversation.num_dataset_entries * 3
    assert generate_image.call_count == expected_media_count
    assert generate_video.call_count == expected_media_count

    for conversation in dataset:
        turn = conversation.turns[0]
        passage_count = len(turn.texts[1].contents)

        assert len(turn.images) == 1
        assert turn.images[0].name == "image_url"
        assert len(turn.images[0].contents) == passage_count

        assert len(turn.videos) == 1
        assert turn.videos[0].name == "video_url"
        assert len(turn.videos[0].contents) == passage_count


def test_synthetic_rankings_media_batch_size_zero_disables_media(
    synthetic_config, mock_tokenizer
):
    """Batch size zero disables synthetic rankings media even with dimensions set."""
    synthetic_config.input.rankings.passages.mean = 2
    synthetic_config.input.rankings.passages.stddev = 0
    synthetic_config.input.image.width.mean = 8
    synthetic_config.input.image.height.mean = 8
    synthetic_config.input.image.batch_size = 0
    synthetic_config.input.video.width = 8
    synthetic_config.input.video.height = 8
    synthetic_config.input.video.batch_size = 0

    composer = SyntheticRankingsDatasetComposer(synthetic_config, mock_tokenizer)
    dataset = composer.create_dataset()

    for conversation in dataset:
        turn = conversation.turns[0]
        assert turn.images == []
        assert turn.videos == []
