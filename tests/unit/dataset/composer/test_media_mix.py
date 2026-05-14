# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import Counter

from aiperf.common.config import (
    AudioLengthConfig,
    ConversationConfig,
    EndpointConfig,
    ImageHeightConfig,
    ImageWidthConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.common.config.media_mix_config import (
    AudioProfileConfig,
    ImageProfileConfig,
    MediaMixArchetype,
    ModalityEntry,
    TextOverrideConfig,
)
from aiperf.common.config.prompt_config import OutputTokensConfig
from aiperf.dataset.composer.media_mix_resolver import MediaMixResolver
from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer


class TestMediaMixResolver:
    def test_single_archetype_always_selected(self):
        archetypes = [
            MediaMixArchetype(
                weight=1.0,
                modalities=[
                    ModalityEntry(
                        modality="image",
                        profiles=[
                            ImageProfileConfig(
                                weight=1.0,
                                width=ImageWidthConfig(mean=256),
                                height=ImageHeightConfig(mean=256),
                            )
                        ],
                    )
                ],
            )
        ]
        resolver = MediaMixResolver(archetypes)
        resolved = resolver.resolve_turn()
        assert len(resolved.image_generators) == 1
        assert len(resolved.audio_generators) == 0
        assert len(resolved.video_generators) == 0
        assert resolved.include_text is True

    def test_text_disabled_archetype(self):
        archetypes = [
            MediaMixArchetype(
                weight=1.0,
                text=False,
                modalities=[
                    ModalityEntry(
                        modality="image",
                        profiles=[
                            ImageProfileConfig(
                                weight=1.0,
                                width=ImageWidthConfig(mean=256),
                                height=ImageHeightConfig(mean=256),
                            )
                        ],
                    )
                ],
            )
        ]
        resolver = MediaMixResolver(archetypes)
        resolved = resolver.resolve_turn()
        assert resolved.include_text is False

    def test_text_override_propagated(self):
        archetypes = [
            MediaMixArchetype(
                weight=1.0,
                text=TextOverrideConfig(
                    input_tokens=InputTokensConfig(mean=100, stddev=20),
                    output_tokens=OutputTokensConfig(mean=500),
                ),
                modalities=[],
            )
        ]
        resolver = MediaMixResolver(archetypes)
        resolved = resolver.resolve_turn()
        assert resolved.include_text is True
        assert resolved.input_tokens_mean == 100
        assert resolved.input_tokens_stddev == 20
        assert resolved.output_tokens_mean == 500

    def test_text_no_override_returns_none(self):
        archetypes = [MediaMixArchetype(weight=1.0, modalities=[])]
        resolver = MediaMixResolver(archetypes)
        resolved = resolver.resolve_turn()
        assert resolved.input_tokens_mean is None
        assert resolved.output_tokens_mean is None

    def test_multi_modality_archetype(self):
        archetypes = [
            MediaMixArchetype(
                weight=1.0,
                modalities=[
                    ModalityEntry(
                        modality="image",
                        batch_size=2,
                        profiles=[
                            ImageProfileConfig(
                                weight=1.0,
                                width=ImageWidthConfig(mean=256),
                                height=ImageHeightConfig(mean=256),
                            )
                        ],
                    ),
                    ModalityEntry(
                        modality="audio",
                        batch_size=1,
                        profiles=[
                            AudioProfileConfig(
                                weight=1.0,
                                length=AudioLengthConfig(mean=5.0),
                            )
                        ],
                    ),
                ],
            )
        ]
        resolver = MediaMixResolver(archetypes)
        resolved = resolver.resolve_turn()
        assert len(resolved.image_generators) == 1
        assert resolved.image_generators[0][1] == 2  # batch_size
        assert len(resolved.audio_generators) == 1
        assert resolved.audio_generators[0][1] == 1

    def test_weighted_archetype_distribution(self):
        """Over many samples, archetype selection should follow weights."""
        archetypes = [
            MediaMixArchetype(
                weight=0.8,
                name="image",
                modalities=[
                    ModalityEntry(
                        modality="image",
                        profiles=[
                            ImageProfileConfig(
                                weight=1.0,
                                width=ImageWidthConfig(mean=256),
                                height=ImageHeightConfig(mean=256),
                            )
                        ],
                    )
                ],
            ),
            MediaMixArchetype(
                weight=0.2,
                name="audio",
                modalities=[
                    ModalityEntry(
                        modality="audio",
                        profiles=[
                            AudioProfileConfig(
                                weight=1.0,
                                length=AudioLengthConfig(mean=5.0),
                            )
                        ],
                    )
                ],
            ),
        ]
        resolver = MediaMixResolver(archetypes)

        counts = Counter()
        for _ in range(1000):
            resolved = resolver.resolve_turn()
            if resolved.image_generators:
                counts["image"] += 1
            if resolved.audio_generators:
                counts["audio"] += 1

        # With 0.8/0.2 weights over 1000 samples, expect roughly 800/200
        assert counts["image"] > 600
        assert counts["audio"] > 100
        assert counts["image"] + counts["audio"] == 1000

    def test_batch_size_preserved(self):
        archetypes = [
            MediaMixArchetype(
                weight=1.0,
                modalities=[
                    ModalityEntry(
                        modality="image",
                        batch_size=5,
                        profiles=[
                            ImageProfileConfig(
                                weight=1.0,
                                width=ImageWidthConfig(mean=256),
                                height=ImageHeightConfig(mean=256),
                            )
                        ],
                    )
                ],
            )
        ]
        resolver = MediaMixResolver(archetypes)
        resolved = resolver.resolve_turn()
        _, batch_size = resolved.image_generators[0]
        assert batch_size == 5


class TestSyntheticComposerMediaMix:
    def test_media_mix_creates_dataset(self, mock_tokenizer):
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                conversation=ConversationConfig(num_dataset_entries=10),
                prompt=PromptConfig(
                    input_tokens=InputTokensConfig(mean=10, stddev=0),
                ),
                media_mix=[
                    MediaMixArchetype(
                        weight=0.5,
                        modalities=[
                            ModalityEntry(
                                modality="image",
                                profiles=[
                                    ImageProfileConfig(
                                        weight=1.0,
                                        width=ImageWidthConfig(mean=10),
                                        height=ImageHeightConfig(mean=10),
                                    )
                                ],
                            )
                        ],
                    ),
                    MediaMixArchetype(
                        weight=0.5,
                        modalities=[],
                    ),
                ],
            ),
        )
        composer = SyntheticDatasetComposer(config, mock_tokenizer)
        dataset = composer.create_dataset()
        assert len(dataset) == 10

        # Some turns should have images, some should not
        has_images = 0
        no_images = 0
        for conv in dataset:
            for turn in conv.turns:
                if turn.images:
                    has_images += 1
                else:
                    no_images += 1
        assert has_images > 0
        assert no_images > 0

    def test_media_mix_text_disabled_archetype(self, mock_tokenizer):
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                conversation=ConversationConfig(num_dataset_entries=5),
                prompt=PromptConfig(
                    input_tokens=InputTokensConfig(mean=10, stddev=0),
                ),
                media_mix=[
                    MediaMixArchetype(
                        weight=1.0,
                        text=False,
                        modalities=[
                            ModalityEntry(
                                modality="image",
                                profiles=[
                                    ImageProfileConfig(
                                        weight=1.0,
                                        width=ImageWidthConfig(mean=10),
                                        height=ImageHeightConfig(mean=10),
                                    )
                                ],
                            )
                        ],
                    ),
                ],
            ),
        )
        composer = SyntheticDatasetComposer(config, mock_tokenizer)
        dataset = composer.create_dataset()

        for conv in dataset:
            for turn in conv.turns:
                assert len(turn.texts) == 0
                assert len(turn.images) == 1

    def test_media_mix_bypasses_disabled_check(self, mock_tokenizer):
        """Media mix should work even when global prompt/image/audio are all disabled."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                conversation=ConversationConfig(num_dataset_entries=2),
                prompt=PromptConfig(
                    input_tokens=InputTokensConfig(mean=0),
                ),
                media_mix=[
                    MediaMixArchetype(
                        weight=1.0,
                        modalities=[
                            ModalityEntry(
                                modality="image",
                                profiles=[
                                    ImageProfileConfig(
                                        weight=1.0,
                                        width=ImageWidthConfig(mean=10),
                                        height=ImageHeightConfig(mean=10),
                                    )
                                ],
                            )
                        ],
                    ),
                ],
            ),
        )
        # Should not raise "All synthetic data are disabled"
        composer = SyntheticDatasetComposer(config, mock_tokenizer)
        dataset = composer.create_dataset()
        assert len(dataset) == 2
