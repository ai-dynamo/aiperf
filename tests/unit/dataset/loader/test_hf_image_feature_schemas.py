# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that exercise the real HuggingFace ``datasets`` library against the
loaders, so the dataset row shapes come from HF's own feature decoding rather
than hand-rolled mocks. Catches schema-shape surprises like the VisionArena
``Image(decode=False)`` regression where handcrafted ``[PIL, PIL]`` lists in
unit tests didn't match what HF actually emitted (``[{"bytes": ...}, ...]``).
"""

import io
from typing import Any

import pytest
from datasets import Dataset, Sequence
from datasets import Image as HFImage
from PIL import Image as PILImage

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.dataset.loader.hf_conversation import HFConversationDatasetLoader
from aiperf.dataset.loader.hf_instruction_response import (
    HFInstructionResponseDatasetLoader,
)


def _jpeg_bytes(width: int = 4, height: int = 4) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (width, height), color=(255, 0, 0)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def user_config() -> UserConfig:
    return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))


def _vision_arena_dataset(image_feature: Any) -> Dataset:
    """Build a VisionArena-shaped Dataset with the given Image feature.

    Schema matches lmarena-ai/VisionArena-Chat:
    ``conversation: List(List({content, role}))``,
    ``images: <feature>``.
    """
    ds = Dataset.from_dict(
        {
            "conversation": [[[{"role": "user", "content": "What is this?"}]]],
            "images": [[{"bytes": _jpeg_bytes(), "path": None}]],
        }
    )
    return ds.cast_column("images", image_feature)


@pytest.mark.asyncio
class TestHFConversationLoaderRealSchemas:
    """Run HFConversationDatasetLoader against real HF Datasets so the row
    shapes are produced by the actual ``datasets`` library, not by us."""

    def _loader(self, user_config: UserConfig) -> HFConversationDatasetLoader:
        return HFConversationDatasetLoader(
            user_config=user_config,
            hf_dataset_name="lmarena-ai/VisionArena-Chat",
            hf_split="train",
            conversation_column="conversation",
            message_content_key="content",
            image_column="images",
        )

    async def test_vision_arena_schema_decode_false_yields_image(self, user_config):
        """The original bug: List(Image(decode=False)) → list[dict] from HF.

        Pre-fix, _extract_images returned [] for this shape and inputs.json
        was text-only.
        """
        dataset = _vision_arena_dataset(Sequence(HFImage(decode=False)))
        assert str(dataset.features["images"]) == "List(Image(mode=None, decode=False))"

        loader = self._loader(user_config)
        conversations = await loader.convert_to_conversations({"dataset": dataset})

        assert len(conversations) == 1
        turn = conversations[0].turns[0]
        assert len(turn.images) == 1, (
            "vision_arena schema must yield image content; this regression slipped "
            "past hand-rolled-dict tests because the real HF row shape was unknown."
        )
        assert turn.images[0].contents[0].startswith("data:image/jpeg;base64,")

    async def test_vision_arena_schema_decode_false_streaming(self, user_config):
        """Streaming flag must not change image extraction behavior."""
        dataset = _vision_arena_dataset(Sequence(HFImage(decode=False)))
        streaming = dataset.to_iterable_dataset()

        loader = self._loader(user_config)
        conversations = await loader.convert_to_conversations({"dataset": streaming})

        assert len(conversations[0].turns[0].images) == 1

    async def test_decode_true_list_schema_still_works(self, user_config):
        """Regression coverage: Sequence(Image(decode=True)) keeps yielding
        PIL Images; the unified loop must handle both shapes."""
        dataset = _vision_arena_dataset(Sequence(HFImage(decode=True)))

        loader = self._loader(user_config)
        conversations = await loader.convert_to_conversations({"dataset": dataset})

        turn = conversations[0].turns[0]
        assert len(turn.images) == 1
        assert turn.images[0].contents[0].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
class TestHFInstructionResponseLoaderRealSchemas:
    """MMStar-style: scalar Image(decode=True). Regression coverage so the
    common decoded-PIL path never breaks while we evolve the loader."""

    async def test_mmstar_schema_decode_true_yields_image(self, user_config):
        ds = Dataset.from_dict(
            {
                "question": ["Describe this image."],
                "image": [{"bytes": _jpeg_bytes(), "path": None}],
            }
        ).cast_column("image", HFImage(decode=True))
        assert str(ds.features["image"]) == "Image(mode=None, decode=True)"

        loader = HFInstructionResponseDatasetLoader(
            user_config=user_config,
            hf_dataset_name="Lin-Chen/MMStar",
            hf_split="val",
            prompt_column="question",
            image_column="image",
        )
        conversations = await loader.convert_to_conversations({"dataset": ds})

        turn = conversations[0].turns[0]
        assert len(turn.images) == 1
        assert turn.images[0].contents[0].startswith("data:image/jpeg;base64,")
