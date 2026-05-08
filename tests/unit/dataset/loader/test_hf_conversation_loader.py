# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io

import pytest
from PIL import Image as PILImage
from pytest import param

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.models import Conversation
from aiperf.dataset.loader.hf_conversation import HFConversationDatasetLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


def _make_pil_image(width: int = 4, height: int = 4) -> PILImage.Image:
    return PILImage.new("RGB", (width, height), color=(255, 0, 0))


def _jpeg_bytes(width: int = 4, height: int = 4) -> bytes:
    buf = io.BytesIO()
    _make_pil_image(width, height).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def user_config() -> UserConfig:
    return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))


@pytest.fixture
async def loader(user_config: UserConfig) -> HFConversationDatasetLoader:
    return HFConversationDatasetLoader(
        user_config=user_config,
        hf_dataset_name="lmarena-ai/VisionArena-Chat",
        hf_split="train",
        conversation_column="conversation",
        message_content_key="content",
    )


@pytest.fixture
async def llava_loader(user_config: UserConfig) -> HFConversationDatasetLoader:
    return HFConversationDatasetLoader(
        user_config=user_config,
        hf_dataset_name="lmms-lab/LLaVA-OneVision-Data",
        hf_split="train",
        hf_subset="sharegpt4o",
        conversation_column="conversations",
        message_content_key="value",
        image_column="image",
    )


@pytest.mark.asyncio
class TestHFConversationDatasetLoader:
    async def test_preferred_sampling_strategy_is_sequential(self, loader):
        assert (
            loader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    async def test_extracts_first_message_as_prompt(self, loader):
        data = {
            "dataset": [
                {
                    "conversation": [
                        {"role": "user", "content": "What animal is in this image?"},
                        {"role": "assistant", "content": "It's a cat."},
                        {"role": "user", "content": "What color is it?"},
                    ]
                }
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert conversations[0].turns[0].texts[0].contents[0] == (
            "What animal is in this image?"
        )

    async def test_discards_subsequent_messages(self, loader):
        data = {
            "dataset": [
                {
                    "conversation": [
                        {"role": "user", "content": "First message"},
                        {"role": "assistant", "content": "Reply"},
                        {"role": "user", "content": "Follow up"},
                    ]
                }
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations[0].turns) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "First message"

    async def test_each_row_becomes_one_conversation(self, loader):
        data = {
            "dataset": [
                {"conversation": [{"role": "user", "content": f"Q{i}"}]}
                for i in range(3)
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 3
        assert all(isinstance(c, Conversation) for c in conversations)

    async def test_skips_empty_conversation(self, loader):
        data = {
            "dataset": [
                {"conversation": []},
                {"conversation": [{"role": "user", "content": "Valid"}]},
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "Valid"

    async def test_skips_non_dict_message(self, loader):
        data = {
            "dataset": [
                {"conversation": ["raw_string_message"]},
                {"conversation": [{"role": "user", "content": "Valid"}]},
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "Valid"

    async def test_skips_non_str_content_value(self, loader):
        data = {
            "dataset": [
                {"conversation": [{"role": "user", "content": ["list", "value"]}]},
                {"conversation": [{"role": "user", "content": None}]},
                {"conversation": [{"role": "user", "content": "Valid"}]},
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "Valid"

    async def test_skips_missing_conversation_column(self, loader):
        data = {
            "dataset": [
                {"other_field": "value"},
                {"conversation": [{"role": "user", "content": "Valid"}]},
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1

    async def test_skips_empty_prompt(self, loader):
        data = {
            "dataset": [
                {"conversation": [{"role": "user", "content": ""}]},
                {"conversation": [{"role": "user", "content": "   "}]},
                {"conversation": [{"role": "user", "content": "Valid"}]},
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1

    async def test_session_ids_are_unique(self, loader):
        data = {
            "dataset": [
                {"conversation": [{"role": "user", "content": f"Q{i}"}]}
                for i in range(5)
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        session_ids = [c.session_id for c in conversations]
        assert len(set(session_ids)) == 5

    async def test_unwraps_list_of_lists_turns(self, loader):
        # VisionArena wraps each turn in its own list
        data = {
            "dataset": [
                {
                    "conversation": [
                        [{"content": "What's this?", "role": "user"}],
                        [{"content": "It's a cat.", "role": "assistant"}],
                    ]
                }
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert conversations[0].turns[0].texts[0].contents[0] == "What's this?"

    async def test_strips_image_placeholder_token(self, llava_loader):
        data = {
            "dataset": [
                {
                    "conversations": [
                        {"from": "human", "value": "<image>\nDescribe this image."},
                    ],
                    "image": None,
                }
            ]
        }
        conversations = await llava_loader.convert_to_conversations(data)
        assert conversations[0].turns[0].texts[0].contents[0] == "Describe this image."

    async def test_uses_custom_message_content_key(self, llava_loader):
        data = {
            "dataset": [
                {
                    "conversations": [
                        {"from": "human", "value": "Describe the scene."},
                        {"from": "gpt", "value": "A busy street."},
                    ],
                    "image": None,
                }
            ]
        }
        conversations = await llava_loader.convert_to_conversations(data)
        assert conversations[0].turns[0].texts[0].contents[0] == "Describe the scene."

    async def test_attaches_single_pil_image(self, llava_loader):
        pil_img = _make_pil_image()
        data = {
            "dataset": [
                {
                    "conversations": [{"from": "human", "value": "What is this?"}],
                    "image": pil_img,
                }
            ]
        }
        conversations = await llava_loader.convert_to_conversations(data)
        turn = conversations[0].turns[0]
        assert len(turn.images) == 1
        assert turn.images[0].contents[0].startswith("data:image/jpeg;base64,")

    async def test_attaches_first_image_from_list(self, user_config):
        loader = HFConversationDatasetLoader(
            user_config=user_config,
            hf_dataset_name="lmarena-ai/VisionArena-Chat",
            hf_split="train",
            conversation_column="conversation",
            message_content_key="content",
            image_column="images",
        )
        pil_img1 = _make_pil_image()
        pil_img2 = _make_pil_image(8, 8)
        data = {
            "dataset": [
                {
                    "conversation": [{"role": "user", "content": "What is this?"}],
                    "images": [pil_img1, pil_img2],
                }
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations[0].turns[0].images) == 1

    @pytest.mark.parametrize(
        "images_value",
        [
            param(
                {"bytes": _jpeg_bytes(), "path": None},
                id="undecoded-bytes-dict-scalar",
            ),
            param(
                [{"bytes": _jpeg_bytes(), "path": None}],
                id="undecoded-bytes-dict-list",
            ),
            param(
                [
                    {"bytes": _jpeg_bytes(), "path": None},
                    {"bytes": _jpeg_bytes(8, 8), "path": None},
                ],
                id="undecoded-bytes-dict-list-multiple",
            ),
        ],
    )  # fmt: skip
    async def test_attaches_image_from_undecoded_hf_dict(
        self, user_config, images_value
    ):
        loader = HFConversationDatasetLoader(
            user_config=user_config,
            hf_dataset_name="lmarena-ai/VisionArena-Chat",
            hf_split="train",
            conversation_column="conversation",
            message_content_key="content",
            image_column="images",
        )
        data = {
            "dataset": [
                {
                    "conversation": [{"role": "user", "content": "What is this?"}],
                    "images": images_value,
                }
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        turn = conversations[0].turns[0]
        assert len(turn.images) == 1
        assert turn.images[0].contents[0].startswith("data:image/jpeg;base64,")

    async def test_skips_corrupt_image_bytes(self, user_config):
        loader = HFConversationDatasetLoader(
            user_config=user_config,
            hf_dataset_name="lmarena-ai/VisionArena-Chat",
            hf_split="train",
            conversation_column="conversation",
            message_content_key="content",
            image_column="images",
        )
        data = {
            "dataset": [
                {
                    "conversation": [{"role": "user", "content": "Bad image"}],
                    "images": [{"bytes": b"not-a-real-image", "path": None}],
                }
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert conversations[0].turns[0].images == []

    async def test_path_only_dict_returns_no_images(self, user_config):
        # Locks in the documented "not handled" contract for path-only HF dicts
        # (bytes is None, path is a string). Update if path-only is ever supported.
        loader = HFConversationDatasetLoader(
            user_config=user_config,
            hf_dataset_name="lmarena-ai/VisionArena-Chat",
            hf_split="train",
            conversation_column="conversation",
            message_content_key="content",
            image_column="images",
        )
        data = {
            "dataset": [
                {
                    "conversation": [{"role": "user", "content": "Path only"}],
                    "images": [{"bytes": None, "path": "some_image.jpg"}],
                }
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert conversations[0].turns[0].images == []

    async def test_skips_truncated_image_at_load_time(self, user_config):
        # Truncated valid JPEG: header passes PILImage.open (lazy) but the
        # subsequent re-encode in _pil_to_image raises OSError. Locks in the
        # widened try/except so one corrupt row can't abort the full loader run.
        full = _jpeg_bytes(256, 256)
        truncated = full[: int(len(full) * 0.95)]
        loader = HFConversationDatasetLoader(
            user_config=user_config,
            hf_dataset_name="lmarena-ai/VisionArena-Chat",
            hf_split="train",
            conversation_column="conversation",
            message_content_key="content",
            image_column="images",
        )
        data = {
            "dataset": [
                {
                    "conversation": [{"role": "user", "content": "Truncated"}],
                    "images": [{"bytes": truncated, "path": None}],
                },
                {
                    "conversation": [{"role": "user", "content": "Good"}],
                    "images": [{"bytes": full, "path": None}],
                },
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 2
        assert conversations[0].turns[0].images == []
        assert len(conversations[1].turns[0].images) == 1

    async def test_skips_truncated_pil_image_at_load_time(self, user_config):
        # Lazy PIL Image (from PILImage.open on truncated bytes) passes the
        # isinstance(PILImage.Image) check but raises OSError when _pil_to_image
        # forces re-encode. Locks in the symmetric try/except on the PIL branch
        # so HF decode=True datasets carrying a corrupt lazy image are also
        # skipped instead of aborting the loader.
        full = _jpeg_bytes(256, 256)
        truncated_pil = PILImage.open(io.BytesIO(full[: int(len(full) * 0.95)]))
        good_pil = _make_pil_image()
        loader = HFConversationDatasetLoader(
            user_config=user_config,
            hf_dataset_name="lmarena-ai/VisionArena-Chat",
            hf_split="train",
            conversation_column="conversation",
            message_content_key="content",
            image_column="images",
        )
        data = {
            "dataset": [
                {
                    "conversation": [{"role": "user", "content": "Truncated PIL"}],
                    "images": [truncated_pil],
                },
                {
                    "conversation": [{"role": "user", "content": "Good"}],
                    "images": [good_pil],
                },
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 2
        assert conversations[0].turns[0].images == []
        assert len(conversations[1].turns[0].images) == 1

    async def test_no_images_when_image_column_not_set(self, loader):
        data = {
            "dataset": [{"conversation": [{"role": "user", "content": "Text only"}]}]
        }
        conversations = await loader.convert_to_conversations(data)
        assert conversations[0].turns[0].images == []

    async def test_empty_dataset_returns_empty_list(self, loader):
        conversations = await loader.convert_to_conversations({"dataset": []})
        assert conversations == []

    async def test_non_streaming_returns_all_rows(self, user_config):
        from aiperf.common.config.loadgen_config import LoadGeneratorConfig

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=3),
        )
        loader = HFConversationDatasetLoader(
            user_config=config,
            hf_dataset_name="test/data",
            hf_split="train",
            conversation_column="conversation",
            streaming=False,
        )
        data = {
            "dataset": [
                {"conversation": [{"role": "user", "content": f"Q{i}"}]}
                for i in range(10)
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 10

    async def test_streaming_capped_by_request_count(self, user_config):
        from aiperf.common.config.loadgen_config import LoadGeneratorConfig

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=3),
        )
        loader = HFConversationDatasetLoader(
            user_config=config,
            hf_dataset_name="test/data",
            hf_split="train",
            conversation_column="conversation",
            streaming=True,
        )
        data = {
            "dataset": [
                {"conversation": [{"role": "user", "content": f"Q{i}"}]}
                for i in range(10)
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 3

    async def test_streaming_falls_back_to_num_dataset_entries(self, user_config):
        from aiperf.common.config.conversation_config import ConversationConfig
        from aiperf.common.config.loadgen_config import LoadGeneratorConfig

        conversation = ConversationConfig(num_dataset_entries=4)
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input={"conversation": conversation},
            loadgen=LoadGeneratorConfig(benchmark_duration=60),
        )
        loader = HFConversationDatasetLoader(
            user_config=config,
            hf_dataset_name="test/data",
            hf_split="train",
            conversation_column="conversation",
            streaming=True,
        )
        data = {
            "dataset": [
                {"conversation": [{"role": "user", "content": f"Q{i}"}]}
                for i in range(10)
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 4

    async def test_streaming_defaults_to_false(self, loader):
        assert loader.streaming is False

    async def test_streaming_stored_when_true(self, user_config):
        loader = HFConversationDatasetLoader(
            user_config=user_config,
            hf_dataset_name="test/data",
            hf_split="train",
            conversation_column="conversation",
            streaming=True,
        )
        assert loader.streaming is True
