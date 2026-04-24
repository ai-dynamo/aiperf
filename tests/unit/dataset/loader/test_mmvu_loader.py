# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.models import Conversation
from aiperf.dataset.loader.mmvu import MMVUDatasetLoader


@pytest.fixture
def user_config() -> UserConfig:
    return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))


@pytest.fixture
async def loader(user_config: UserConfig) -> MMVUDatasetLoader:
    return MMVUDatasetLoader(
        user_config=user_config,
        hf_dataset_name="yale-nlp/MMVU",
        hf_split="validation",
        video_column="video",
    )


class TestMMVUFormatPrompt:
    def test_multiple_choice_formats_choices(self):
        row = {
            "question": "What technique is shown?",
            "choices": {
                "A": "Dolly Zoom",
                "B": "Pan",
                "C": "Tilt",
                "D": "Zoom",
                "E": "",
            },
        }
        prompt = MMVUDatasetLoader._format_prompt(row)
        assert prompt == "What technique is shown? A.Dolly Zoom B.Pan C.Tilt D.Zoom"

    def test_open_ended_skips_empty_choices(self):
        row = {
            "question": "What algorithm is used?",
            "choices": {"A": "", "B": "", "C": "", "D": "", "E": ""},
        }
        prompt = MMVUDatasetLoader._format_prompt(row)
        assert prompt == "What algorithm is used?"

    def test_missing_choices_returns_question_only(self):
        row = {"question": "Describe the video."}
        prompt = MMVUDatasetLoader._format_prompt(row)
        assert prompt == "Describe the video."

    def test_non_dict_choices_returns_question_only(self):
        row = {"question": "What is shown?", "choices": "A.Yes B.No"}
        prompt = MMVUDatasetLoader._format_prompt(row)
        assert prompt == "What is shown?"

    def test_empty_question_with_choices_returns_choices_only(self):
        row = {"question": "", "choices": {"A": "Yes", "B": "No"}}
        prompt = MMVUDatasetLoader._format_prompt(row)
        assert prompt == "A.Yes B.No"

    def test_none_question_does_not_produce_none_string(self):
        row = {"question": None, "choices": {"A": "Yes", "B": "No"}}
        prompt = MMVUDatasetLoader._format_prompt(row)
        assert "None" not in prompt
        assert prompt == "A.Yes B.No"

    def test_whitespace_only_question_treated_as_empty(self):
        row = {"question": "   ", "choices": {"A": "Yes", "B": "No"}}
        prompt = MMVUDatasetLoader._format_prompt(row)
        assert prompt == "A.Yes B.No"


@pytest.mark.asyncio
class TestMMVUConvertToConversations:
    async def test_converts_rows_to_conversations(self, loader):
        data = {
            "dataset": [
                {
                    "question": "What technique is shown?",
                    "choices": {
                        "A": "Dolly Zoom",
                        "B": "Pan",
                        "C": "",
                        "D": "",
                        "E": "",
                    },
                    "video": "https://example.com/video.mp4",
                }
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert isinstance(conversations[0], Conversation)

    async def test_prompt_text_matches_format_prompt(self, loader):
        data = {
            "dataset": [
                {
                    "question": "What is shown?",
                    "choices": {"A": "Cat", "B": "Dog", "C": "", "D": "", "E": ""},
                    "video": "https://example.com/video.mp4",
                }
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert (
            conversations[0].turns[0].texts[0].contents[0]
            == "What is shown? A.Cat B.Dog"
        )

    async def test_video_url_attached_to_turn(self, loader):
        data = {
            "dataset": [
                {
                    "question": "What is shown?",
                    "choices": {"A": "", "B": "", "C": "", "D": "", "E": ""},
                    "video": "https://example.com/0.mp4",
                }
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        turn = conversations[0].turns[0]
        assert len(turn.videos) == 1
        assert turn.videos[0].contents[0] == "https://example.com/0.mp4"

    async def test_skips_rows_with_empty_question(self, loader):
        data = {
            "dataset": [
                {"question": "", "choices": {}, "video": "https://example.com/0.mp4"},
                {
                    "question": "Valid?",
                    "choices": {},
                    "video": "https://example.com/1.mp4",
                },
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1

    async def test_skips_rows_with_missing_video(self, loader):
        data = {
            "dataset": [
                {"question": "Q1?", "choices": {}, "video": None},
                {
                    "question": "Q2?",
                    "choices": {},
                    "video": "https://example.com/1.mp4",
                },
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert conversations[0].turns[0].texts[0].contents[0] == "Q2?"

    async def test_empty_dataset_returns_empty_list(self, loader):
        data = {"dataset": []}
        conversations = await loader.convert_to_conversations(data)
        assert conversations == []

    async def test_session_ids_are_unique(self, loader):
        data = {
            "dataset": [
                {
                    "question": f"Q{i}",
                    "choices": {},
                    "video": f"https://example.com/{i}.mp4",
                }
                for i in range(5)
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        session_ids = [c.session_id for c in conversations]
        assert len(set(session_ids)) == 5

    async def test_each_row_becomes_single_turn(self, loader):
        data = {
            "dataset": [
                {"question": "Q", "choices": {}, "video": "https://example.com/0.mp4"}
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations[0].turns) == 1


class TestExtractVideos:
    def test_url_string_stored_as_is(self, loader):
        row = {"video": "https://huggingface.co/datasets/yale-nlp/MMVU/0.mp4"}
        videos = loader._extract_videos(row, "video")
        assert len(videos) == 1
        assert (
            videos[0].contents[0]
            == "https://huggingface.co/datasets/yale-nlp/MMVU/0.mp4"
        )

    def test_local_path_prefixed_with_file_scheme(self, loader):
        row = {"video": "/tmp/video.mp4"}
        videos = loader._extract_videos(row, "video")
        assert videos[0].contents[0] == "file:///tmp/video.mp4"

    def test_file_url_passed_through(self, loader):
        row = {"video": "file:///tmp/video.mp4"}
        videos = loader._extract_videos(row, "video")
        assert videos[0].contents[0] == "file:///tmp/video.mp4"

    def test_data_uri_passed_through(self, loader):
        data_uri = "data:video/mp4;base64,abc123"
        row = {"video": data_uri}
        videos = loader._extract_videos(row, "video")
        assert videos[0].contents[0] == data_uri

    def test_bytes_dict_defaults_to_mp4_mime(self, loader):
        raw = b"fake-video-bytes"
        row = {"video": {"bytes": raw}}
        videos = loader._extract_videos(row, "video")
        expected = f"data:video/mp4;base64,{base64.b64encode(raw).decode()}"
        assert videos[0].contents[0] == expected

    def test_bytes_dict_infers_mime_from_path(self, loader):
        raw = b"fake-video-bytes"
        row = {"video": {"bytes": raw, "path": "clip.webm"}}
        videos = loader._extract_videos(row, "video")
        assert videos[0].contents[0].startswith("data:video/webm;base64,")

    def test_missing_column_returns_empty(self, loader):
        row = {"other": "value"}
        videos = loader._extract_videos(row, "video")
        assert videos == []

    def test_none_value_returns_empty(self, loader):
        row = {"video": None}
        videos = loader._extract_videos(row, "video")
        assert videos == []

    def test_empty_string_returns_empty(self, loader):
        row = {"video": ""}
        videos = loader._extract_videos(row, "video")
        assert videos == []

    def test_bytes_dict_with_none_bytes_returns_empty(self, loader):
        row = {"video": {"bytes": None}}
        videos = loader._extract_videos(row, "video")
        assert videos == []
