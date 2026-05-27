# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from pytest import param

from aiperf.common.exceptions import TokenizerError
from aiperf.common.tokenizer import Tokenizer, _missing_tokenizer_class_hint

HF_MISSING_CLASS_ERROR = (
    "Tokenizer class TokenizersBackend does not exist or is not currently imported."
)


class TestMissingTokenizerClassHint:
    def test_matches_hf_missing_class_error(self) -> None:
        hint = _missing_tokenizer_class_hint(ValueError(HF_MISSING_CLASS_ERROR))
        assert hint is not None
        assert "TokenizersBackend" in hint
        assert "transformers" in hint

    def test_returns_none_for_unrelated_error(self) -> None:
        assert _missing_tokenizer_class_hint(ValueError("nope")) is None

    def test_returns_none_for_repo_not_found_error(self) -> None:
        # The OSError transformers raises when a repo / local path is invalid.
        msg = (
            "Can't load tokenizer for 'nvidia/typo'. If you were trying to load it "
            "from 'https://huggingface.co/models', make sure you don't have a local "
            "directory with the same name."
        )
        assert _missing_tokenizer_class_hint(OSError(msg)) is None

    @pytest.mark.parametrize(
        "version, expect_v5_suggestion",
        [
            param("4.57.3", True, id="v4-suggests-upgrade"),
            param("4.56.0", True, id="v4-floor-suggests-upgrade"),
            param("5.0.1", False, id="v5-omits-suggestion"),
            param("5.8.0", False, id="v5-omits-suggestion-later"),
        ],
    )  # fmt: skip
    def test_v5_suggestion_only_on_v4(
        self, version: str, expect_v5_suggestion: bool
    ) -> None:
        with patch("transformers.__version__", version):
            hint = _missing_tokenizer_class_hint(ValueError(HF_MISSING_CLASS_ERROR))
        assert hint is not None
        assert version in hint
        assert ("transformers>=5" in hint) is expect_v5_suggestion


class TestFromPretrainedWrapsHint:
    def test_wraps_missing_class_error_with_hint(self) -> None:
        with (
            patch(
                "aiperf.common.tokenizer.Tokenizer._load_from_hub",
                side_effect=ValueError(HF_MISSING_CLASS_ERROR),
            ),
            pytest.raises(TokenizerError) as excinfo,
        ):
            Tokenizer.from_pretrained("nvidia/GLM-5-NVFP4")
        message = str(excinfo.value)
        assert "Failed to load tokenizer 'nvidia/GLM-5-NVFP4'" in message
        assert "TokenizersBackend" in message
        assert "transformers" in message

    def test_wraps_unrelated_error_without_hint(self) -> None:
        with (
            patch(
                "aiperf.common.tokenizer.Tokenizer._load_from_hub",
                side_effect=OSError("Can't load tokenizer for 'nvidia/typo'."),
            ),
            pytest.raises(TokenizerError) as excinfo,
        ):
            Tokenizer.from_pretrained("nvidia/typo")
        message = str(excinfo.value)
        assert "Failed to load tokenizer 'nvidia/typo'" in message
        assert "Can't load tokenizer" in message
        # The v5 upgrade suggestion must NOT leak into unrelated errors.
        assert "transformers>=5" not in message
