# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace AutoProcessor wrapper for multimodal token estimation."""

import base64
import io
import logging
from typing import TYPE_CHECKING

from aiperf.common.models.modality_token_counts import ModalityTokenCounts

if TYPE_CHECKING:
    from PIL import Image as PILImage

_logger = logging.getLogger(__name__)


def _decode_data_uri(data_uri: str) -> "PILImage.Image":
    """Decode a base64 data URI to a PIL Image."""
    from PIL import Image as PILImage

    if data_uri.startswith("data:"):
        # Strip "data:<mime>;base64," prefix
        _, encoded = data_uri.split(",", 1)
    else:
        encoded = data_uri
    image_bytes = base64.b64decode(encoded)
    return PILImage.open(io.BytesIO(image_bytes))


class Processor:
    """Wrapper around HuggingFace AutoProcessor for multimodal token estimation."""

    def __init__(self, processor: object) -> None:
        self._processor = processor

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        trust_remote_code: bool = False,
        revision: str = "main",
    ) -> "Processor | None":
        """Load an AutoProcessor, returning None if the model has no processor."""
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                name,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
            _logger.info(f"Loaded AutoProcessor for '{name}'")
            return cls(processor)
        except Exception as e:
            _logger.info(f"AutoProcessor not available for '{name}': {e}")
            return None

    def estimate_input_modality_counts(
        self,
        text_parts: list[str],
        images: list[str],
    ) -> ModalityTokenCounts:
        """Estimate per-modality input token counts locally.

        Calls the processor with text+images for total, then text-only for text count.
        Image count is derived by subtraction at the AutoProcessor level.
        """
        pil_images = [_decode_data_uri(img) for img in images]
        combined_text = " ".join(text_parts) if text_parts else ""

        # Total tokens (text + images)
        total_inputs = self._processor(
            text=combined_text,
            images=pil_images,
            return_tensors="np",
        )
        total_len = total_inputs["input_ids"].shape[-1]

        # Text-only tokens
        text_inputs = self._processor(
            text=combined_text,
            return_tensors="np",
        )
        text_len = text_inputs["input_ids"].shape[-1]

        image_len = max(0, total_len - text_len)
        return ModalityTokenCounts(text=text_len, image=image_len)
