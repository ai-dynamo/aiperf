# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel


class ModalityTokenCounts(AIPerfBaseModel):
    """Per-modality token breakdown. Sum of non-None fields equals parent total."""

    text: int | None = Field(default=None, description="Text tokens.")
    image: int | None = Field(default=None, description="Image tokens.")
    video: int | None = Field(default=None, description="Video tokens.")
    audio: int | None = Field(default=None, description="Audio tokens.")

    @property
    def total(self) -> int:
        """Sum of all non-None modality counts."""
        return sum(
            v for v in (self.text, self.image, self.video, self.audio) if v is not None
        )

    def scale_to(self, target_total: int) -> "ModalityTokenCounts":
        """Scale proportionally so modality counts sum to target_total.

        The largest modality absorbs rounding remainder to preserve exact totals.
        """
        current_total = self.total
        if current_total == 0:
            return ModalityTokenCounts()

        fields = {
            "text": self.text,
            "image": self.image,
            "video": self.video,
            "audio": self.audio,
        }
        non_none = {k: v for k, v in fields.items() if v is not None}

        # Scale each proportionally
        scaled = {
            k: round(target_total * v / current_total) for k, v in non_none.items()
        }

        # Absorb rounding drift into the largest modality
        diff = target_total - sum(scaled.values())
        if diff != 0:
            largest = max(scaled, key=scaled.get)
            scaled[largest] += diff

        return ModalityTokenCounts(**scaled)
