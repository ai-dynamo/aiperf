# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import io
import mimetypes
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import soundfile as sf
from datasets import load_dataset as hf_load_dataset
from PIL import Image as PILImage

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Audio, Conversation, Image, Video
from aiperf.dataset import utils
from aiperf.dataset.loader.base_public_dataset import BasePublicDatasetLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class BaseHFDatasetLoader(BasePublicDatasetLoader):
    """Base class for loading datasets from HuggingFace via the datasets library."""

    def __init__(
        self,
        run: BenchmarkRun | None = None,
        *,
        hf_dataset_name: str,
        hf_split: str = "train",
        hf_subset: str | None = None,
        streaming: bool = False,
        **kwargs,
    ) -> None:
        self.hf_dataset_name = hf_dataset_name
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.streaming = streaming
        super().__init__(run=run, **kwargs)

    async def load_dataset(self) -> dict[str, Any]:
        """Load the dataset from HuggingFace."""
        self.info(
            f"Loading HuggingFace dataset '{self.hf_dataset_name}' (split={self.hf_split})"
        )
        try:
            dataset = await asyncio.get_running_loop().run_in_executor(
                None, self._load_hf_dataset
            )
        except Exception as e:
            raise DatasetLoaderError(
                f"Failed to load HuggingFace dataset '{self.hf_dataset_name}': {e}. "
                f"If the dataset is gated, authenticate with 'uv run hf auth login' "
                f"and accept the terms on the dataset's HuggingFace page."
            ) from e
        return {"dataset": dataset}

    def _load_hf_dataset(self) -> Any:
        return hf_load_dataset(
            self.hf_dataset_name,
            name=self.hf_subset,
            split=self.hf_split,
            trust_remote_code=False,
            streaming=self.streaming,
        )

    def _pil_to_image(self, pil_image: PILImage.Image) -> Image:
        """Convert a PIL Image to an AIPerf Image with a base64 JPEG data URL."""
        b64 = utils.encode_image(pil_image, "JPEG")
        return Image(name="", contents=[f"data:image/jpeg;base64,{b64}"])

    def _extract_images(self, row: dict[str, Any], image_column: str) -> list[Image]:
        """Extract images from a dataset row column.

        Accepts scalar or list-wrapped values; returns the first valid image as
        a single-element list, or ``[]`` if none. Handles HF-decoded PIL Images
        and undecoded ``{"bytes": ..., "path": ...}`` dicts (datasets declared
        with ``Image(decode=False)``, e.g. VisionArena, return raw byte dicts).

        Path-only dicts (``bytes is None``) aren't handled — VisionArena (the
        dataset that motivated this fix) embeds bytes inline; we log a debug
        message so an operator pointing aiperf at a path-only dataset can see
        why ``inputs.json`` is text-only. Both branches are wrapped in the same
        ``try`` so header-detection errors (``UnidentifiedImageError``) and
        load-time errors raised when ``_pil_to_image`` re-encodes (``OSError``
        from truncated payloads) skip the bad image instead of aborting the
        loader.
        """
        value = row.get(image_column)
        items = value if isinstance(value, list) else [value]
        for item in items:
            try:
                if isinstance(item, PILImage.Image):
                    return [self._pil_to_image(item)]
                if not isinstance(item, dict):
                    continue
                if item.get("bytes"):
                    pil = PILImage.open(io.BytesIO(item["bytes"]))
                    return [self._pil_to_image(pil)]
                if item.get("path"):
                    self.debug(
                        f"path-only HF image dict not supported: {item.get('path')}"
                    )
            except (OSError, PILImage.UnidentifiedImageError):
                continue
        return []

    def _extract_videos(self, row: dict[str, Any], video_column: str) -> list[Video]:
        """Extract videos from a dataset row column.

        Handles URL strings and dicts with raw bytes (HF video format).
        URL strings are passed through directly; bytes are base64-encoded.

        Scalar-only: if a future dataset declares ``Sequence(Video(decode=False))``,
        mirror ``_extract_images``' list-unwrap loop here to avoid the same
        silent-empty regression that motivated this file's image fix.
        """
        value = row.get(video_column)
        if isinstance(value, str) and value:
            # Pass through any valid URI scheme; only prepend file:// for bare paths
            url = (
                value
                if "://" in value or value.startswith("data:")
                else f"file://{value}"
            )
            return [Video(name="", contents=[url])]
        if isinstance(value, dict) and "bytes" in value and value["bytes"]:
            path = value.get("path", "")
            mime_type = mimetypes.guess_type(path)[0] if path else None
            mime_type = mime_type or "video/mp4"
            b64 = base64.b64encode(value["bytes"]).decode("utf-8")
            return [Video(name="", contents=[f"data:{mime_type};base64,{b64}"])]
        return []

    def _extract_audio(self, row: dict[str, Any], audio_column: str) -> list[Audio]:
        """Extract audio from a dataset row column.

        Handles HF Audio dicts with array/sampling_rate fields.
        Encodes the numpy array as WAV and returns base64 in the format
        expected by the chat endpoint: 'wav,<base64>'.
        """
        value = row.get(audio_column)
        if not isinstance(value, dict):
            return []
        array = value.get("array")
        sr = value.get("sampling_rate")
        if array is None or sr is None:
            return []
        try:
            buf = io.BytesIO()
            sf.write(buf, array, sr, format="WAV")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return [Audio(name="", contents=[f"wav,{b64}"])]
        except (OSError, ValueError, RuntimeError) as e:
            self.debug(
                lambda exc=e: f"Failed to encode WAV from column '{audio_column}': {exc.__class__.__name__}: {exc}"
            )
            return []

    def _max_conversations(self) -> int | None:
        """Return the maximum number of conversations to build from the dataset.

        Returns None for non-streaming datasets.

        For streaming datasets, caps at the largest profiling-phase request count
        when set, otherwise the active dataset's `entries`, to prevent fetching
        the entire remote dataset in duration-based benchmarks.
        """
        if not self.streaming:
            return None

        request_counts = [
            phase.requests
            for phase in self.run.cfg.get_profiling_phases()
            if getattr(phase, "requests", None) is not None
        ]
        if request_counts:
            return max(request_counts)

        dataset = self.run.cfg.get_default_dataset()
        return getattr(dataset, "entries", None)

    @abstractmethod
    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]: ...

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL
