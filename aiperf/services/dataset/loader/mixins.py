# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar

from aiperf.common.models import Audio, Image, Text
from aiperf.services.dataset.loader.models import CustomDatasetT

MediaTypeT = TypeVar("MediaTypeT", bound=Text | Image | Audio)


class MediaConversionMixin:
    """Mixin providing shared media conversion functionality for dataset loaders.
    It is used to construct text, image, and audio data from a CustomDatasetT object.
    """

    def convert_all_media_data(
        self, data: CustomDatasetT, name: str | None = None
    ) -> tuple[MediaTypeT, ...]:
        """Convert all media data from a CustomDatasetT object.

        Args:
            data: The data to convert media data from.

        Returns:
            A tuple of media objects.
        """
        return (
            self._convert_to_media_objects(data, Text, "text", name),
            self._convert_to_media_objects(data, Image, "image", name),
            self._convert_to_media_objects(data, Audio, "audio", name),
        )

    def _convert_to_media_objects(
        self,
        data: CustomDatasetT,
        media_class: type[MediaTypeT],
        field: str,
        name: str | None = None,
    ) -> list[MediaTypeT]:
        """Generic method to construct media data from a CustomDatasetT object.

        Args:
            data: The data to construct media data from.
            media_class: The target media class (Text, Image, or Audio).
            field: The name of the field (e.g., 'text', 'image', 'audio').
            name: The name of the media field.

        Returns:
            A list of media objects.
        """
        # Check singular field first
        value = getattr(data, field, None)
        if value is not None:
            return [media_class(name=name, contents=[value])]

        # Check plural field
        value = getattr(data, f"{field}s", None)
        if not value:
            return []

        # If already correct media objects, return as is
        if all(isinstance(v, media_class) for v in value):
            return value  # type: ignore

        return [media_class(name=name, contents=value)]
