# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.models import Media
from aiperf.common.types import MediaT
from aiperf.dataset.loader.models import CustomDatasetT


class MediaConversionMixin:
    """Mixin providing shared media conversion functionality for dataset loaders.
    It is used to construct text, image, and audio data from a CustomDatasetT object.
    """

    @property
    def _media_classes(self) -> list[type[MediaT]]:
        """Dynamically get all Media subclasses."""
        return Media.__subclasses__()

    def convert_to_media_objects(
        self, data: CustomDatasetT, name: str = ""
    ) -> dict[str, list[MediaT]]:
        """Convert all custom dataset into media objects.

        Args:
            data: The custom dataset to convert into media objects.
            name: The name of the media field.

        Returns:
            A dictionary of media objects.
        """
        media_objects: dict[str, list[MediaT]] = {}
        for media_class in self._media_classes:
            media_objects[media_class.media_type] = self._convert_to_media_objects(
                data,
                media_class=media_class,
                field=media_class.media_type,
                name=name,
            )
        return media_objects

    def _convert_to_media_objects(
        self,
        data: CustomDatasetT,
        media_class: type[MediaT],
        field: str,
        name: str = "",
    ) -> list[MediaT]:
        """Generic method to construct media objects from a CustomDatasetT object.

        Args:
            data: The custom dataset to construct media objects from.
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
        values = getattr(data, f"{field}s", None)
        if values is None:
            return []

        # If already correct media objects, return as is
        if all(isinstance(v, media_class) for v in values):
            return values

        return [media_class(name=name, contents=values)]
