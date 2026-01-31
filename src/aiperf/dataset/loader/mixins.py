# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from urllib.parse import urlparse

from aiperf.common.enums import AudioFormat, MediaType
from aiperf.common.models import Media
from aiperf.common.types import MediaT, MediaTypeT
from aiperf.dataset import utils
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
            media_class: The target media class (Text, Image, Audio, or Video).
            field: The name of the field (e.g., 'text', 'image', 'audio', 'video').
            name: The name of the media field.

        Returns:
            A list of media objects.
        """
        # Check singular field first
        value = getattr(data, field, None)
        if value is not None:
            # Handle media content (encode local files to base64)
            if field in [MediaType.IMAGE, MediaType.VIDEO, MediaType.AUDIO]:
                value = self._handle_media_content(value, media_type=MediaType(field))
            return [media_class(name=name, contents=[value])]

        # Check plural field
        values = getattr(data, f"{field}s", None)
        if values is None or not isinstance(values, Iterable):
            return []

        # If already correct media objects, return as is
        if all(isinstance(v, media_class) for v in values):
            return values

        # Handle media content (encode local files to base64)
        if field in [MediaType.IMAGE, MediaType.VIDEO, MediaType.AUDIO]:
            values = [
                self._handle_media_content(v, media_type=MediaType(field))
                for v in values
            ]

        return [media_class(name=name, contents=values)]

    def _is_url(self, content: str) -> bool:
        """Check if content is a valid URL with scheme and netloc.

        Args:
            content: The content to check.

        Returns:
            True if content is a URL, False otherwise.

        Raises:
            ValueError: If URL has only scheme or only netloc (invalid).
        """
        url = urlparse(content)

        # Valid URL with both scheme and netloc
        if url.scheme and url.netloc:
            return True

        # Invalid URL - has one but not both
        if url.scheme or url.netloc:
            raise ValueError(f"Valid URL must have both a scheme and netloc: {content}")

        # Not a URL
        return False

    def _is_already_encoded(self, content: str, media_type: MediaTypeT) -> bool:
        """Check if content is already encoded in the expected format.

        Args:
            content: The content to check.
            media_type: The media type (MediaType.IMAGE, MediaType.AUDIO, MediaType.VIDEO).

        Returns:
            True if content is already encoded, False otherwise.
        """
        url = urlparse(content)

        if media_type in [MediaType.IMAGE, MediaType.VIDEO]:
            # Check for data URL format
            return url.scheme == "data"

        elif media_type == MediaType.AUDIO:
            # Check for "format,base64" format
            if "," in content and not url.scheme:
                parts = content.split(",", 1)
                return len(parts) == 2 and parts[0].lower() in [
                    AudioFormat.WAV,
                    AudioFormat.MP3,
                ]
            return False

        return False

    def _encode_media_file(self, content: str, media_type: MediaTypeT) -> str:
        """Encode a local media file to base64.

        Args:
            content: The file path to encode.
            media_type: The media type (MediaType.IMAGE, MediaType.AUDIO, MediaType.VIDEO).

        Returns:
            The base64-encoded content in the appropriate format.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            RuntimeError: If the format is unsupported.
        """
        match media_type:
            case MediaType.IMAGE:
                img = utils.open_image(content)
                img_base64 = utils.encode_image(img, img.format)
                return f"data:image/{img.format.lower()};base64,{img_base64}"
            case MediaType.AUDIO:
                audio_bytes, audio_format = utils.open_audio(content)
                return utils.encode_audio(audio_bytes, audio_format)
            case MediaType.VIDEO:
                video_bytes, video_format = utils.open_video(content)
                return utils.encode_video(video_bytes, video_format)
            case _:
                raise ValueError(f"Unsupported media type: {media_type}")

    def _handle_media_content(self, content: str, media_type: MediaTypeT) -> str:
        """Generic handler for media content encoding.

        If the content is a URL, it's returned as-is.
        If it's already encoded, it's returned as-is.
        If it's a local file path, it's loaded and encoded to base64.

        Args:
            content: The media content - URL, encoded string, or local file path.
            media_type: The media type (MediaType.IMAGE, MediaType.AUDIO, MediaType.VIDEO).

        Returns:
            The processed media content.

        Raises:
            FileNotFoundError: If the local file doesn't exist.
            RuntimeError: If the media format is unsupported.
            ValueError: If URL format is invalid.
        """
        # Check if it's already encoded first (before URL check)
        # This handles data URLs which have a scheme but no netloc
        if self._is_already_encoded(content, media_type):
            return content

        # Check if it's a URL - These are passed through directly to the server
        if self._is_url(content):
            return content

        # Otherwise, it's a local file path - encode it
        return self._encode_media_file(content, media_type)
