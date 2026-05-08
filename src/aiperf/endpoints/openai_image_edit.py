# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import binascii
from typing import Any

from aiperf.common.models import (
    ImageDataItem,
    ImageResponseData,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint

_MIME_BY_EXT: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
}

# Keys derived from turn data that --extra-inputs must not overwrite.
_RESERVED_PAYLOAD_KEYS: frozenset[str] = frozenset({"prompt", "image", "url"})


def _sniff_mime_from_magic_bytes(image_bytes: bytes) -> str | None:
    """Detect MIME type from the magic bytes of a raw base64-decoded image."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    return None


class ImageEditEndpoint(BaseEndpoint):
    """OpenAI Image Edit (image-to-image) endpoint.

    Multipart upload of a reference image plus text prompt. Compatible with
    SGLang's /v1/images/edits and FLUX.2 unified diffusion serving.

    See: https://platform.openai.com/docs/api-reference/images/createEdit
    See: https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format an OpenAI Image Edit multipart payload from RequestInfo.

        The image stays base64-encoded in the payload so it survives
        ``model_dump(mode="json")`` upstream; the transport layer decodes
        it just before emitting multipart bytes.

        ``mask`` is not supported via ``--extra-inputs`` because the server
        expects it as a binary file upload, not a Form string.
        """
        if not request_info.turns:
            raise ValueError("Image edit endpoint requires at least one turn.")

        turn = request_info.turns[0]
        model_endpoint = request_info.model_endpoint

        if not turn.texts or not turn.texts[0].contents:
            raise ValueError("Image edit endpoint requires text prompt in first turn.")
        if not turn.images or not turn.images[0].contents:
            raise ValueError(
                "Image edit endpoint requires a reference image in turn.images[0]."
            )

        prompt = turn.texts[0].contents[0]
        image_content = turn.images[0].contents[0]
        if not image_content:
            raise ValueError("Reference image content is empty.")

        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": turn.model or model_endpoint.primary_model_name,
            "response_format": "b64_json",
            "n": 1,
        }

        if image_content.startswith(("http://", "https://")):
            payload["url"] = image_content
        else:
            payload["image"] = self._build_image_field(image_content)

        for key, value in model_endpoint.endpoint.extra or []:
            if key in _RESERVED_PAYLOAD_KEYS:
                self.warning(
                    f"--extra-inputs {key!r} is managed by the endpoint and was ignored."
                )
                continue
            payload[key] = value

        self.trace(lambda: f"Formatted image edit payload keys: {list(payload)}")
        return payload

    @staticmethod
    def _build_image_field(content: str) -> dict[str, Any]:
        """Decode a data URL or raw base64 string into a multipart file descriptor."""
        b64 = content
        explicit_mime: str | None = None
        if content.startswith("data:"):
            try:
                header, b64 = content.split(",", 1)
            except ValueError as exc:
                raise ValueError(
                    "Malformed data URL for image content (missing comma)."
                ) from exc
            if header.startswith("data:") and ";" in header:
                explicit_mime = header.removeprefix("data:").split(";", 1)[0] or None

        try:
            image_bytes = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                "Image content is not valid base64; expected a data URL or "
                "raw base64 image (PNG/JPEG/WebP/GIF/BMP)."
            ) from exc

        mime = explicit_mime or _sniff_mime_from_magic_bytes(image_bytes) or "image/png"
        ext = mime.split("/", 1)[1] if "/" in mime else "png"
        filename_ext = "jpg" if ext == "jpeg" else ext
        return {
            "b64_data": b64,
            "filename": f"image.{filename_ext}",
            "content_type": _MIME_BY_EXT.get(ext, mime),
        }

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse an OpenAI Image Edit response."""
        json_obj = response.get_json()
        if not json_obj:
            self.debug(
                lambda: f"No JSON object found in response: {response.get_raw()}"
            )
            return None

        images: list[ImageDataItem] = []
        for item in json_obj.get("data", []) or []:
            images.append(
                ImageDataItem(
                    url=item.get("url"),
                    b64_json=item.get("b64_json"),
                    revised_prompt=item.get("revised_prompt"),
                )
            )

        response_data = ImageResponseData(
            images=images,
            size=json_obj.get("size"),
            quality=json_obj.get("quality"),
            output_format=json_obj.get("output_format"),
            background=json_obj.get("background"),
        )

        usage = json_obj.get("usage") or None
        return ParsedResponse(perf_ns=response.perf_ns, data=response_data, usage=usage)
