# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

# Keys protected from --extra-inputs (turn-derived or binary uploads).
_RESERVED_PAYLOAD_KEYS: frozenset[str] = frozenset({"prompt", "image", "url", "mask"})

# Base64-encoded magic-byte prefixes. PNG bytes `\x89PNG\r\n\x1a\n` -> `iVBORw0KGgo`,
# JPEG `\xff\xd8\xff` -> `/9j/`, GIF87a/89a -> `R0lGODdh`/`R0lGODlh`, RIFF -> `UklGR`
# (reported as WebP — image_edit inputs are image-only), BMP `BM` -> `Qk`.
_MIME_BY_B64_PREFIX: tuple[tuple[str, str], ...] = (
    ("iVBORw0KGgo", "image/png"),
    ("/9j/", "image/jpeg"),
    ("R0lGODlh", "image/gif"),
    ("R0lGODdh", "image/gif"),
    ("UklGR", "image/webp"),
    ("Qk", "image/bmp"),
)


def _sniff_mime_from_b64_prefix(b64: str) -> str | None:
    """Return the MIME type implied by a base64-encoded image's leading characters."""
    for prefix, mime in _MIME_BY_B64_PREFIX:
        if b64.startswith(prefix):
            return mime
    return None


class ImageEditEndpoint(BaseEndpoint):
    """OpenAI Image Edit (image-to-image) endpoint.

    Multipart upload of a reference image plus text prompt. Compatible with
    SGLang's /v1/images/edits and FLUX.2 unified diffusion serving. The
    ``url`` form-field path (HTTP/HTTPS reference image) is an SGLang
    extension; stock OpenAI ``/v1/images/edits`` only accepts file uploads.
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

        turn = request_info.turns[-1]
        model_endpoint = request_info.model_endpoint

        if not turn.texts or not turn.texts[0].contents:
            raise ValueError("Image edit endpoint requires a text prompt.")
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

        if image_content.lower().startswith(("http://", "https://")):
            payload["url"] = image_content
        else:
            payload["image"] = self._build_image_field(image_content)

        self._merge_with_reserved_filter(
            payload, model_endpoint.endpoint.extra or [], "--extra-inputs"
        )
        self._merge_with_reserved_filter(
            payload, (turn.extra_body or {}).items(), "extra_body"
        )

        self.trace(lambda: f"Formatted image edit payload keys: {list(payload)}")
        return payload

    def _merge_with_reserved_filter(
        self,
        payload: dict[str, Any],
        items: Any,
        label: str,
    ) -> None:
        """Merge ``items`` into ``payload`` while skipping reserved multipart keys.

        Reserved keys (``prompt``, ``image``, ``url``, ``mask``) are endpoint-
        managed; collisions log a warning and are dropped. ``label`` identifies
        the source for the warning text.
        """
        for key, value in items:
            if key in _RESERVED_PAYLOAD_KEYS:
                self.warning(
                    f"{label} {key!r} is managed by the endpoint and was ignored."
                )
                continue
            payload[key] = value

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
            if ";" in header:
                candidate = header.removeprefix("data:").split(";", 1)[0]
                # Only honor `image/*` MIME claims; a malformed
                # `data:text/html;base64,...` falls back to sniffing.
                if candidate.startswith("image/"):
                    explicit_mime = candidate

        mime = explicit_mime or _sniff_mime_from_b64_prefix(b64)
        if mime is None:
            raise ValueError(
                "Image content is not a recognized image format; expected a "
                "data URL or raw base64 image (PNG/JPEG/WebP/GIF/BMP)."
            )

        # Strip subtype suffix (e.g., `svg+xml` -> `svg`) so the filename is clean.
        ext = mime.split("/", 1)[1].split("+", 1)[0] if "/" in mime else "png"
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
