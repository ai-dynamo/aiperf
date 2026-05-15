# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ImageEditEndpoint."""

import base64

import orjson
import pytest
from pydantic import TypeAdapter

from aiperf.common.models import Image, Text, Turn
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.endpoints.openai_image_edit import (
    ImageEditEndpoint,
    _sniff_mime_from_b64_prefix,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)

# Tiny 1x1 PNG used as a stand-in for a real reference image.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08"
    b"\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfc\xcf\xc0\x00"
    b"\x00\x00\x03\x00\x01\x9a\xa3\x9eS\x00\x00\x00\x00IEND\xaeB`\x82"
)
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")
_PNG_DATA_URL = f"data:image/png;base64,{_PNG_B64}"


class TestImageEditEndpoint:
    """Tests for ImageEditEndpoint format_payload + parse_response."""

    @pytest.fixture
    def model_endpoint(self) -> ModelEndpointInfo:
        """ModelEndpointInfo configured for IMAGE_EDIT against FLUX.2-Klein-4B."""
        return create_model_endpoint(
            EndpointType.IMAGE_EDIT, model_name="black-forest-labs/FLUX.2-klein-4B"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint: ModelEndpointInfo) -> ImageEditEndpoint:
        """ImageEditEndpoint instance backed by a mock transport."""
        return create_endpoint_with_mock_transport(ImageEditEndpoint, model_endpoint)

    # ===== format_payload =====

    def test_format_payload_with_data_url_image(self, endpoint, model_endpoint) -> None:
        """Data URL image content keeps the base64 string in the payload."""
        turn = Turn(
            texts=[Text(contents=["Make the cat blue"])],
            images=[Image(contents=[_PNG_DATA_URL])],
            model="black-forest-labs/FLUX.2-klein-4B",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["prompt"] == "Make the cat blue"
        assert payload["model"] == "black-forest-labs/FLUX.2-klein-4B"
        assert payload["response_format"] == "b64_json"
        assert payload["n"] == 1
        image_field = payload["image"]
        assert image_field["b64_data"] == _PNG_B64
        assert image_field["filename"].endswith(".png")
        assert image_field["content_type"] == "image/png"
        assert "url" not in payload

    def test_format_payload_payload_is_json_serialisable(
        self, endpoint, model_endpoint
    ) -> None:
        """Regression: payload must remain JSON-serialisable.

        Raw bytes here break DatasetManager._generate_inputs_json_file
        (pydantic model_dump(mode='json') + orjson.dumps).
        """
        turn = Turn(
            texts=[Text(contents=["edit"])],
            images=[Image(contents=[_PNG_DATA_URL])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        TypeAdapter(dict).dump_python(payload, mode="json")
        orjson.dumps(payload)

    def test_format_payload_with_raw_base64_image(
        self, endpoint, model_endpoint
    ) -> None:
        """A raw base64 string (no data URL prefix) sniffs MIME from magic bytes."""
        turn = Turn(
            texts=[Text(contents=["Edit prompt"])],
            images=[Image(contents=[_PNG_B64])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["image"]["b64_data"] == _PNG_B64
        assert payload["image"]["content_type"] == "image/png"

    def test_format_payload_raw_base64_jpeg_sniffed_from_magic_bytes(
        self, endpoint, model_endpoint
    ) -> None:
        """JPEG content without data URL prefix is detected via FFD8FF magic bytes."""
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        turn = Turn(
            texts=[Text(contents=["edit"])],
            images=[Image(contents=[b64])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["image"]["content_type"] == "image/jpeg"
        assert payload["image"]["filename"].endswith(".jpg")

    def test_format_payload_with_http_url_image(self, endpoint, model_endpoint) -> None:
        """An http(s) URL is passed through as the multipart `url` field, no decode."""
        turn = Turn(
            texts=[Text(contents=["edit"])],
            images=[Image(contents=["https://example.com/source.png"])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["url"] == "https://example.com/source.png"
        assert "image" not in payload

    @pytest.mark.parametrize(
        "url",
        [
            "HTTPS://example.com/img.png",
            "HTTP://example.com/img.png",
            "Https://Example.COM/img.png",
        ],
    )
    def test_format_payload_url_scheme_is_case_insensitive(
        self, endpoint, model_endpoint, url: str
    ) -> None:
        """RFC-legal uppercase/mixed-case schemes still route to the `url` field."""
        turn = Turn(
            texts=[Text(contents=["edit"])],
            images=[Image(contents=[url])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["url"] == url
        assert "image" not in payload

    def test_format_payload_jpeg_data_url_uses_jpg_filename(
        self, endpoint, model_endpoint
    ) -> None:
        """JPEG data URLs round-trip with a `.jpg` filename and image/jpeg type."""
        b64 = base64.b64encode(b"\xff\xd8\xff\xd9").decode("ascii")
        turn = Turn(
            texts=[Text(contents=["edit"])],
            images=[Image(contents=[f"data:image/jpeg;base64,{b64}"])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["image"]["filename"].endswith(".jpg")
        assert payload["image"]["content_type"] == "image/jpeg"
        assert payload["image"]["b64_data"] == b64

    def test_format_payload_extra_inputs_merged_after_image(self, endpoint) -> None:
        """Extra inputs (size, num_inference_steps, guidance_scale, ...) are merged in."""
        model_endpoint = create_model_endpoint(
            EndpointType.IMAGE_EDIT,
            model_name="black-forest-labs/FLUX.2-klein-4B",
            extra=[
                ("size", "1024x1024"),
                ("num_inference_steps", 28),
                ("guidance_scale", 4.0),
                ("seed", 42),
            ],
        )
        ep = create_endpoint_with_mock_transport(ImageEditEndpoint, model_endpoint)
        turn = Turn(
            texts=[Text(contents=["edit"])],
            images=[Image(contents=[_PNG_DATA_URL])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = ep.format_payload(request_info)

        assert payload["size"] == "1024x1024"
        assert payload["num_inference_steps"] == 28
        assert payload["guidance_scale"] == 4.0
        assert payload["seed"] == 42

    def test_format_payload_extra_inputs_cannot_overwrite_reserved(self) -> None:
        """Reserved keys (prompt, image, url, mask) are protected from --extra-inputs."""
        model_endpoint = create_model_endpoint(
            EndpointType.IMAGE_EDIT,
            model_name="black-forest-labs/FLUX.2-klein-4B",
            extra=[
                ("prompt", "HIJACKED"),
                ("image", "not-a-file"),
                ("url", "https://malicious.example"),
                ("mask", "/tmp/mask.png"),
                ("size", "512x512"),
            ],
        )
        ep = create_endpoint_with_mock_transport(ImageEditEndpoint, model_endpoint)
        turn = Turn(
            texts=[Text(contents=["legitimate prompt"])],
            images=[Image(contents=[_PNG_DATA_URL])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = ep.format_payload(request_info)

        assert payload["prompt"] == "legitimate prompt"
        assert payload["image"]["b64_data"] == _PNG_B64
        assert "url" not in payload
        assert "mask" not in payload
        assert payload["size"] == "512x512"

    def test_format_payload_extra_body_merged_after_endpoint_extra(self) -> None:
        model_endpoint = create_model_endpoint(
            EndpointType.IMAGE_EDIT,
            model_name="black-forest-labs/FLUX.2-klein-4B",
            extra=[("size", "512x512"), ("seed", 1)],
        )
        ep = create_endpoint_with_mock_transport(ImageEditEndpoint, model_endpoint)
        turn = Turn(
            texts=[Text(contents=["edit"])],
            images=[Image(contents=[_PNG_DATA_URL])],
            extra_body={"seed": 2, "guidance_scale": 4.0},
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = ep.format_payload(request_info)

        assert payload["size"] == "512x512"
        assert payload["seed"] == 2
        assert payload["guidance_scale"] == 4.0

    def test_format_payload_uses_dispatching_turn(
        self, endpoint, model_endpoint
    ) -> None:
        parent_turn = Turn(
            texts=[Text(contents=["parent edit"])],
            images=[Image(contents=[_PNG_DATA_URL])],
            model="parent-model",
            extra_body={"seed": 1},
        )
        child_turn = Turn(
            texts=[Text(contents=["child edit"])],
            images=[Image(contents=["https://example.com/child.png"])],
            model="child-model",
            extra_body={"seed": 2},
        )
        request_info = create_request_info(
            model_endpoint=model_endpoint, turns=[parent_turn, child_turn]
        )

        payload = endpoint.format_payload(request_info)

        assert payload["prompt"] == "child edit"
        assert payload["model"] == "child-model"
        assert payload["url"] == "https://example.com/child.png"
        assert "image" not in payload
        assert payload["seed"] == 2

    def test_format_payload_non_image_data_url_mime_ignored(
        self, endpoint, model_endpoint
    ) -> None:
        """Data URLs that claim a non-image MIME type fall back to magic-byte sniff."""
        b64 = base64.b64encode(_PNG_BYTES).decode("ascii")
        turn = Turn(
            texts=[Text(contents=["edit"])],
            images=[Image(contents=[f"data:text/html;base64,{b64}"])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        # Magic bytes (PNG signature) should win over the bogus header.
        assert payload["image"]["content_type"] == "image/png"
        assert payload["image"]["filename"].endswith(".png")

    def test_format_payload_data_url_without_mime_metadata_sniffs(
        self, endpoint, model_endpoint
    ) -> None:
        """A bare ``data:,<b64>`` URL (no MIME / no semicolon) falls back to magic-byte sniff."""
        b64 = base64.b64encode(_PNG_BYTES).decode("ascii")
        turn = Turn(
            texts=[Text(contents=["edit"])],
            images=[Image(contents=[f"data:,{b64}"])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["image"]["content_type"] == "image/png"
        assert payload["image"]["b64_data"] == b64

    def test_format_payload_svg_xml_data_url_strips_subtype_suffix(
        self, endpoint, model_endpoint
    ) -> None:
        """`image/svg+xml` -> filename `image.svg` (not `image.svg+xml`)."""
        b64 = base64.b64encode(b"<svg/>").decode("ascii")
        turn = Turn(
            texts=[Text(contents=["edit"])],
            images=[Image(contents=[f"data:image/svg+xml;base64,{b64}"])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["image"]["filename"] == "image.svg"
        # Content-Type retains the full `image/svg+xml` MIME — only the filename strips.
        assert payload["image"]["content_type"] == "image/svg+xml"

    def test_format_payload_no_turns_raises(self, endpoint, model_endpoint) -> None:
        """Missing turns raises a clear ValueError instead of indexing into an empty list."""
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])
        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_no_text_raises(self, endpoint, model_endpoint) -> None:
        """Turn without a text prompt is rejected up front."""
        turn = Turn(texts=[], images=[Image(contents=[_PNG_DATA_URL])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        with pytest.raises(ValueError, match="requires a text prompt"):
            endpoint.format_payload(request_info)

    def test_format_payload_no_image_raises(self, endpoint, model_endpoint) -> None:
        """Turn without a reference image is rejected up front."""
        turn = Turn(texts=[Text(contents=["edit"])], images=[])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        with pytest.raises(ValueError, match="requires a reference image"):
            endpoint.format_payload(request_info)

    def test_format_payload_empty_image_content_raises(
        self, endpoint, model_endpoint
    ) -> None:
        """Empty string in turn.images[0].contents[0] raises a clear error."""
        turn = Turn(
            texts=[Text(contents=["edit"])],
            images=[Image(contents=[""])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        with pytest.raises(ValueError, match="content is empty"):
            endpoint.format_payload(request_info)

    def test_format_payload_invalid_base64_raises(
        self, endpoint, model_endpoint
    ) -> None:
        """Image content not matching any known b64 prefix raises a clear error."""
        turn = Turn(
            texts=[Text(contents=["edit"])],
            images=[Image(contents=["not!base64!data"])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        with pytest.raises(ValueError, match="not a recognized image format"):
            endpoint.format_payload(request_info)

    def test_format_payload_malformed_data_url_raises(
        self, endpoint, model_endpoint
    ) -> None:
        """Data URL without a comma separator is rejected before b64 decode."""
        turn = Turn(
            texts=[Text(contents=["edit"])],
            images=[Image(contents=["data:image/png;base64"])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        with pytest.raises(ValueError, match="Malformed data URL"):
            endpoint.format_payload(request_info)

    # ===== parse_response =====

    @pytest.mark.parametrize(
        "json_data,expected_b64,expected_url",
        [
            pytest.param(
                {"data": [{"b64_json": "aGVsbG8="}]},
                "aGVsbG8=",
                None,
                id="b64_only",
            ),
            pytest.param(
                {"data": [{"url": "https://cdn/edit.png"}]},
                None,
                "https://cdn/edit.png",
                id="url_only",
            ),
            pytest.param(
                {"data": [{"b64_json": "aGVsbG8=", "revised_prompt": "X"}]},
                "aGVsbG8=",
                None,
                id="with_revised_prompt",
            ),
        ],
    )  # fmt: skip
    def test_parse_response_image_formats(
        self, endpoint, json_data, expected_b64, expected_url
    ) -> None:
        """parse_response handles b64_json, url, and revised_prompt entries."""
        mock_response = create_mock_response(json_data=json_data)
        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.images) == 1
        assert parsed.data.images[0].b64_json == expected_b64
        assert parsed.data.images[0].url == expected_url

    def test_parse_response_metadata_fields(self, endpoint) -> None:
        """Top-level size/quality/output_format/background flow through to ImageResponseData."""
        json_data = {
            "data": [{"b64_json": "img"}],
            "size": "1024x1024",
            "quality": "hd",
            "output_format": "png",
            "background": "transparent",
        }
        mock_response = create_mock_response(json_data=json_data)
        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.size == "1024x1024"
        assert parsed.data.quality == "hd"
        assert parsed.data.output_format == "png"
        assert parsed.data.background == "transparent"

    def test_parse_response_no_json_returns_none(self, endpoint) -> None:
        """Non-JSON response bodies return None instead of raising."""
        mock_response = create_mock_response(json_data=None)
        mock_response.get_raw.return_value = ""
        assert endpoint.parse_response(mock_response) is None

    def test_parse_response_empty_data_returns_no_images(self, endpoint) -> None:
        """An empty `data` array produces a parsed response with zero images."""
        mock_response = create_mock_response(json_data={"data": []})
        parsed = endpoint.parse_response(mock_response)
        assert parsed is not None
        assert parsed.data.images == []

    def test_parse_response_perf_ns_preserved(self, endpoint) -> None:
        """perf_ns from the raw response is preserved on the ParsedResponse."""
        mock_response = create_mock_response(
            perf_ns=999_888_777, json_data={"data": [{"b64_json": "x"}]}
        )
        parsed = endpoint.parse_response(mock_response)
        assert parsed is not None
        assert parsed.perf_ns == 999_888_777


class TestSniffMimeFromB64Prefix:
    """Covers each branch of _sniff_mime_from_b64_prefix."""

    @pytest.mark.parametrize(
        "magic_bytes,expected",
        [
            pytest.param(b"\x89PNG\r\n\x1a\n", "image/png", id="png"),
            pytest.param(b"\xff\xd8\xff\xe0", "image/jpeg", id="jpeg"),
            pytest.param(b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp", id="webp"),
            pytest.param(b"GIF87a", "image/gif", id="gif87a"),
            pytest.param(b"GIF89a", "image/gif", id="gif89a"),
            pytest.param(b"BM\x00\x00", "image/bmp", id="bmp"),
        ],
    )  # fmt: skip
    def test_b64_prefix_dispatch(self, magic_bytes: bytes, expected: str) -> None:
        """Each supported image format dispatches via its base64 prefix."""
        b64 = base64.b64encode(magic_bytes).decode("ascii")
        assert _sniff_mime_from_b64_prefix(b64) == expected

    def test_unknown_prefix_returns_none(self) -> None:
        """Non-image b64 content (or non-b64 garbage) returns None."""
        # "garbage" b64-encoded -> "Z2FyYmFnZQ==" — doesn't match any prefix.
        assert _sniff_mime_from_b64_prefix("Z2FyYmFnZQ==") is None
        # Pure non-base64 garbage also returns None (no decode is attempted).
        assert _sniff_mime_from_b64_prefix("not!base64!data") is None
