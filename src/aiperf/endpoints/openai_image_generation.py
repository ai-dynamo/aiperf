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


class ImageGenerationEndpoint(BaseEndpoint):
    """OpenAI Image Generation endpoint.

    Supports image generation from text prompts using models like DALL-E.
    Handles both streaming and non-streaming responses.

    See: https://platform.openai.com/docs/api-reference/images/create
    See: https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cli.md
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format OpenAI Image Generation request payload from RequestInfo.

        Supports all OpenAI Image Generation API parameters:
        - prompt (required): Text description from turn.texts[0]
        - model (optional): From turn.model or model_endpoint.primary_model_name
        - stream (optional): From model_endpoint.endpoint.streaming
        - n, size, quality, style, response_format, background, moderation,
          output_format, output_compression, partial_images, user:
          Pass via --extra-inputs "input_name:value"

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            OpenAI Image Generation API payload with all specified parameters
        """
        if not request_info.turns:
            raise ValueError("Image generation endpoint requires at least one turn.")

        turn = request_info.turns[0]
        model_endpoint = request_info.model_endpoint

        if not turn.texts or not turn.texts[0].contents:
            raise ValueError(
                "Image generation endpoint requires text prompt in first turn."
            )

        prompt = turn.texts[0].contents[0]

        # NOTE: response_format is set to b64_json by default, but can be overridden by --extra-inputs response_format:<format>
        #       This is because sglang generate only supports b64_json responses currently.
        payload = {
            "prompt": prompt,
            "model": turn.model or model_endpoint.primary_model_name,
            "response_format": "b64_json",
            "n": 1,
        }

        if model_endpoint.endpoint.streaming:
            payload["stream"] = True

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse OpenAI Image Generation response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted image data and usage info
        """
        json_obj = response.get_json()
        if not json_obj:
            self.debug(
                lambda: f"No JSON object found in response: {response.get_raw()}"
            )
            return None

        images = []

        if "b64_json" in json_obj:
            # Streaming responses contain b64_json directly in the response
            images.append(
                ImageDataItem(
                    b64_json=json_obj.get("b64_json"),
                    partial_image_index=json_obj.get("partial_image_index"),
                )
            )
        elif "data" in json_obj:
            # Non-streaming responses contain data array with image items
            for item in json_obj.get("data", []):
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
