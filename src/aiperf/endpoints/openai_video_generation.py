# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from aiperf.common.models import (
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
    VideoResponseData,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint


class VideoGenerationEndpoint(BaseEndpoint):
    """OpenAI/SGLang Video Generation endpoint.

    Supports text-to-video generation with async job polling.
    Compatible with SGLang's /v1/videos endpoint and OpenAI's video API.

    The video generation API follows an async job pattern:
    1. POST /v1/videos - Submit job, get job ID with status "queued"
    2. GET /v1/videos/{id} - Poll until status is "completed" or "failed"
    3. GET /v1/videos/{id}/content - Download the generated video

    See: https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/openai_api.md
    See: https://platform.openai.com/docs/api-reference/videos
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format OpenAI/SGLang Video Generation request payload from RequestInfo.

        Supports all common video generation API parameters:
        - prompt (required): Text description from turn.texts[0]
        - model (optional): From turn.model or model_endpoint.primary_model_name

        Additional parameters via --extra-inputs:
        - size: Video resolution (e.g., "1280x720", "720x1280")
        - seconds: Video duration (e.g., 4, 8, 12)
        - seed: Random seed for reproducibility
        - num_inference_steps: Diffusion denoising steps
        - guidance_scale: Classifier-free guidance scale
        - negative_prompt: Concepts to exclude
        - fps: Frames per second
        - num_frames: Total frames to generate

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            Video generation API payload with all specified parameters
        """
        if not request_info.turns:
            raise ValueError("Video generation endpoint requires at least one turn.")

        turn = request_info.turns[0]
        model_endpoint = request_info.model_endpoint

        if not turn.texts or not turn.texts[0].contents:
            raise ValueError(
                "Video generation endpoint requires text prompt in first turn."
            )

        prompt = turn.texts[0].contents[0]

        payload = {
            "prompt": prompt,
            "model": turn.model or model_endpoint.primary_model_name,
        }

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse OpenAI/SGLang Video Generation response.

        Handles the VideoResponse format returned by both initial job submission
        and status polling requests. The response contains job metadata including
        status, progress, and video URL when completed.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted video job data
        """
        json_obj = response.get_json()
        if not json_obj:
            self.debug(
                lambda: f"No JSON object found in response: {response.get_raw()}"
            )
            return None

        # Parse SGLang/OpenAI VideoResponse format
        video_data = VideoResponseData(
            video_id=json_obj.get("id"),
            object=json_obj.get("object"),
            status=json_obj.get("status"),
            progress=json_obj.get("progress"),
            url=json_obj.get("url"),
            size=json_obj.get("size"),
            seconds=json_obj.get("seconds"),
            quality=json_obj.get("quality"),
            model=json_obj.get("model"),
            created_at=json_obj.get("created_at"),
            completed_at=json_obj.get("completed_at"),
            expires_at=json_obj.get("expires_at"),
            inference_time_s=json_obj.get("inference_time_s"),
            peak_memory_mb=json_obj.get("peak_memory_mb"),
            error=json_obj.get("error"),
        )

        usage = json_obj.get("usage") or None

        return ParsedResponse(perf_ns=response.perf_ns, data=video_data, usage=usage)
