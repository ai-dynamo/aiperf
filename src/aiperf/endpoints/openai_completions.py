# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.models import (
    BaseResponseData,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.common.types import JsonObject, RequestOutputT
from aiperf.endpoints.base_endpoint import BaseEndpoint


class CompletionsEndpoint(BaseEndpoint):
    """OpenAI Completions endpoint.

    Supports text completions with streaming.
    """

    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format payload for a completions request.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            OpenAI Completions API payload
        """
        if len(request_info.turns) != 1:
            raise ValueError("Completions endpoint only supports one turn.")

        turn = request_info.turns[0]
        model_endpoint = request_info.model_endpoint

        prompts = [
            content for text in turn.texts for content in text.contents if content
        ]

        extra = model_endpoint.endpoint.extra or []

        payload = {
            # A single prompt goes on the wire as a bare string (the canonical
            # OpenAI form); some gateways reject the list[str] wrapping.
            "prompt": prompts[0] if len(prompts) == 1 else prompts,
            "model": turn.model or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if turn.max_tokens:
            payload["max_tokens"] = turn.max_tokens

        if extra:
            payload.update(extra)

        if turn.extra_body:
            payload.update(turn.extra_body)

        # Read the merged payload, not endpoint.streaming: the extras above can
        # override "stream", and a server rejects stream_options when stream is
        # false ("Stream options can only be defined when stream=True").
        if payload.get("stream"):
            # Requested for every streaming run, not just server-token-count
            # ones: vLLM rides per-request metrics (including
            # metrics.speculative_decoding) on the trailing usage chunk and
            # only emits that chunk when include_usage is set, so gating it on
            # an unrelated flag would silently drop those metrics. Authors who
            # want it off can set stream_options.include_usage explicitly.
            stream_options = payload.get("stream_options")
            # An explicit null parses straight from the CLI
            # (--extra-inputs '{"stream_options": null}'); treat it as absent so
            # this endpoint agrees with chat instead of silently skipping.
            if stream_options is None:
                stream_options = {}
            if isinstance(stream_options, dict):
                # Copy rather than mutate: the payload merge aliases
                # endpoint.extra / turn.extra_body, which are long-lived config
                # reused across every request, so an in-place edit would rewrite
                # the author's config and leak into every subsequent request.
                merged = {**stream_options}
                merged.setdefault("include_usage", True)
                payload["stream_options"] = merged

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse OpenAI Completions response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text content and usage data
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        data = self.extract_completions_response_data(json_obj)
        usage = json_obj.get("usage") or None
        spec_decode_stats = self.extract_spec_decode_stats(json_obj)

        if data or usage or spec_decode_stats:
            return ParsedResponse(
                perf_ns=response.perf_ns,
                data=data,
                usage=usage,
                spec_decode_stats=spec_decode_stats,
            )

        return None

    def extract_completions_response_data(
        self, json_obj: JsonObject
    ) -> BaseResponseData | None:
        """Extract content from OpenAI Completions JSON response.

        Handles both text_completion and completion object types.

        Args:
            json_obj: Deserialized OpenAI response

        Returns:
            Extracted text data or None if no content
        """
        match json_obj.get("object"):
            case "completion" | "text_completion":
                choices = json_obj.get("choices")
                if not choices:
                    self.debug(lambda: f"No choices found in response: {json_obj}")
                    return None
                return self.make_text_response_data(choices[0].get("text"))
            case _:
                # Unrecognized object: the server can return arbitrary bodies
                # (error JSON, proxy pages, truncated streams on crash). Degrade
                # to None like the no-choices case above rather than raising, so
                # the worker records a failure and keeps going.
                return None
