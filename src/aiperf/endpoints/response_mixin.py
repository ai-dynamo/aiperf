# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jmespath

from aiperf.common.models import InferenceServerResponse, ParsedResponse


class JMESPathResponseMixin:
    """Mixin: JMESPath + auto-detect response parsing.

    Reads optional ``response_field`` from endpoint.extra to compile a JMESPath
    query used during response parsing.  Falls back to auto-detection when no
    query is configured or when the query fails to match.
    """

    def _init_response_parser(self) -> None:
        extra = self.model_endpoint.endpoint.extra
        extra_dict = dict(extra) if extra else {}
        response_field = extra_dict.get("response_field")
        self._compiled_jmespath = None
        if response_field:
            try:
                self._compiled_jmespath = jmespath.compile(response_field)
                self.info(f"Compiled JMESPath query: '{response_field}'")
            except (jmespath.exceptions.JMESPathError, TypeError) as e:
                self.error(
                    f"Failed to compile JMESPath query: '{response_field}' - {e!r}"
                )

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse response with auto-detection or custom JMESPath query.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with auto-detected type (text, embeddings, rankings)
        """
        json_obj = response.get_json()
        if not json_obj:
            if text := response.get_text():
                return ParsedResponse(
                    perf_ns=response.perf_ns, data=self.make_text_response_data(text)
                )
            return None

        response_data = None
        if self._compiled_jmespath:
            try:
                if value := self._compiled_jmespath.search(json_obj):
                    response_data = self.convert_to_response_data(value)
            except (jmespath.exceptions.JMESPathError, TypeError) as e:
                self.warning(f"JMESPath search failed: {e!r}. Trying auto-detection.")

        if not response_data:
            response_data = self.auto_detect_and_extract(json_obj)

        return (
            ParsedResponse(perf_ns=response.perf_ns, data=response_data)
            if response_data
            else None
        )
