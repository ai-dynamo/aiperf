# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jmespath

from aiperf.common.models import InferenceServerResponse, ParsedResponse


class JMESPathResponseMixin:
    """Response-parsing mixin: JMESPath + auto-detect.

    Reads ``endpoint.extra.response_field`` once at init, compiles it as a
    JMESPath query (https://jmespath.org), and uses the query as a first-pass
    extractor in ``parse_response``. Falls back to auto-detect (embeddings,
    then rankings, then text) when the query is absent, mismatched, or raises.

    Wiring contract -- required for the mixin to function:

    1. Mix it in to the LEFT of ``BaseEndpoint`` so this class's
       ``parse_response`` wins MRO::

           class MyEndpoint(JMESPathResponseMixin, BaseEndpoint):
               ...

    2. Call ``self._init_response_parser()`` from ``__init__`` AFTER
       ``super().__init__(*args, **kwargs)``::

           def __init__(self, *args, **kwargs):
               super().__init__(*args, **kwargs)
               self._init_response_parser()

       Skipping step 2 makes the first ``parse_response`` raise
       ``AttributeError`` on ``_compiled_jmespath``.

    3. Do NOT override ``parse_response``; override
       ``convert_to_response_data`` / ``auto_detect_and_extract`` (inherited
       from ``BaseEndpoint``) to change the typed shape returned.

    See ``RawEndpoint`` for a reference subclass.
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
                # Deliberate degrade: a malformed response_field should not
                # crash the endpoint at construction. parse_response will fall
                # back to auto-detect, and the user sees the misconfiguration
                # via this log.
                self.error(
                    f"Failed to compile JMESPath query {response_field!r}: {e!r}. "
                    "Falling back to auto-detect response parsing — fix or remove "
                    "endpoint.extra.response_field to silence this log."
                )

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse response with auto-detection or custom JMESPath query.

        Resolution order:

        1. If the response has no JSON body, fall back to the raw text body
           wrapped in ``TextResponseData`` (or ``None`` if both are empty).
        2. If a ``response_field`` JMESPath query was compiled at init,
           run it against the JSON; on a non-empty match, hand the value to
           ``convert_to_response_data`` for type detection.
        3. If JMESPath produced no value (no query, empty match, or runtime
           ``JMESPathError`` / ``TypeError`` — logged at warning, not re-raised),
           fall back to ``auto_detect_and_extract`` which probes for
           embeddings, rankings, then text in that order.
        4. Wrap whatever we got in a ``ParsedResponse`` carrying
           ``response.perf_ns``; return ``None`` when no shape matched.

        Args:
            response: Raw response from inference server.

        Returns:
            Parsed response with the most specific detected type, or ``None``
            when neither JMESPath nor auto-detection found extractable data.
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
