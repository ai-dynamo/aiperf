# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2
import orjson

from aiperf.common.exceptions import InvalidStateError
from aiperf.common.models import RequestInfo
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.response_mixin import JMESPathResponseMixin

NAMED_TEMPLATES: dict[str, str] = {
    "nv-embedqa": '{"text": {{ texts|tojson }}}',
}


class TemplateEndpoint(JMESPathResponseMixin, BaseEndpoint):
    """Custom template endpoint using Jinja2 for payload formatting.

    Allows users to define custom request payload formats using Jinja2 templates.
    Templates can be named templates (from NAMED_TEMPLATES), file paths, or
    inline template strings.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        extra = self.model_endpoint.endpoint.extra
        extra_dict = dict(extra) if extra else {}

        template_source = extra_dict.get("payload_template")
        if not template_source:
            raise InvalidStateError(
                "Template endpoint requires 'payload_template' in endpoint.extra configuration"
            )

        if template_source in NAMED_TEMPLATES:
            self.info(f"Using named template: '{template_source}'")
            template_source = NAMED_TEMPLATES[template_source]
        else:
            try:
                template_path = Path(template_source)
                if template_path.is_file():
                    self.info(f"Loading template from file: '{template_path}'")
                    template_source = template_path.read_text(encoding="utf-8")
            except (OSError, ValueError) as e:
                self.debug(f"Not a file or treating as inline template: '{e!r}'")

        self._template = jinja2.Environment(autoescape=True).from_string(
            template_source
        )
        self.info(f"Compiled template ({len(template_source)} chars)")

        self._init_response_parser()

        self._extra_fields = {
            k: v
            for k, v in extra_dict.items()
            if k not in ("payload_template", "response_field")
        }

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format custom template request payload from RequestInfo.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            Custom payload formatted according to the Jinja2 template
        """
        if not request_info.turns:
            raise ValueError("Template endpoint requires at least one turn.")

        turn = request_info.turns[0]

        texts, texts_by_name = self.extract_named_contents(turn.texts)
        images, images_by_name = self.extract_named_contents(turn.images)
        audios, audios_by_name = self.extract_named_contents(turn.audios)
        videos, videos_by_name = self.extract_named_contents(turn.videos)

        queries = texts_by_name.get("query", [])
        passages = texts_by_name.get("passages") or texts_by_name.get("passage", [])

        template_vars = {
            "texts": texts or [],
            "images": images or [],
            "audios": audios or [],
            "videos": videos or [],
            "text": texts[0] if texts else None,
            "image": images[0] if images else None,
            "audio": audios[0] if audios else None,
            "video": videos[0] if videos else None,
            "queries": queries or [],
            "passages": passages or [],
            "query": queries[0] if queries else None,
            "passage": passages[0] if passages else None,
            "texts_by_name": texts_by_name or {},
            "images_by_name": images_by_name or {},
            "audios_by_name": audios_by_name or {},
            "videos_by_name": videos_by_name or {},
            "model": turn.model or self.model_endpoint.primary_model_name,
            "max_tokens": turn.max_tokens,
            "role": turn.role,
            "turn": turn,
            "turns": request_info.turns,
            "request_info": request_info,
            "stream": self.model_endpoint.endpoint.streaming,
        }

        rendered = self._template.render(**template_vars)

        try:
            payload = orjson.loads(rendered)
        except orjson.JSONDecodeError as e:
            self.error(f"Template did not render valid JSON: {rendered} - {e!r}")
            raise ValueError(
                f"Template did not render valid JSON {e!r}: {rendered[:100]}"
            ) from e

        if self._extra_fields:
            payload.update(self._extra_fields)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload
