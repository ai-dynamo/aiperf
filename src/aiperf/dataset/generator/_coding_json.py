# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON-response template generators (mixin for CodingContentGenerator).

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from aiperf.dataset.generator._coding_vocab import (
    _CLASSES,
    _ERROR_MESSAGES,
    _METHODS,
    _MODULES,
    _STATUS_CODES,
    _VARS,
)


class _JsonMixin:
    def _gen_json_response(self, language: str | None = None) -> str:
        return self._template_rng.choice(
            [
                self._gen_json_object,
                self._gen_json_paginated,
                self._gen_json_error,
            ]
        )()

    def _gen_json_object(self) -> str:
        r = self._template_rng
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2, v3 = r.sample(_VARS, 3)
        cls = r.choice(_CLASSES)
        id_suffix = r.randint(1000, 9999)
        num_val = r.randint(0, 1000)
        float_val = r.uniform(0, 1)
        ts = r.randint(1700000000, 1800000000)
        items = [
            f'      {{{{"id": {r.randint(1, 999)}, "name": "{r.choice(_VARS)}"}}}}'
            for _ in range(3)
        ]
        items_str = ",\n".join(items)

        return f"""\
{{{{
  "status": "ok",
  "data": {{{{
    "{v1}": "{cls.lower()}_{id_suffix}",
    "{v2}": {num_val},
    "{v3}": {float_val:.4f},
    "metadata": {{{{
      "action": "{m1}",
      "source": "{m2}",
      "timestamp": "{ts}"
    }}}},
    "items": [
{items_str}
    ]
  }}}}
}}}}
"""

    def _gen_json_paginated(self) -> str:
        r = self._template_rng
        v1, v2 = r.sample(_VARS, 2)
        cls = r.choice(_CLASSES)
        total = r.randint(50, 5000)
        page = r.randint(1, 20)
        per_page = r.choice([10, 20, 50, 100])
        items = [
            f'    {{{{"id": "{cls.lower()}_{r.randint(1000, 9999)}", "{v1}": "{r.choice(_MODULES)}", "{v2}": {r.randint(0, 100)}}}}}'
            for _ in range(min(per_page, 5))
        ]
        items_str = ",\n".join(items)

        return f"""\
{{{{
  "data": [
{items_str}
  ],
  "pagination": {{{{
    "page": {page},
    "per_page": {per_page},
    "total": {total},
    "total_pages": {(total + per_page - 1) // per_page},
    "has_next": {str(page * per_page < total).lower()},
    "has_prev": {str(page > 1).lower()}
  }}}}
}}}}
"""

    def _gen_json_error(self) -> str:
        r = self._template_rng
        err = r.choice(_ERROR_MESSAGES)
        status = r.choice(_STATUS_CODES)
        code = status.split()[0]
        trace_id = f"{r.randint(100000, 999999):06x}-{r.randint(100000, 999999):06x}"
        v1 = r.choice(_VARS)
        cls = r.choice(_CLASSES)

        return f"""\
{{{{
  "error": {{{{
    "code": {code},
    "status": "{status}",
    "message": "{err}",
    "details": [
      {{{{
        "field": "{v1}",
        "reason": "{err}",
        "type": "{cls}"
      }}}}
    ],
    "trace_id": "{trace_id}",
    "documentation_url": "https://docs.example.com/errors/{code}"
  }}}}
}}}}
"""
