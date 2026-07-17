# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-prompt, context, and coding-conversation generators (mixin).

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from aiperf.dataset.generator._coding_text import (
    _USER_REQUESTS,
)
from aiperf.dataset.generator._coding_vocab import (
    _CLASSES,
    _ERROR_MESSAGES,
    _FILE_PATHS,
    _METHODS,
    _MODULES,
    _TYPES,
    _VARS,
)


class _SafeFormatMap(dict):
    """Dict subclass that returns '{key}' for missing keys in str.format_map."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


class _PromptsConvMixin:
    def _gen_user_prompt(self) -> str:
        r = self._template_rng
        template = r.choice(_USER_REQUESTS)
        base = template.format(
            module=r.choice(_MODULES),
            cls=r.choice(_CLASSES),
            method=r.choice(_METHODS),
            var=r.choice(_VARS),
            error=r.choice(_ERROR_MESSAGES),
            type=r.choice(_TYPES),
        )

        if r.random() < 0.3:
            base += "\n\n" + self._gen_prompt_context()
        return base

    def _gen_prompt_context(self) -> str:
        r = self._template_rng
        kind = r.choice(["snippet", "error_output", "constraint"])
        if kind == "snippet":
            cls = r.choice(_CLASSES)
            m1 = r.choice(_METHODS)
            v1, v2 = r.sample(_VARS, 2)
            f = r.choice(_FILE_PATHS)
            return (
                f"Here's the relevant code from `{f}`:\n\n"
                f"```\n"
                f"class {cls}:\n"
                f"    def {m1}(self, {v1}):\n"
                f"        {v2} = self._{v1}\n"
                f"        return {v2}\n"
                f"```"
            )
        elif kind == "error_output":
            err = r.choice(_ERROR_MESSAGES)
            cls = r.choice(_CLASSES)
            m1 = r.choice(_METHODS)
            f = r.choice(_FILE_PATHS)
            return (
                f"Error output:\n\n"
                f"```\n"
                f'  File "{f}", line {r.randint(10, 300)}, in {m1}\n'
                f'    raise RuntimeError("{err}")\n'
                f"RuntimeError: {err}\n"
                f"```"
            )
        else:
            return r.choice(
                (
                    "Constraint: no new dependencies allowed in this PR.",
                    "This is on the hot path — keep allocations minimal.",
                    "Must remain backward-compatible with the v1 API.",
                    f"The {r.choice(_MODULES)} service is frozen — only touch {r.choice(_MODULES)}.",
                    f"Target is under {r.randint(5, 50)}ms p99 latency.",
                    "We need this for the release on Friday — keep it simple.",
                )
            )

    def _gen_coding_conversation(self) -> str:
        r = self._template_rng
        return r.choice(
            [
                self._gen_conv_bugfix,
                self._gen_conv_review,
                self._gen_conv_feature,
                self._gen_conv_debug,
                self._gen_conv_qa,
                self._gen_conv_refactor,
                self._gen_conv_perf,
                self._gen_conv_cicd,
                self._gen_conv_ml_debug,
                self._gen_conv_test_write,
                self._gen_conv_migration,
                self._gen_conv_deploy,
                self._gen_conv_security,
                self._gen_conv_distributed,
                self._gen_conv_observability,
                self._gen_conv_db_optimize,
                self._gen_conv_architecture_review,
                self._gen_conv_incident_response,
            ]
        )()

    def _conv_ids(self) -> dict[str, str]:
        r = self._template_rng
        return {
            "cls": r.choice(_CLASSES),
            "module": r.choice(_MODULES),
            "method": r.choice(_METHODS),
            "var": r.choice(_VARS),
            "error": r.choice(_ERROR_MESSAGES),
        }

    def _conv_bridge(self, pool: tuple[str, ...], ids: dict[str, str]) -> str:
        r = self._template_rng
        return r.choice(pool).format_map(_SafeFormatMap(ids))

    def _conv_user_msg(self, ids: dict[str, str]) -> str:
        r = self._template_rng
        template = r.choice(_USER_REQUESTS)
        return template.format_map(_SafeFormatMap(ids))
