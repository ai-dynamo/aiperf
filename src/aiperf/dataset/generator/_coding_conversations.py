# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversation-template generators (mixin for CodingContentGenerator).

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from aiperf.dataset.generator._coding_text import (
    _BRIDGE_ANALYZE,
    _BRIDGE_EXPLAIN,
    _BRIDGE_FIX,
    _BRIDGE_PERF,
    _BRIDGE_REFACTOR,
    _BRIDGE_SUMMARY,
    _BRIDGE_TEST,
    _BRIDGE_WRITE_TEST,
    _FOLLOWUP_QUESTIONS,
    _LANGUAGES,
)


class _ConversationsMixin:
    def _gen_conv_bugfix(self) -> str:
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_review(self) -> str:
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}\n\n"
            f"{self._gen_git_diff(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[User]\n{self._conv_bridge(_FOLLOWUP_QUESTIONS, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_feature(self) -> str:
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_search_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_WRITE_TEST, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_debug(self) -> str:
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()
        error_block = r.choice(
            [
                lambda: self._gen_error_traceback(language=lang),
                self._gen_cuda_error,
            ]
        )()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}\n\n{error_block}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_search_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_qa(self) -> str:
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_EXPLAIN, ids)}",
            f"[User]\n{self._conv_bridge(_FOLLOWUP_QUESTIONS, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_refactor(self) -> str:
        """Multi-file refactoring: search callers, read multiple files, edit each."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_search_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_REFACTOR, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\nNow let me update the callers.\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_REFACTOR, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_perf(self) -> str:
        """Performance investigation: profile, read hot path, optimize, benchmark."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_PERF, ids)}\n\n"
            f"{self._gen_tool_bash(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_PERF, ids)}\n\n"
            f"{self._gen_tool_search(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[User]\n{self._conv_bridge(_FOLLOWUP_QUESTIONS, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_EXPLAIN, ids)}\n\n"
            f"{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_cicd(self) -> str:
        """CI/CD debugging: failing pipeline, read logs, fix config, re-run."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        ci_output = self._gen_cicd_output(language=lang)

        turns = [
            f"[User]\nThe CI pipeline is failing on the {ids['module']} service. "
            f"Can you take a look?\n\n{ci_output}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
            f"[User]\n{self._conv_bridge(_FOLLOWUP_QUESTIONS, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_EXPLAIN, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_ml_debug(self) -> str:
        """ML/GPU debugging: CUDA error, read training code, fix, re-run."""
        ids = self._conv_ids()

        cuda_err = self._gen_cuda_error()
        training_code = self._gen_ml_training_code()
        training_log = self._gen_ml_training_log()
        inference_code = self._gen_ml_inference_code()

        turns = [
            f"[User]\nI'm getting a CUDA error during training. "
            f"Here's the error:\n\n{cuda_err}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"<tool_name>read</tool_name>\n"
            f'<parameter name="file_path">train.py</parameter>\n'
            f"<result>\n{training_code}\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"<tool_name>read</tool_name>\n"
            f'<parameter name="file_path">inference.py</parameter>\n'
            f"<result>\n{inference_code}\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language='python')}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"<tool_name>bash</tool_name>\n"
            f'<parameter name="command">python train.py --max-steps 10</parameter>\n'
            f"<result>\n{training_log}\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
            "[User]\nCan you also check if the inference script has the same issue?",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._conv_bridge(_BRIDGE_EXPLAIN, ids)}\n\n"
            f"{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_test_write(self) -> str:
        """Test writing session: read code, write tests, iterate on failures."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\nWrite comprehensive tests for {ids['cls']}.{ids['method']}(). "
            f"Cover the happy path, edge cases, and error handling.",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_search(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_WRITE_TEST, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[User]\n{self._conv_bridge(_FOLLOWUP_QUESTIONS, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_WRITE_TEST, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)
