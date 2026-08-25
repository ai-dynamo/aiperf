# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the environment-variable documentation generator."""

from tools.generate_env_vars_docs import EnvVarsDocsGenerator


def test_generate_includes_common_settings() -> None:
    """Common settings are part of the generated environment reference."""
    generated_docs = EnvVarsDocsGenerator().generate().files[0].content

    assert "`AIPERF_HTTP_CONNECTION_LIMIT`" in generated_docs
    assert "`AIPERF_WORKER_STALE_TIME`" in generated_docs


def test_generate_documents_every_subsystem_heading() -> None:
    """Each ``_XxxSettings`` class contributes its own subsystem section."""
    generated_docs = EnvVarsDocsGenerator().generate().files[0].content

    for heading in ("## HTTP", "## WORKER", "## ZMQ", "## RECORD"):
        assert heading in generated_docs
