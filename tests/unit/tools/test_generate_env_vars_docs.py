# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the environment-variable documentation generator."""

from tools.generate_env_vars_docs import EnvVarsDocsGenerator


def test_generate_includes_operator_service_base_url() -> None:
    """Operator settings remain part of the generated environment reference."""
    generated_docs = EnvVarsDocsGenerator().generate().files[0].content

    assert "`AIPERF_OPERATOR_BASE_URL`" in generated_docs


def test_generate_includes_operator_root_job_timeout() -> None:
    """Operator root settings remain part of the generated environment reference."""
    generated_docs = EnvVarsDocsGenerator().generate().files[0].content

    assert "`AIPERF_JOB_TIMEOUT_SECONDS`" in generated_docs
    assert "`AIPERF_ACCURACY`" not in generated_docs


def test_generate_operator_cluster_name_uses_root_prefix() -> None:
    """The cluster-name description names its actual environment variable."""
    generated_docs = EnvVarsDocsGenerator().generate().files[0].content

    assert "Set via AIPERF_CLUSTER_NAME" in generated_docs
    assert "AIPERF_OPERATOR_CLUSTER_NAME" not in generated_docs
