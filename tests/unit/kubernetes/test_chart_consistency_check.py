# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the chart README install-command scan in tools/check_chart_consistency.py.

The check used to be an unscoped whole-file substring test, so any unrelated
prose mentioning the namespace satisfied a gate that claimed to validate the
``helm install`` command.
"""

import pytest
from pytest import param

from tools.check_chart_consistency import (
    CHART_README,
    chart_readme_install_namespaces,
)

_REAL_INSTALL = """\
```bash
helm install aiperf-operator deploy/helm/aiperf-operator \\
  --namespace aiperf-system --create-namespace
```
"""


@pytest.mark.parametrize(
    "markdown, expected",
    [
        param(_REAL_INSTALL, ["aiperf-system"], id="multiline_install"),
        param(
            "```bash\nhelm install aiperf-operator ./deploy/helm/aiperf-operator -n aiperf-system\n```\n",
            ["aiperf-system"],
            id="short_flag",
        ),
        param(
            "```bash\nhelm upgrade --install aiperf-operator ./deploy/helm/aiperf-operator --namespace=aiperf-system\n```\n",
            ["aiperf-system"],
            id="upgrade_install_equals_form",
        ),
        param(
            "```bash\nhelm install aiperf-operator ./deploy/helm/aiperf-operator\n```\n",
            [""],
            id="missing_namespace_flag",
        ),
        param(
            "```bash\nhelm install aiperf-operator ./deploy/helm/aiperf-operator -n other-ns\n```\n",
            ["other-ns"],
            id="wrong_namespace",
        ),
        param(
            "Install into `--namespace aiperf-system` as usual.\n",
            [],
            id="prose_mention_is_not_a_command",
        ),
        param(
            "| `x` | see `--namespace aiperf-system` |\n",
            [],
            id="values_table_mention_is_not_a_command",
        ),
        param(
            "```bash\nkubectl get pods --namespace aiperf-system\n```\n",
            [],
            id="unrelated_fenced_command_ignored",
        ),
    ],
)  # fmt: skip
def test_chart_readme_install_namespaces_scopes_to_install_commands(
    markdown: str, expected: list[str]
) -> None:
    assert chart_readme_install_namespaces(markdown) == expected


def test_chart_readme_install_namespaces_real_readme_uses_default_namespace() -> None:
    from aiperf.kubernetes.constants import DEFAULT_OPERATOR_NAMESPACE

    namespaces = chart_readme_install_namespaces(
        CHART_README.read_text(encoding="utf-8")
    )
    assert namespaces, "chart README has no fenced helm install command"
    assert set(namespaces) == {DEFAULT_OPERATOR_NAMESPACE}
