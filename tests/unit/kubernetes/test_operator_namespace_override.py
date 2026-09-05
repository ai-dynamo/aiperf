# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The k8s test harness must honor a non-default operator namespace.

``--k8s-operator-namespace`` / ``K8S_TEST_OPERATOR_NAMESPACE`` exists so the
suite can run against a cluster that installed the chart somewhere other than
``aiperf-system``. Helpers that pin the namespace to a module-level literal
silently ignore that flag and target the wrong (or a nonexistent) namespace,
which surfaces as an unrelated timeout rather than a configuration error.
These tests lock the flag's reach.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pytest import param

from aiperf.kubernetes.constants import DEFAULT_OPERATOR_NAMESPACE

_TESTS_ROOT = Path(__file__).parents[2]
_K8S_TESTS = _TESTS_ROOT / "kubernetes"


def test_default_operator_namespace_is_the_chart_default() -> None:
    """Guards the literal every helper below defaults to."""
    assert DEFAULT_OPERATOR_NAMESPACE == "aiperf-system"


@pytest.mark.parametrize(
    "relative_path",
    [
        param("conftest.py", id="settings"),
        param("chaos/chaos_injector.py", id="chaos_injector"),
        param("chaos_aiperf/conftest.py", id="chaos_aiperf_conftest"),
        param("helpers/operator.py", id="operator_deployer"),
        param("helpers/helm.py", id="helm_deployer"),
        param("audit/operator_runner.py", id="audit_runner"),
        param("test_sweeps.py", id="sweeps"),
        param("chaos/test_sweep_controller_kill.py", id="sweep_controller_kill"),
        param("chaos_common/test_crd_injector.py", id="crd_injector_unit"),
    ],
)  # fmt: skip
def test_k8s_test_helpers_have_no_hardcoded_operator_namespace(
    relative_path: str,
) -> None:
    """No helper may bake the namespace in as a string literal.

    Docstrings are exempt (they legitimately quote the default by name);
    executable literals are not.
    """
    source = (_K8S_TESTS / relative_path).read_text()
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value == DEFAULT_OPERATOR_NAMESPACE
        and id(node) not in docstrings
    ]
    assert not offenders, (
        f"{relative_path} hardcodes {DEFAULT_OPERATOR_NAMESPACE!r} at line(s) "
        f"{offenders}; import DEFAULT_OPERATOR_NAMESPACE or thread "
        "k8s_settings.operator_namespace through instead"
    )


def test_chaos_injector_targets_the_configured_namespace() -> None:
    """``ChaosInjector`` must route operator-scoped kubectl calls via the arg."""
    from tests.kubernetes.chaos.chaos_injector import ChaosInjector

    injector = ChaosInjector(kubectl=object(), operator_namespace="custom-ns")
    assert injector.operator_namespace == "custom-ns"
    assert ChaosInjector(kubectl=object()).operator_namespace == (
        DEFAULT_OPERATOR_NAMESPACE
    )

    source = (_K8S_TESTS / "chaos" / "chaos_injector.py").read_text()
    tree = ast.parse(source)
    body = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "ChaosInjector"
    )
    bare_uses = [
        node.lineno
        for node in ast.walk(body)
        if isinstance(node, ast.Name)
        and node.id == "OPERATOR_NAMESPACE"
        # the ctor default is the one legitimate reference
        and node.lineno
        not in {
            arg.lineno
            for fn in ast.walk(body)
            if isinstance(fn, ast.FunctionDef)
            for arg in fn.args.defaults
        }
    ]
    assert not bare_uses, (
        "ChaosInjector methods reference the module-level OPERATOR_NAMESPACE at "
        f"line(s) {bare_uses}; use self.operator_namespace so "
        "--k8s-operator-namespace is honored"
    )
