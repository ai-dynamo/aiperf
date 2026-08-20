#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assert the AIPerf operator's code-side defaults match the Helm chart's values.

Without this gate, code-side and chart-side ports / URLs / namespace defaults
drift silently — the kind of mismatch that caused round-1's BASE_URL pointing
at port 8080 (where there's no FastAPI), round-3's metrics port not being
exposed on the Service, and the round-5 ``RESULTS_SERVER_PORT`` env-var gap.

What this script asserts:

1. ``OperatorEnvironment.SERVICE.BASE_URL`` default port matches
   ``Values.resultsServer.port`` — round-1 collapse depends on this.
2. ``OperatorEnvironment.METRICS_PORT`` default matches
   ``Values.operator.metrics.port`` — round-3 ServiceMonitor + Service
   wiring depends on this.
3. ``aiperf.kubernetes.results_operator.RESULTS_SERVER_PORT`` default
   matches ``Values.resultsServer.port`` — round-5 CLI port-forward
   target needs to match what the chart binds.
4. ``DEFAULT_OPERATOR_NAMESPACE`` constant matches the chart README's
   install-command default.

Runs as a unit-scope check: pure import of the settings classes + a YAML
parse of ``values.yaml``. Designed to be called from pre-commit and CI.

Exit codes:
    0 — all consistent
    1 — at least one drift detected (printed to stderr)
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
VALUES_YAML = REPO_ROOT / "deploy" / "helm" / "aiperf-operator" / "values.yaml"


def _load_values() -> dict:
    import yaml

    return yaml.safe_load(VALUES_YAML.read_text(encoding="utf-8"))


def _isolated_default(field_path: str) -> object:
    """Read a Pydantic settings field's pure default value.

    Imports the class without instantiating it (so ``BaseSettings`` doesn't
    pick up shell env vars from the running pre-commit / CI environment),
    then reads the ``model_fields[<name>].default`` attribute.

    Modules that resolve ``os.environ`` at import time only read clean values
    on their *first* import, and a previous import elsewhere in the process
    may have already cached a poisoned value. Reload inside the stripped-env
    block so the read is clean either way, then reload again once the real
    env is restored so the rest of the process sees normal module state.
    """
    # Strip any AIPERF_* env vars that would override defaults at instantiation
    # time. Also clear AIPERF_* env vars from os.environ so the import-time
    # ``int(os.environ.get(...))`` pattern in results_operator.py reads clean.
    poisoned_env = {k: v for k, v in os.environ.items() if k.startswith("AIPERF_")}
    for key in poisoned_env:
        del os.environ[key]
    module_path, attr_path = field_path.split(":", 1)
    try:
        module = importlib.reload(importlib.import_module(module_path))
        obj: object = module
        for part in attr_path.split("."):
            obj = getattr(obj, part)
        return obj
    finally:
        os.environ.update(poisoned_env)
        if poisoned_env:
            importlib.reload(sys.modules[module_path])


def _check(name: str, code_value: object, chart_value: object, errs: list[str]) -> None:
    if code_value != chart_value:
        errs.append(
            f"DRIFT: {name}\n"
            f"  code side:  {code_value!r}\n"
            f"  chart side: {chart_value!r}\n"
            f"  → values.yaml or the Pydantic default has drifted; "
            f"reconcile both before merging."
        )


def main() -> int:
    if not VALUES_YAML.is_file():
        print(f"Error: chart values not found: {VALUES_YAML}", file=sys.stderr)
        return 1

    values = _load_values()
    errs: list[str] = []

    # 1. BASE_URL default port == resultsServer.port
    from aiperf.operator.environment import _OperatorServiceSettings

    base_url_default = _OperatorServiceSettings.model_fields["BASE_URL"].default
    base_url_port = urlparse(str(base_url_default)).port
    chart_results_port = values["resultsServer"]["port"]
    _check(
        "BASE_URL default port vs Values.resultsServer.port",
        base_url_port,
        chart_results_port,
        errs,
    )

    # 2. METRICS_PORT default == operator.metrics.port
    from aiperf.operator.environment import _OperatorEnvironment

    metrics_port_default = _OperatorEnvironment.model_fields["METRICS_PORT"].default
    chart_metrics_port = values["operator"]["metrics"]["port"]
    _check(
        "OperatorEnvironment.METRICS_PORT vs Values.operator.metrics.port",
        metrics_port_default,
        chart_metrics_port,
        errs,
    )

    # 3. RESULTS_SERVER_PORT (env-driven module constant) default == resultsServer.port
    rsp_default = _isolated_default(
        "aiperf.kubernetes.results_operator:RESULTS_SERVER_PORT"
    )
    _check(
        "kubernetes.results_operator.RESULTS_SERVER_PORT vs Values.resultsServer.port",
        rsp_default,
        chart_results_port,
        errs,
    )

    # 4. DEFAULT_OPERATOR_NAMESPACE matches chart README install-command default
    from aiperf.kubernetes.constants import DEFAULT_OPERATOR_NAMESPACE

    chart_readme = (
        REPO_ROOT / "deploy" / "helm" / "aiperf-operator" / "README.md"
    ).read_text(encoding="utf-8")
    if f"--namespace {DEFAULT_OPERATOR_NAMESPACE}" not in chart_readme:
        errs.append(
            f"DRIFT: DEFAULT_OPERATOR_NAMESPACE={DEFAULT_OPERATOR_NAMESPACE!r} "
            f"is not used in deploy/helm/aiperf-operator/README.md install command. "
            f"Either the constant is wrong, or the README's `--namespace <ns>` "
            f"in the install command needs to follow the constant."
        )

    if errs:
        print(
            "ERROR: aiperf operator code↔chart consistency drift detected:\n",
            file=sys.stderr,
        )
        for err in errs:
            print(err, file=sys.stderr)
            print(file=sys.stderr)
        return 1

    print("OK: code-side defaults match Helm chart values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
