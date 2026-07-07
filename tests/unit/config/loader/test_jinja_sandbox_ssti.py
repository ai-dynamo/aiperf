# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSTI hardening: load-time config rendering runs in a jinja2 sandbox.

The k8s operator renders attacker-controlled AIPerfJob/AIPerfSweep CRD
``spec.benchmark`` body fields (model, endpoint.url, phase fields, ...)
through ``expand_config_dict`` during create. A bare ``jinja2.Environment``
made that a server-side template injection -> RCE gadget: a payload in any
body string executed arbitrary shell in the operator pod on the normal,
error-free create path. These tests pin the sandbox so the classic escape
gadgets are blocked (raise) instead of executing, while legitimate templating
(variables, arithmetic, filters, loops) keeps rendering.
"""

from __future__ import annotations

import jinja2
import pytest
from jinja2.sandbox import SandboxedEnvironment
from pytest import param

from aiperf.config.loader.errors import ConfigurationError
from aiperf.config.loader.jinja import (
    _JINJA_ENV,
    expand_config_dict,
    render_jinja2_templates,
)

# Each gadget executes / leaks internals under a bare Environment and MUST be
# blocked (SecurityError, wrapped as ConfigurationError) under the sandbox.
SSTI_GADGETS = [
    param(
        "{{ cycler.__init__.__globals__.os.popen('id').read() }}",
        id="cycler-os-popen",
    ),
    param("{{ ''.__class__.__mro__ }}", id="str-class-mro"),
    param("{{ self.__init__.__globals__ }}", id="self-init-globals"),
    param("{{ ''.__class__.__base__.__subclasses__() }}", id="subclasses-escape"),
    param(
        "{{ cycler.__init__.__globals__.__builtins__['__import__']('os').system('id') }}",
        id="builtins-import-os-system",
    ),
    param("{{ namespace.__init__.__globals__ }}", id="namespace-init-globals"),
]


def test_module_env_is_sandboxed() -> None:
    """The load-time env must be a sandbox subclass, not a bare Environment.

    Guards against an accidental revert to ``jinja2.Environment(...)`` that would
    silently re-open the SSTI -> RCE hole on the operator create path.
    """
    assert isinstance(_JINJA_ENV, SandboxedEnvironment)


@pytest.mark.parametrize("gadget", SSTI_GADGETS)
def test_ssti_gadget_blocked_not_executed(gadget: str) -> None:
    """Each SSTI gadget in a body field raises instead of executing.

    Pre-fix (bare Environment) these render/execute and this ``raises`` fails;
    post-fix the sandbox raises SecurityError, wrapped as ConfigurationError.
    """
    with pytest.raises(ConfigurationError):
        render_jinja2_templates({"model": gadget}, context={})


def test_ssti_side_effect_never_runs(tmp_path) -> None:
    """Gold-standard 'did not execute': a filesystem side effect must not happen.

    The gadget would ``touch`` a sentinel file if the shell ran. Under the
    sandbox the attribute-escape is blocked before ``os.popen`` is ever reached,
    so the file is never created. Pre-fix this file WOULD exist.
    """
    sentinel = tmp_path / "aiperf_ssti_pwned"
    gadget = (
        "{{ cycler.__init__.__globals__.os.popen('touch "
        + str(sentinel)
        + "').read() }}"
    )
    with pytest.raises(ConfigurationError):
        render_jinja2_templates({"model": gadget}, context={})
    assert not sentinel.exists(), "SSTI gadget executed a shell command"


@pytest.mark.parametrize("gadget", SSTI_GADGETS)
def test_e2e_benchmark_model_gadget_blocked(gadget: str) -> None:
    """End-to-end via the operator's expand step: a gadget in benchmark.model.

    ``AIPerfJobSpecConverter.to_aiperf_config`` deep-copies ``spec.benchmark``
    and passes it straight to ``expand_config_dict``. A gadget planted at
    ``benchmark.model`` (top-level ``model`` in the body dict) must raise, never
    execute, on that path.
    """
    with pytest.raises(ConfigurationError):
        expand_config_dict({"model": gadget}, substitute_env=False)


def test_bare_sandbox_raises_security_error() -> None:
    """Documents the underlying primitive: the sandbox raises SecurityError.

    render_jinja2_templates wraps it as ConfigurationError; here we assert the
    raw class so a reader can see exactly what the sandbox does.
    """
    env = SandboxedEnvironment()
    with pytest.raises(jinja2.exceptions.SecurityError):
        env.from_string(
            "{{ cycler.__init__.__globals__.os.popen('id').read() }}"
        ).render()


def test_legit_templating_still_renders() -> None:
    """Sandbox must NOT break legitimate config templating.

    Exercises variable substitution, cross-variable arithmetic, attribute-chain
    access into named list entries, a stdlib filter, and {% for %}/{% if %}.
    """
    data = {
        "variables": {
            "concurrency_per_gpu": 30,
            "gpu_count": 4,
            "total": "{{ concurrency_per_gpu * gpu_count }}",
        },
        "benchmark": {
            "model": "mock-{{ total }}",
            "phases": [
                {"name": "warmup", "concurrency": 10},
                {
                    "name": "profiling",
                    "concurrency": "{{ phases.warmup.concurrency * 4 }}",
                    "requests": "{{ total | int + 1 }}",
                    "note": "{% for i in range(3) %}{% if i > 0 %}{{ i }}{% endif %}{% endfor %}",
                },
            ],
        },
    }
    result = expand_config_dict(data, substitute_env=False)

    assert result["variables"]["total"] == 120
    profiling = next(
        p for p in result["benchmark"]["phases"] if p["name"] == "profiling"
    )
    assert result["benchmark"]["model"] == "mock-120"
    assert profiling["concurrency"] == 40
    assert profiling["requests"] == 121
    # "12" renders then _coerce_rendered coerces the all-digit string to int.
    assert profiling["note"] == 12
