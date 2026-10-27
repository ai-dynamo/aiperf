# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A worker start-up failure that lands while the controller is still starting.

When every worker fails to start, the controller ends the run. If that happens
while ``_start_services`` is still registering or configuring, tearing down from
the error handler races the start-up task: start-up carries on to CONFIGURED and
PROFILING after teardown began, then fails, and that failure can reach runner
shutdown before the in-flight stop reaches ``os._exit``, hanging the process.

The race is timing-dependent, so this pins it deterministically with an
injected ``sitecustomize``: configure is held for a second, so the worker's
failure always arrives in CONFIGURING, and the cancel teardown is held for three,
so start-up gets to continue while a teardown is in flight -- the window the
hang needs. Start-up must not advance to CONFIGURED or PROFILING once every
worker has failed.
"""

import os

import pytest

from tests.harness.utils import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults

_HOLD_CONFIGURE = """
import importlib.abc, importlib.util, sys


class _PatchAfterLoad(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "aiperf.controller.system_controller":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            import asyncio

            cls = module.SystemController

            def held(method, seconds):
                async def wrapper(self, *args, **kwargs):
                    await asyncio.sleep(seconds)
                    return await method(self, *args, **kwargs)

                return wrapper

            cls._profile_configure_all_services = held(
                cls._profile_configure_all_services, 1.0
            )
            cls._await_cancel_result_domains = held(
                cls._await_cancel_result_domains, 3.0
            )

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _PatchAfterLoad())
"""

_AWS_CREDENTIAL_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_every_worker_failing_during_configure_aborts_start_up(
    cli: AIPerfCLI, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(_HOLD_CONFIGURE)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, [str(hook), os.environ.get("PYTHONPATH")])),
    )
    for var in _AWS_CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

    result = await cli.run(
        f"""
        aiperf profile \
            --model {defaults.model} \
            --tokenizer builtin \
            --url https://example.com \
            --auth-type sigv4 \
            --aws-region us-east-1 \
            --aws-service execute-api \
            --request-count 1 \
            --wait-for-model-timeout 0 \
            --workers-max 1 \
            --no-server-metrics \
            --no-gpu-telemetry \
            --ui none
        """,
        timeout=60.0,
        assert_success=False,
    )

    output = result.stdout + result.stderr
    assert result.exit_code == 1, output[-2000:]
    assert "No AWS credentials found" in output
    after_failure = output[output.index("Every worker failed to start") :]
    assert "AIPerf System is CONFIGURED" not in after_failure, after_failure[:2000]
    assert "AIPerf System is PROFILING" not in after_failure, after_failure[:2000]
