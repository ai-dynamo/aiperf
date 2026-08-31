# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Machine-readable stdout must stay parseable regardless of console width.

Every command with a JSON/YAML output mode writes through
``kube_console.emit_raw``, which bypasses Rich. Routing the same payload through
``kube_console.console.print`` hard-wraps any line wider than the resolved
console width (80 whenever stdout is not a tty), which splits long image
references and URLs mid-token and makes the document unparseable.

The tests below pin the console to width 80 so the guard fires deterministically
instead of depending on the terminal that happens to run pytest.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import orjson
import pytest
import yaml
from pytest import param

from aiperf.cli_commands.kube import preflight as preflight_cmd
from aiperf.cli_commands.kube.profile_deploy import deploy_via_operator
from aiperf.config.kube import KubeManageOptions, KubeOptions
from aiperf.kubernetes import console as kube_console
from aiperf.kubernetes.preflight import CheckResult, CheckStatus, PreflightResults
from tests.harness import fixed_console

if TYPE_CHECKING:
    from collections.abc import Iterator

# 150 characters: longer than the 80-column fallback Rich uses without a tty,
# and a single unbreakable token so any wrap lands inside the value.
LONG_IMAGE = (
    "registry.example.com/some-really-long-organization-name/"
    "aiperf-benchmark-runner-with-extras:"
    "v1.2.3-rc4-cuda12.6-py311-abcdef0123456789abcdef0123456789"
)


@pytest.fixture(autouse=True)
def narrow_console(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the kube console to 80 columns so a Rich-routed payload would wrap."""
    monkeypatch.setattr(kube_console, "console", fixed_console(80))
    yield


def _longest_line(text: str) -> int:
    return max((len(line) for line in text.splitlines()), default=0)


class TestEmitRaw:
    """`emit_raw` is the single machine-readable output path."""

    def test_emit_raw_json_with_long_token_stays_parseable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        payload = {"image": LONG_IMAGE}

        kube_console.emit_raw(
            orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode()
        )

        out = capsys.readouterr().out
        assert orjson.loads(out) == payload
        assert _longest_line(out) > 80

    def test_emit_raw_yaml_with_long_token_stays_parseable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document = yaml.safe_dump({"image": LONG_IMAGE}, width=float("inf"))

        kube_console.emit_raw(document, end="")

        out = capsys.readouterr().out
        assert yaml.safe_load(out) == {"image": LONG_IMAGE}
        assert out == document

    def test_emit_raw_default_end_appends_single_newline(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        kube_console.emit_raw("{}")

        assert capsys.readouterr().out == "{}\n"


class TestMachineReadableStdout:
    """The suppression window must keep records of every level off stdout."""

    @pytest.mark.parametrize(
        "method",
        [
            param("warning", id="warning"),
            param("error", id="error"),
            param("critical", id="critical"),
        ],
    )  # fmt: skip
    def test_records_go_to_stderr_not_stdout(
        self, method: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with kube_console.machine_readable_stdout():
            getattr(kube_console.logger, method)("diagnostic detail")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "diagnostic detail" in captured.err

    def test_handler_console_and_level_are_restored(self) -> None:
        kube_logger = logging.getLogger("aiperf.kube")
        original_level = kube_logger.level
        original_consoles = [
            getattr(handler, "console", None) for handler in kube_logger.handlers
        ]

        with kube_console.machine_readable_stdout():
            pass

        assert kube_logger.level == original_level
        assert [
            getattr(handler, "console", None) for handler in kube_logger.handlers
        ] == original_consoles

    def test_handler_console_is_restored_after_exception(self) -> None:
        kube_logger = logging.getLogger("aiperf.kube")
        original_consoles = [
            getattr(handler, "console", None) for handler in kube_logger.handlers
        ]

        with pytest.raises(RuntimeError), kube_console.machine_readable_stdout():
            raise RuntimeError("check restoration")

        assert [
            getattr(handler, "console", None) for handler in kube_logger.handlers
        ] == original_consoles


class TestPreflightJsonOutput:
    """`aiperf kube preflight -o json` emits only the JSON document."""

    @staticmethod
    def _failing_checker_class() -> type:
        class FakeChecker:
            def __init__(self, namespace: str, **_: Any) -> None:
                self.namespace = namespace

            async def run_all_checks(self) -> PreflightResults:
                results = PreflightResults(
                    checks=[
                        CheckResult(
                            name="Cluster Connectivity",
                            status=CheckStatus.FAIL,
                            message=f"image {LONG_IMAGE} unreachable",
                        )
                    ]
                )
                results.print_summary()
                return results

        return FakeChecker

    @pytest.mark.asyncio
    async def test_failed_checks_json_has_no_log_prefix(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "aiperf.kubernetes.preflight.CLIPreflightChecker",
            self._failing_checker_class(),
        )

        with pytest.raises(SystemExit) as excinfo:
            await preflight_cmd._run_preflight(
                manage_options=KubeManageOptions(namespace="bench-ns"),
                image=LONG_IMAGE,
                image_pull_secrets=None,
                secrets=None,
                endpoint_url=None,
                workers=1,
                node_selector=None,
                tolerations=None,
                output="json",
            )

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert captured.out.startswith("{")
        payload = orjson.loads(captured.out)
        assert payload["passed"] is False
        assert payload["checks"][0]["message"].endswith("unreachable")
        # Diagnostics stay visible, just not on the machine-readable stream.
        assert "pre-flight checks failed" in captured.err

    @pytest.mark.asyncio
    async def test_text_mode_still_logs_the_failure_summary(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "aiperf.kubernetes.preflight.CLIPreflightChecker",
            self._failing_checker_class(),
        )
        logging.getLogger("aiperf.kube").setLevel(logging.INFO)

        with pytest.raises(SystemExit):
            await preflight_cmd._run_preflight(
                manage_options=KubeManageOptions(namespace="bench-ns"),
                image=None,
                image_pull_secrets=None,
                secrets=None,
                endpoint_url=None,
                workers=1,
                node_selector=None,
                tolerations=None,
                output="text",
            )

        assert "pre-flight checks failed" in capsys.readouterr().out


class TestProfileDryRun:
    """`aiperf kube profile --dry-run` emits a parseable AIPerfJob CR."""

    @pytest.mark.asyncio
    async def test_dry_run_json_with_long_image_stays_parseable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = MagicMock()
        config.benchmark.endpoint.urls = []
        config.benchmark.get_model_names.return_value = ["model-a"]

        await deploy_via_operator(
            {"image": LONG_IMAGE},
            KubeOptions(image=LONG_IMAGE),
            config,
            "my-bench",
            "aiperf-benchmarks",
            dry_run=True,
            detach=True,
            no_wait=False,
            attach_port=0,
        )

        out = capsys.readouterr().out
        cr = orjson.loads(out)
        assert cr["spec"]["image"] == LONG_IMAGE
        assert _longest_line(out) > 80
