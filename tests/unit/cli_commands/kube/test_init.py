# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `aiperf kube init` cyclopts subcommand wiring.

Focus is on:
- module exposes `app` cyclopts.App; subcommand registered in `aiperf kube`
- `init_config` callable signature accepts the documented flags
- search/list short-circuits don't invoke `_generate_template`
- `_generate_template` maps CLI args onto the template helpers correctly
- error paths surface through `cli_utils.exit_on_error` (SystemExit, not raw traceback)

The full template-generation behaviour (real templates, AIPerfJob wrapping,
overwrite prompt) is exercised in `tests/unit/cli_commands/test_kube_init.py`
against the real `aiperf.config.templates` library; this file deliberately
mocks those helpers to test only the CLI shim.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_init_module_importable() -> None:
    """The init module must be importable and expose an `app` attribute."""
    from aiperf.cli_commands.kube import init

    assert hasattr(init, "app"), "init.app (cyclopts App) must be defined"


def test_init_registered_in_kube_app() -> None:
    """The `init` subcommand must be wired into `aiperf kube`."""
    from aiperf.cli_commands.kube._app import app

    assert "init" in set(app)


class TestInitCallableSignature:
    """`init_config` must accept the documented CLI flags as kwargs."""

    @pytest.mark.parametrize(
        "param_name",
        [
            "template",
            "list_templates",
            "search",
            "category",
            "verbose",
            "model",
            "url",
            "output",
            "job_name",
        ],
    )  # fmt: skip
    def test_signature_has_param(self, param_name: str) -> None:
        from aiperf.cli_commands.kube.init import init_config

        sig = inspect.signature(init_config)
        assert param_name in sig.parameters

    def test_signature_defaults(self) -> None:
        """Defaults match the docstring contract: bool flags off, paths/strings None."""
        from aiperf.cli_commands.kube.init import init_config

        sig = inspect.signature(init_config)
        assert sig.parameters["template"].default is None
        assert sig.parameters["list_templates"].default is False
        assert sig.parameters["search"].default is None
        assert sig.parameters["category"].default is None
        assert sig.parameters["verbose"].default is False
        assert sig.parameters["model"].default is None
        assert sig.parameters["url"].default is None
        assert sig.parameters["output"].default is None
        assert sig.parameters["job_name"].default == "my-benchmark"


class TestInitDispatch:
    """`init_config` must dispatch to the correct helper based on flag combinations."""

    def test_search_short_circuits_to_handle_search(self) -> None:
        """--search bypasses both --list and template generation."""
        from aiperf.cli_commands.kube.init import init_config

        with (
            patch("aiperf.config._cli_runner_templates.handle_search") as mock_search,
            patch("aiperf.config._cli_runner_templates.handle_list") as mock_list,
            patch("aiperf.cli_commands.kube.init._generate_template") as mock_generate,
        ):
            init_config(search="goodput", verbose=True)

        mock_search.assert_called_once()
        # call kwargs include verbose=True and the kube-specific cmd hint
        kwargs = mock_search.call_args.kwargs
        args = mock_search.call_args.args
        assert "goodput" in args or kwargs.get("query") == "goodput"
        assert kwargs.get("verbose") is True
        assert kwargs.get("cmd") == "aiperf kube init"
        mock_list.assert_not_called()
        mock_generate.assert_not_called()

    def test_list_flag_short_circuits_to_handle_list(self) -> None:
        """--list dispatches handle_list and skips template generation."""
        from aiperf.cli_commands.kube.init import init_config

        with (
            patch("aiperf.config._cli_runner_templates.handle_list") as mock_list,
            patch("aiperf.cli_commands.kube.init._generate_template") as mock_generate,
        ):
            init_config(list_templates=True, category="throughput", verbose=True)

        mock_list.assert_called_once()
        kwargs = mock_list.call_args.kwargs
        args = mock_list.call_args.args
        # category is positional in handle_list(category, *, verbose, cmd)
        assert "throughput" in args or kwargs.get("category") == "throughput"
        assert kwargs.get("verbose") is True
        assert kwargs.get("cmd") == "aiperf kube init"
        mock_generate.assert_not_called()

    def test_search_takes_precedence_over_list(self) -> None:
        """--search wins when both --search and --list are passed."""
        from aiperf.cli_commands.kube.init import init_config

        with (
            patch("aiperf.config._cli_runner_templates.handle_search") as mock_search,
            patch("aiperf.config._cli_runner_templates.handle_list") as mock_list,
        ):
            init_config(search="x", list_templates=True)

        mock_search.assert_called_once()
        mock_list.assert_not_called()

    def test_default_path_calls_generate_template(self, tmp_path: Path) -> None:
        """No search/list flags => `_generate_template` runs with all kwargs forwarded."""
        from aiperf.cli_commands.kube.init import init_config

        out = tmp_path / "bench.yaml"
        with patch("aiperf.cli_commands.kube.init._generate_template") as mock_generate:
            init_config(
                template="goodput_slo",
                model="m",
                url="http://svc:8000",
                output=out,
                job_name="job-1",
            )

        mock_generate.assert_called_once_with(
            template="goodput_slo",
            model="m",
            url="http://svc:8000",
            output=out,
            job_name="job-1",
        )


class TestGenerateTemplateHelper:
    """`_generate_template` glues template lib + AIPerfJob wrapper + writer."""

    def test_calls_helpers_in_order(self, tmp_path: Path) -> None:
        from aiperf.cli_commands.kube import init as init_mod

        info = MagicMock()
        info.name = "minimal"
        info.title = "Minimal Starter"

        out = tmp_path / "out.yaml"

        with (
            patch(
                "aiperf.config.templates.get_template", return_value=info
            ) as mock_get,
            patch(
                "aiperf.config.templates.load_template_content",
                return_value="raw-body",
            ) as mock_load,
            patch(
                "aiperf.config._cli_runner_templates.build_overrides",
                return_value={"model": "m"},
            ) as mock_build,
            patch(
                "aiperf.config.templates.strip_spdx_header",
                return_value="stripped-body",
            ) as mock_strip,
            patch(
                "aiperf.config.templates.apply_overrides",
                return_value="overridden-body",
            ) as mock_apply,
            patch(
                "aiperf.kubernetes.init_template.wrap_as_aiperf_job",
                return_value="wrapped-body",
            ) as mock_wrap,
            patch.object(init_mod, "_write_wrapped_template") as mock_write,
        ):
            init_mod._generate_template(
                template="minimal",
                model="m",
                url="http://x:8000",
                output=out,
                job_name="run-1",
            )

        mock_get.assert_called_once_with("minimal")
        mock_load.assert_called_once_with("minimal")
        mock_build.assert_called_once_with("raw-body", "m", "http://x:8000")
        mock_strip.assert_called_once_with("raw-body")
        mock_apply.assert_called_once_with("stripped-body", {"model": "m"})
        mock_wrap.assert_called_once()
        wrap_kwargs = mock_wrap.call_args.kwargs
        assert wrap_kwargs.get("filename") == "out.yaml"
        assert wrap_kwargs.get("job_name") == "run-1"
        mock_write.assert_called_once_with(
            "wrapped-body", out, "minimal", "Minimal Starter"
        )

    def test_skips_apply_overrides_when_empty(self, tmp_path: Path) -> None:
        """When build_overrides returns falsy, apply_overrides is NOT called."""
        from aiperf.cli_commands.kube import init as init_mod

        info = MagicMock()
        info.name = "minimal"
        info.title = "T"

        with (
            patch("aiperf.config.templates.get_template", return_value=info),
            patch(
                "aiperf.config.templates.load_template_content",
                return_value="body",
            ),
            patch(
                "aiperf.config._cli_runner_templates.build_overrides",
                return_value={},
            ),
            patch(
                "aiperf.config.templates.strip_spdx_header",
                return_value="body",
            ),
            patch("aiperf.config.templates.apply_overrides") as mock_apply,
            patch(
                "aiperf.kubernetes.init_template.wrap_as_aiperf_job",
                return_value="wrapped",
            ),
            patch.object(init_mod, "_write_wrapped_template"),
        ):
            init_mod._generate_template(
                template="minimal",
                model=None,
                url=None,
                output=None,
                job_name="my-benchmark",
            )

        mock_apply.assert_not_called()

    def test_default_template_name_is_minimal(self, tmp_path: Path) -> None:
        """Passing template=None resolves to 'minimal'."""
        from aiperf.cli_commands.kube import init as init_mod

        info = MagicMock()
        info.name = "minimal"
        info.title = "T"

        with (
            patch(
                "aiperf.config.templates.get_template", return_value=info
            ) as mock_get,
            patch(
                "aiperf.config.templates.load_template_content",
                return_value="body",
            ),
            patch(
                "aiperf.config._cli_runner_templates.build_overrides",
                return_value={},
            ),
            patch(
                "aiperf.config.templates.strip_spdx_header",
                return_value="body",
            ),
            patch(
                "aiperf.kubernetes.init_template.wrap_as_aiperf_job",
                return_value="wrapped",
            ),
            patch.object(init_mod, "_write_wrapped_template"),
        ):
            init_mod._generate_template(
                template=None,
                model=None,
                url=None,
                output=None,
                job_name="x",
            )

        mock_get.assert_called_once_with("minimal")

    def test_default_filename_when_no_output(self) -> None:
        """When --output is None, the filename hint is 'benchmark.yaml'."""
        from aiperf.cli_commands.kube import init as init_mod

        info = MagicMock()
        info.name = "minimal"
        info.title = "T"

        with (
            patch("aiperf.config.templates.get_template", return_value=info),
            patch(
                "aiperf.config.templates.load_template_content",
                return_value="body",
            ),
            patch(
                "aiperf.config._cli_runner_templates.build_overrides",
                return_value={},
            ),
            patch(
                "aiperf.config.templates.strip_spdx_header",
                return_value="body",
            ),
            patch(
                "aiperf.kubernetes.init_template.wrap_as_aiperf_job",
                return_value="wrapped",
            ) as mock_wrap,
            patch.object(init_mod, "_write_wrapped_template"),
        ):
            init_mod._generate_template(
                template=None,
                model=None,
                url=None,
                output=None,
                job_name="x",
            )

        assert mock_wrap.call_args.kwargs["filename"] == "benchmark.yaml"

    def test_unknown_template_exits_nonzero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """get_template KeyError -> kube_console.print_error + SystemExit(1)."""
        from aiperf.cli_commands.kube import init as init_mod

        with (
            patch(
                "aiperf.config.templates.get_template",
                side_effect=KeyError("template 'nope' not found"),
            ),
            patch("aiperf.kubernetes.console.print_error") as mock_err,
            pytest.raises(SystemExit) as exc_info,
        ):
            init_mod._generate_template(
                template="nope",
                model=None,
                url=None,
                output=None,
                job_name="x",
            )

        assert exc_info.value.code == 1
        mock_err.assert_called_once()


class TestWriteWrappedTemplate:
    """`_write_wrapped_template` writes file + prints next-steps via kube_console."""

    def test_stdout_path_when_output_none(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aiperf.cli_commands.kube.init import _write_wrapped_template

        _write_wrapped_template(
            "hello-content", output=None, info_name="minimal", info_title="T"
        )
        out = capsys.readouterr().out
        assert out == "hello-content"

    def test_writes_file_and_calls_console_helpers(self, tmp_path: Path) -> None:
        from aiperf.cli_commands.kube.init import _write_wrapped_template

        out = tmp_path / "subdir" / "bench.yaml"
        with (
            patch("aiperf.kubernetes.console.print_success") as mock_success,
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.print_action") as mock_action,
        ):
            _write_wrapped_template(
                "wrapped-yaml", output=out, info_name="minimal", info_title="Minimal"
            )

        assert out.exists()
        assert out.read_text() == "wrapped-yaml"
        mock_success.assert_called_once()
        mock_info.assert_called_once()
        # Three numbered next-step lines
        assert mock_action.call_count == 3

    def test_existing_file_user_declines_does_not_overwrite(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from aiperf.cli_commands.kube.init import _write_wrapped_template

        out = tmp_path / "exists.yaml"
        out.write_text("old")
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")

        _write_wrapped_template("new", output=out, info_name="minimal", info_title="T")

        assert out.read_text() == "old"
        assert "Aborted." in capsys.readouterr().out

    def test_existing_file_user_confirms_overwrites(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.cli_commands.kube.init import _write_wrapped_template

        out = tmp_path / "exists.yaml"
        out.write_text("old")
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")

        with (
            patch("aiperf.kubernetes.console.print_success"),
            patch("aiperf.kubernetes.console.print_info"),
            patch("aiperf.kubernetes.console.print_action"),
        ):
            _write_wrapped_template(
                "new", output=out, info_name="minimal", info_title="T"
            )

        assert out.read_text() == "new"


class TestInitErrorWrapping:
    """`exit_on_error` must catch unexpected helper exceptions and exit cleanly."""

    def test_unexpected_exception_in_generate_becomes_system_exit(self) -> None:
        from aiperf.cli_commands.kube.init import init_config

        with (
            patch(
                "aiperf.cli_commands.kube.init._generate_template",
                side_effect=RuntimeError("boom"),
            ),
            patch("aiperf.cli_utils.console"),
            pytest.raises(SystemExit) as exc_info,
        ):
            init_config()

        # exit_on_error defaults to exit_code=1
        assert exc_info.value.code == 1
