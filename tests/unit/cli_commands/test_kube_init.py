# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `aiperf kube init`.

Verifies the full template-library surface (list, search, generate with overrides)
and the AIPerfJob wrapping applied to every generated config.
"""

from __future__ import annotations

from pathlib import Path

import ruamel.yaml

from aiperf.cli_commands.kube.init import init_config


def _parse_yaml_from_cr(text: str) -> dict:
    """Strip comments and parse the AIPerfJob CR body."""
    yaml_lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return ruamel.yaml.YAML().load("\n".join(yaml_lines))


class TestInitGenerate:
    """Default generation path: template -> wrapped AIPerfJob."""

    def test_default_template_prints_aiperf_job_to_stdout(self, capsys) -> None:
        init_config(output=None)

        out = capsys.readouterr().out
        assert "apiVersion: aiperf.nvidia.com/v1alpha1" in out
        assert "kind: AIPerfJob" in out
        assert "  benchmark:" in out

    def test_writes_to_file_and_is_valid_yaml(self, tmp_path: Path) -> None:
        output_file = tmp_path / "benchmark.yaml"
        init_config(output=output_file)

        assert output_file.exists()
        parsed = _parse_yaml_from_cr(output_file.read_text())
        assert parsed["kind"] == "AIPerfJob"
        assert "benchmark" in parsed["spec"]

    def test_filename_appears_in_usage_comments(self, tmp_path: Path) -> None:
        output_file = tmp_path / "my-bench.yaml"
        init_config(output=output_file)

        content = output_file.read_text()
        assert "my-bench.yaml" in content

    def test_custom_job_name(self, tmp_path: Path) -> None:
        output_file = tmp_path / "run.yaml"
        init_config(output=output_file, job_name="run-42")

        parsed = _parse_yaml_from_cr(output_file.read_text())
        assert parsed["metadata"]["name"] == "run-42"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        output_file = tmp_path / "sub" / "deep" / "benchmark.yaml"
        init_config(output=output_file)

        assert output_file.exists()


class TestInitOverrides:
    """--model / --url patch the singular/plural form actually present."""

    def test_model_override_on_singular_form(self, tmp_path: Path) -> None:
        """'minimal' template uses `model:` (singular); override must match."""
        output_file = tmp_path / "out.yaml"
        init_config(template="minimal", model="my-model", output=output_file)

        parsed = _parse_yaml_from_cr(output_file.read_text())
        assert parsed["spec"]["benchmark"]["model"] == "my-model"

    def test_url_override_on_singular_form(self, tmp_path: Path) -> None:
        output_file = tmp_path / "out.yaml"
        init_config(template="minimal", url="http://svc:8000", output=output_file)

        parsed = _parse_yaml_from_cr(output_file.read_text())
        assert parsed["spec"]["benchmark"]["endpoint"]["url"] == "http://svc:8000"


class TestInitOverwritePrompt:
    """Existing output file prompts via input() and respects the response."""

    def test_refuses_to_overwrite_when_user_declines(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        output_file = tmp_path / "existing.yaml"
        output_file.write_text("old content")

        monkeypatch.setattr("builtins.input", lambda _prompt: "n")
        init_config(output=output_file)

        assert output_file.read_text() == "old content"

    def test_overwrites_when_user_confirms(self, tmp_path: Path, monkeypatch) -> None:
        output_file = tmp_path / "existing.yaml"
        output_file.write_text("old content")

        monkeypatch.setattr("builtins.input", lambda _prompt: "y")
        init_config(output=output_file)

        content = output_file.read_text()
        assert "kind: AIPerfJob" in content
        assert "old content" not in content


class TestInitListAndSearch:
    """--list and --search paths print the template catalog instead of generating."""

    def test_list_prints_category_headers(self, capsys) -> None:
        init_config(list_templates=True)

        out = capsys.readouterr().out
        assert "Getting Started" in out
        assert "minimal" in out
        # Hint should reference the kube command, not config init
        assert "aiperf kube init --template" in out

    def test_search_matches_template_name(self, capsys) -> None:
        init_config(search="minimal")

        out = capsys.readouterr().out
        assert "minimal" in out

    def test_search_no_match_prints_kube_hint(self, capsys) -> None:
        init_config(search="zzz_no_such_template_zzz")

        out = capsys.readouterr().out
        assert "No templates match" in out
        assert "aiperf kube init --list" in out
