# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `aiperf config init`.

The command reuses `templates_cli` helpers (covered in
`tests/unit/config/test_templates_cli.py`); these tests focus on behaviors
unique to `aiperf config init`: default template selection, unknown-template
error path, overwrite prompt, stdout vs. file output, the kube-agnostic
"Next steps" banner, and that hints reference `aiperf config init`
rather than the kube variant.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aiperf.cli_commands.config_cli_init import init_config


def _parse(text: str) -> dict:
    """Parse YAML while ignoring the # SPDX / yaml-language-server headers."""
    return yaml.safe_load(text) or {}


class TestGenerate:
    """Default-generation path: unknown args → 'minimal' template to stdout/file."""

    def test_default_template_is_minimal(self, capsys) -> None:
        init_config()

        out = capsys.readouterr().out
        assert "model: meta-llama/Llama-3.1-8B-Instruct" in out
        assert "endpoint:" in out

    def test_writes_template_to_file(self, tmp_path: Path) -> None:
        output_file = tmp_path / "out.yaml"
        init_config(output=output_file)

        parsed = _parse(output_file.read_text())
        assert "model" in parsed or "models" in parsed
        assert "endpoint" in parsed

    def test_unknown_template_exits_with_message(self, capsys) -> None:
        with pytest.raises(SystemExit):
            init_config(template="zzz_no_such_template")

        out = capsys.readouterr().out
        assert "zzz_no_such_template" in out
        # Error lists available templates so users can self-correct.
        assert "minimal" in out

    def test_generated_output_strips_spdx_header(self, capsys) -> None:
        """Users pipe/commit the output; SPDX lines on every generated file
        would pollute downstream configs. Templates strip them at generation."""
        init_config(template="minimal")

        out = capsys.readouterr().out
        assert "SPDX-FileCopyrightText" not in out
        assert "SPDX-License-Identifier" not in out

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        output_file = tmp_path / "sub" / "deep" / "out.yaml"
        init_config(output=output_file)

        assert output_file.exists()


class TestOverrides:
    """--model / --url must land on the singular/plural form the template uses."""

    def test_model_override_on_singular_template(self, tmp_path: Path) -> None:
        """'minimal' template declares `model:` (singular)."""
        output_file = tmp_path / "out.yaml"
        init_config(template="minimal", model="custom-llm", output=output_file)

        parsed = _parse(output_file.read_text())
        assert parsed["model"] == "custom-llm"
        # No plural key leaked in.
        assert "models" not in parsed

    def test_url_override_on_singular_template(self, tmp_path: Path) -> None:
        output_file = tmp_path / "out.yaml"
        init_config(template="minimal", url="http://svc:8000", output=output_file)

        parsed = _parse(output_file.read_text())
        assert parsed["endpoint"]["url"] == "http://svc:8000"
        assert "urls" not in parsed["endpoint"]

    def test_both_overrides_simultaneously(self, tmp_path: Path) -> None:
        output_file = tmp_path / "out.yaml"
        init_config(
            template="minimal",
            model="custom-llm",
            url="http://svc:8000",
            output=output_file,
        )

        parsed = _parse(output_file.read_text())
        assert parsed["model"] == "custom-llm"
        assert parsed["endpoint"]["url"] == "http://svc:8000"

    def test_no_overrides_leaves_template_values_intact(self, tmp_path: Path) -> None:
        output_file = tmp_path / "out.yaml"
        init_config(template="minimal", output=output_file)

        parsed = _parse(output_file.read_text())
        # Template's default value — no override applied.
        assert parsed["model"] == "meta-llama/Llama-3.1-8B-Instruct"


class TestOverwritePrompt:
    """Existing file prompts via input(); reply is honored."""

    def test_refuses_to_overwrite_when_user_declines(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        output_file = tmp_path / "existing.yaml"
        output_file.write_text("do not overwrite")

        monkeypatch.setattr("builtins.input", lambda _prompt: "n")
        init_config(output=output_file)

        assert output_file.read_text() == "do not overwrite"
        assert "Aborted" in capsys.readouterr().out

    def test_overwrites_when_user_confirms(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        output_file = tmp_path / "existing.yaml"
        output_file.write_text("do not overwrite")

        monkeypatch.setattr("builtins.input", lambda _prompt: "y")
        init_config(output=output_file)

        content = output_file.read_text()
        assert "do not overwrite" not in content
        assert "endpoint:" in content

    def test_overwrite_prompt_accepts_capitalized_yes(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        output_file = tmp_path / "existing.yaml"
        output_file.write_text("old")

        monkeypatch.setattr("builtins.input", lambda _prompt: "YES")
        init_config(output=output_file)

        assert "old" not in output_file.read_text()


class TestNextStepsBanner:
    """The banner must advertise both bare-metal and kube usage paths."""

    def test_banner_mentions_both_profile_and_kube_profile(
        self, tmp_path: Path, capsys
    ) -> None:
        output_file = tmp_path / "out.yaml"
        init_config(output=output_file)

        out = capsys.readouterr().out
        assert f"aiperf profile --config {output_file}" in out
        assert f"aiperf kube profile --config {output_file}" in out
        assert "--image <your-image>" in out

    def test_stdout_mode_prints_no_banner(self, capsys) -> None:
        """Piping to stdout must stay pure YAML — banners would break `>` usage."""
        init_config()

        out = capsys.readouterr().out
        assert "Next steps" not in out
        assert "Created" not in out


class TestListAndSearch:
    """--list / --search short-circuit generation and reference `aiperf config init`."""

    def test_list_hints_reference_config_init_not_kube(self, capsys) -> None:
        init_config(list_templates=True)

        out = capsys.readouterr().out
        assert "Use 'aiperf config init --template <name>'" in out
        # Must not suggest the kube path when the user invoked `config init`.
        assert "aiperf kube init --template" not in out

    def test_search_hits_bypass_generation(self, capsys, tmp_path: Path) -> None:
        """--search must not also generate a template to stdout."""
        init_config(search="minimal")

        out = capsys.readouterr().out
        # Search results show the name...
        assert "minimal" in out
        # ...but no YAML body got emitted alongside.
        assert "endpoint:" not in out
        assert "phases:" not in out

    def test_search_no_match_references_config_init(self, capsys) -> None:
        init_config(search="zzz_no_such_template")

        out = capsys.readouterr().out
        assert "aiperf config init --list" in out
        assert "aiperf kube init --list" not in out

    def test_list_with_category_filter(self, capsys) -> None:
        init_config(list_templates=True, category="Load Testing")

        out = capsys.readouterr().out
        assert "Load Testing" in out
        # 'minimal' is in "Getting Started", not "Load Testing".
        assert "minimal" not in out


class TestVerboseListings:
    """--verbose surfaces Tags/Difficulty columns in list/search output."""

    def test_verbose_list_shows_tags_and_difficulty(self, capsys) -> None:
        init_config(list_templates=True, verbose=True)

        out = capsys.readouterr().out
        assert "Tags" in out
        assert "Difficulty" in out

    def test_non_verbose_list_hides_difficulty_column(self, capsys) -> None:
        init_config(list_templates=True, verbose=False)

        out = capsys.readouterr().out
        assert "Difficulty" not in out
