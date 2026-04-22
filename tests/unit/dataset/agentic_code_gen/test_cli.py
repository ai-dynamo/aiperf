# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agentic Code CLI helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aiperf.dataset.agentic_code_gen.cli import _apply_cli_overrides, validate
from aiperf.dataset.agentic_code_gen.models import SessionDistributionConfig


class TestCliOverrides:
    def test_apply_cli_overrides_revalidates_config(self) -> None:
        config = SessionDistributionConfig()

        overridden = _apply_cli_overrides(config, max_isl=1024, max_osl=2048)

        assert overridden.max_prompt_tokens == 1024
        assert overridden.generation_length.max == 2048.0

    @pytest.mark.parametrize(
        ("max_isl", "max_osl"),
        [
            (0, None),
            (None, 0),
        ],
    )
    def test_apply_cli_overrides_rejects_invalid_values(
        self, max_isl: int | None, max_osl: int | None
    ) -> None:
        config = SessionDistributionConfig()

        with pytest.raises(ValidationError):
            _apply_cli_overrides(config, max_isl=max_isl, max_osl=max_osl)


class TestValidateCommand:
    def test_validate_missing_file_exits_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "missing.jsonl"

        with pytest.raises(SystemExit) as exc:
            validate(missing)

        assert exc.value.code == 1
        assert "is not a file" in capsys.readouterr().out
