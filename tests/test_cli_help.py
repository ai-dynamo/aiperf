# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.cli import app


class TestCLIHelp:
    def test_profile_help_does_not_show_parameters(self, capsys):
        """This test is to ensure that the help text for the profile command does
        not show miscellaneous un-grouped parameters."""
        app(["profile", "-h"])
        assert "─ Parameters ─" not in capsys.readouterr().out

    def test_no_args_does_not_crash(self, capsys):
        """This test is to ensure that the CLI does not crash when no arguments are provided."""
        app([])
        out = capsys.readouterr().out
        assert "Usage: aiperf COMMAND" in out
        assert "─ Commands ─" in out
