# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from django.db.backends.postgresql.client import DatabaseClient


class PostgreSqlDbshellCommandTestCase:
    def settings_to_cmd_args_env(self, settings_dict, parameters=None):
        return DatabaseClient.settings_to_cmd_args_env(settings_dict, parameters or [])

    def test_parameters(self):
        assert self.settings_to_cmd_args_env({"NAME": "dbname"}, ["--help"]) == (
            ["psql", "dbname", "--help"],
            None,
        )
