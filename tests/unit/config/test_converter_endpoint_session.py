# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.config.flags._converter_endpoint import build_endpoint
from aiperf.config.flags.cli_config import CLIConfig


def test_session_affinity_fields_are_added_to_endpoint() -> None:
    endpoint = build_endpoint(
        CLIConfig(
            session_header="x-dynamo-session-id",
            session_body_field="session_id",
        )
    )

    assert endpoint["session_header"] == "x-dynamo-session-id"
    assert endpoint["session_body_field"] == "session_id"
