# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import setproctitle

from aiperf.common.base_service import BaseService
from tests.component_integration.conftest import (
    COMPONENT_INTEGRATION_PROCESS_TITLE,
    _set_component_integration_process_title,
)
from tests.harness.subprocess import _new_process_group_kwargs


def test_component_integration_process_title_uses_suite_name(monkeypatch):
    titles = []
    monkeypatch.setattr(setproctitle, "setproctitle", titles.append)

    _set_component_integration_process_title()

    assert COMPONENT_INTEGRATION_PROCESS_TITLE == "aiperf component_integration_test"
    assert titles == ["aiperf component_integration_test"]


def test_base_service_process_title_is_disabled_for_component_integration(monkeypatch):
    titles = []
    monkeypatch.setattr(setproctitle, "setproctitle", titles.append)

    service = SimpleNamespace(service_id="worker_123", debug=lambda message: None)
    BaseService._set_process_title(service)

    assert titles == []


def test_subprocess_process_group_kwargs_use_process_group_when_supported():
    assert _new_process_group_kwargs(supports_process_group=True) == {
        "process_group": 0
    }


def test_subprocess_process_group_kwargs_use_session_on_python_310():
    assert _new_process_group_kwargs(supports_process_group=False) == {
        "start_new_session": True
    }
