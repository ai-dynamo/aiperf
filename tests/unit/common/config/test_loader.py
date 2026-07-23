# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supersession marker for the v1 config-loader tests.

The v1 loader (``aiperf.common.config.loader``) exposed
``_load_config_file`` / ``load_service_config`` / ``load_user_config`` --
extension-dispatched JSON/YAML readers that built ``ServiceConfig`` /
``UserConfig`` objects with an ``AIPERF_CONFIG_*_FILE`` env-var fallback. None of
that exists on v2: ``aiperf.common.config`` is not a package, there is no
``ServiceConfig``, and the schema-2.x loader is YAML-only.

The v2 loader is ``aiperf.config.loader.core`` (``load_config`` /
``load_config_from_string`` / ``load_config_dict`` / ``validate_config_file``),
and every behavior the v1 suite asserted is already covered against the v2 API:

| v1 test                                            | v2 coverage |
| -------------------------------------------------- | ----------- |
| TestLoadConfigFile.test_json_file / yaml / yml     | tests/unit/config/test_loader_edge_cases.py (load_config / load_config_from_string)
| TestLoadConfigFile.test_file_not_found_raises      | test_loader_edge_cases.py (load_config missing-file -> ConfigurationError)
| TestLoadConfigFile.test_unsupported_extension      | n/a -- v2 loader is YAML-only (no extension dispatch)
| TestLoadConfigFile.test_empty_yaml / empty_json    | test_loader_edge_cases.py (empty / "null" -> error) + test_loader_adversarial.py
| TestLoadConfigFile.test_non_mapping_json/yaml      | test_loader_edge_cases.py (list / scalar -> "must be a mapping")
| TestLoadConfigFile.test_case_insensitive_extension | n/a -- YAML-only
| TestLoadServiceConfig.*                            | n/a -- no ServiceConfig / no service-file loader on v2
| TestLoadUserConfig.*                               | tests/unit/config/test_end_to_end_config_flow.py + test_loader_edge_cases.py

The whole v1 file is therefore DROPPED (dup / no-v2-home). This module keeps a
single guard so the supersession decision is self-verifying: if anyone re-adds
the v1 loader API, this test fails and forces a real port of the suite above.
"""

from __future__ import annotations

import importlib

import pytest


def test_v1_config_loader_api_is_gone() -> None:
    """The v1 ``aiperf.common.config.loader`` module must not exist on v2.

    Guards the supersession documented in this module's docstring: the v1
    loader entry points were replaced wholesale by ``aiperf.config.loader``
    (covered by tests/unit/config/test_loader_*.py). If this import ever starts
    succeeding again, the v1 loader was re-introduced and its tests must be
    properly re-ported rather than left as this stub.
    """
    try:
        importlib.import_module("aiperf.common.config.loader")
    except ModuleNotFoundError:
        return
    pytest.fail(
        "aiperf.common.config.loader was re-introduced -- re-port the v1 "
        "loader test suite (see tests/unit/config/test_loader_edge_cases.py / "
        "test_loader_adversarial.py for the current v2 coverage)."
    )
