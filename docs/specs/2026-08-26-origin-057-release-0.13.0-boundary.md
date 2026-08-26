<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Origin #57 release 0.13.0 boundary

## Purpose

Record the coordinated release action for upstream commit
`e10d53b1d30b5845f56cbea63d0560f10ff5aa4e` without treating Python package
metadata as a native Rust feature.

## Built

The shared Python `aiperf` package and Python mock-server package both declare
`0.13.0`. The plugin-installation and server-metrics-schema examples use that
same release value. Native Rust Cargo package versioning remains independently
owned and unchanged.

## Source anchors

- `pyproject.toml`: Python distribution version.
- `tests/aiperf_mock_server/pyproject.toml`: Python mock-server distribution
  version.
- `docs/plugins/creating-your-first-plugin.md`: installed-package example.
- `docs/server-metrics/server-metrics-json-schema.md`: emitted-version examples.
