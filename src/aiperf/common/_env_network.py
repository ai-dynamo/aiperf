# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Network-facing environment settings subgroups.

Private module for :mod:`aiperf.common.environment`. Contains the
``_APIServerSettings``, ``_CompressionSettings``, ``_HTTPSettings``,
``_LoggingSettings``, and ``_ZMQSettings`` classes. Split out to keep the
top-level ``environment`` module small.
"""

from aiperf.common.environment import _APIServerSettings as _APIServerSettings
from aiperf.common.environment import _CompressionSettings as _CompressionSettings
from aiperf.common.environment import _HTTPSettings as _HTTPSettings
from aiperf.common.environment import _LoggingSettings as _LoggingSettings
from aiperf.common.environment import _ZMQSettings as _ZMQSettings
