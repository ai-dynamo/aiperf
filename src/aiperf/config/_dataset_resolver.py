# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-dataset resolver.

Split out from ``resolvers.py`` to keep that module under the file-size limit.
Imported and re-exported by ``resolvers`` so callers and test patches that
reference ``aiperf.config.resolvers.DatasetResolver`` continue to work.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiperf.config.dataset.resolver import DatasetResolver as DatasetResolver
from aiperf.config.dataset.resolver import _DatasetResolution as _DatasetResolution

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _add_session_id(line: str, session_ids: set[str]) -> None:
    """Parse a JSONL line and add its session_id/chat_id to the set."""
    from aiperf.common.utils import load_json_str

    try:
        data = load_json_str(line)
    except (ValueError, TypeError):
        return
    sid = data.get("session_id") or data.get("chat_id")
    if sid is not None:
        session_ids.add(str(sid))
