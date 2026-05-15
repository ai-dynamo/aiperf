# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""--num-conversations autodefault for dag_jsonl input.

For DAG-shaped (forking) datasets, ``--request-count`` is a literal
wire-request cap that includes fork-spawned children, so the generic
``concurrency * MULT`` default would silently truncate the DAG mid-tree.
Instead, default ``--num-conversations`` to the *root* count (sessions
not referenced by any fork list) and refuse to default
``--request-count``.

The original test body (written against v1 UserConfig autodefault
validators) needs porting to the v2 resolver chain. The
``_count_dag_root_entries`` helper logic should land on the v2
DatasetResolver so the file-reading I/O stays out of the config model.
"""

import pytest

pytest.skip(
    "v1 UserConfig autodefault API removed in v2 refactor; equivalent "
    "num-conversations defaulting for DAG datasets needs to be ported to "
    "the v2 resolver chain. Port pending.",
    allow_module_level=True,
)
