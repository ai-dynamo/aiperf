# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import subprocess


def _new_process_group_kwargs(
    *, supports_process_group: bool | None = None
) -> dict[str, int | bool]:
    if supports_process_group is None:
        supports_process_group = (
            "process_group" in inspect.signature(subprocess.Popen).parameters
        )
    if supports_process_group:
        return {"process_group": 0}
    return {"start_new_session": True}
