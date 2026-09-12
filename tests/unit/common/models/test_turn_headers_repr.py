# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.models import Turn


class TestTurnHeadersRepr:
    def test_secret_headers_hidden_from_repr(self):
        t = Turn(extra_headers={"Authorization": "Bearer secret"})
        assert "Bearer secret" not in str(t)
        assert "Bearer secret" not in repr(t)

    def test_headers_still_preserved_in_model_dump(self):
        t = Turn(extra_headers={"Authorization": "Bearer secret"})
        assert t.model_dump()["extra_headers"] == {"Authorization": "Bearer secret"}
