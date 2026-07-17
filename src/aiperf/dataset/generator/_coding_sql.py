# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL query generator (mixin for CodingContentGenerator).

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from aiperf.dataset.generator._coding_vocab import (
    _DB_TABLES,
    _MODULES,
    _STATUS_CODES,
    _VARS,
)


class _SqlMixin:
    def _gen_sql_query(self) -> str:
        r = self._template_rng
        t1, t2, t3 = r.sample(list(_DB_TABLES), 3)
        v1, v2, v3 = r.sample(list(_VARS), 3)
        kind = r.choice(["select_join", "insert", "create", "alter"])

        if kind == "select_join":
            return self._sql_select_join(t1, t2, t3, v1=v1, v2=v2, v3=v3)
        if kind == "insert":
            return self._sql_insert(t1, v1, v2)
        if kind == "create":
            return self._sql_create(t1, v1, v2, v3)
        return self._sql_alter(t1, t2, v1, v2)

    def _sql_select_join(
        self,
        t1: str,
        t2: str,
        t3: str,
        *,
        v1: str,
        v2: str,
        v3: str,
    ) -> str:
        r = self._template_rng
        limit = r.randint(10, 1000)
        offset = r.randint(0, 500)
        return f"""\
SELECT
    t1.id,
    t1.{v1},
    t1.created_at,
    t2.{v2},
    t2.{v3},
    COUNT(t3.id) AS {v3}_count
FROM {t1} t1
INNER JOIN {t2} t2 ON t2.{t1}_id = t1.id
LEFT JOIN {t3} t3 ON t3.{t2}_id = t2.id
WHERE t1.status = 'active'
  AND t1.created_at >= NOW() - INTERVAL '30 days'
  AND t2.{v2} IS NOT NULL
GROUP BY t1.id, t1.{v1}, t1.created_at, t2.{v2}, t2.{v3}
HAVING COUNT(t3.id) > 0
ORDER BY t1.created_at DESC
LIMIT {limit} OFFSET {offset};
"""

    def _sql_insert(self, t1: str, v1: str, v2: str) -> str:
        r = self._template_rng
        n_rows = r.randint(1, 5)
        rows = []
        for _ in range(n_rows):
            rows.append(
                f"    ('{r.choice(_MODULES)}', {r.randint(1, 1000)}, "
                f"'{r.choice(_STATUS_CODES).split()[0]}', NOW())"
            )
        rows_str = ",\n".join(rows)
        return f"""\
INSERT INTO {t1} ({v1}, {v2}, status, created_at)
VALUES
{rows_str}
ON CONFLICT ({v1})
DO UPDATE SET
    {v2} = EXCLUDED.{v2},
    status = EXCLUDED.status,
    updated_at = NOW()
RETURNING id, {v1}, {v2};
"""

    @staticmethod
    def _sql_create(t1: str, v1: str, v2: str, v3: str) -> str:
        return f"""\
CREATE TABLE IF NOT EXISTS {t1} (
    id BIGSERIAL PRIMARY KEY,
    {v1} VARCHAR(256) NOT NULL,
    {v2} INTEGER DEFAULT 0,
    {v3} JSONB DEFAULT '{{}}'::jsonb,
    status VARCHAR(32) DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT {t1}_{v1}_unique UNIQUE ({v1})
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{t1}_{v1}
    ON {t1} ({v1});
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{t1}_status_created
    ON {t1} (status, created_at DESC);
"""

    def _sql_alter(self, t1: str, t2: str, v1: str, v2: str) -> str:
        r = self._template_rng
        col_type = r.choice(
            ["VARCHAR(256)", "INTEGER", "BOOLEAN", "JSONB", "TIMESTAMPTZ"]
        )
        return f"""\
BEGIN;

ALTER TABLE {t1}
    ADD COLUMN IF NOT EXISTS {v1} {col_type},
    ADD COLUMN IF NOT EXISTS {v2} INTEGER DEFAULT 0;

UPDATE {t1}
SET {v1} = (
    SELECT {v2} FROM {t2}
    WHERE {t2}.{t1}_id = {t1}.id
    LIMIT 1
)
WHERE {v1} IS NULL;

ALTER TABLE {t1}
    ALTER COLUMN {v1} SET NOT NULL;

COMMIT;
"""
