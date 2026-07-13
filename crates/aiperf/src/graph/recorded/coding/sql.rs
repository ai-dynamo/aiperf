// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SQL query template renderers.

use super::templates::TemplateRenderer;
use super::vocab::*;
use crate::graph::recorded::RecordedTraceError;

/// `_gen_sql_query`: dispatch across the SQL statement variants.
pub(super) fn query(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let t = r.sample(TABLES, 3)?;
    let (t1, t2, t3) = (t[0], t[1], t[2]);
    let v = r.sample(VARS, 3)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);
    match r.index(4)? {
        0 => select_join(r, t1, t2, t3, v1, v2, v3),
        1 => insert(r, t1, v1, v2),
        2 => create(t1, v1, v2, v3),
        _ => alter(r, t1, t2, v1, v2),
    }
}

#[allow(clippy::too_many_arguments)]
fn select_join(
    r: &mut TemplateRenderer,
    t1: &str,
    t2: &str,
    t3: &str,
    v1: &str,
    v2: &str,
    v3: &str,
) -> Result<String, RecordedTraceError> {
    let limit = r.number(10, 1000)?;
    let offset = r.number(0, 500)?;
    Ok(format!(
        r#"SELECT
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
"#
    ))
}

fn insert(
    r: &mut TemplateRenderer,
    t1: &str,
    v1: &str,
    v2: &str,
) -> Result<String, RecordedTraceError> {
    let n_rows = r.number(1, 5)?;
    let mut rows: Vec<String> = Vec::new();
    for _ in 0..n_rows {
        let mod_ = r.pick(MODULES)?;
        let n = r.number(1, 1000)?;
        let status = r.pick(STATUS_CODES)?;
        let code = status.split_whitespace().next().unwrap_or(status);
        rows.push(format!("    ('{mod_}', {n}, '{code}', NOW())"));
    }
    let rows_str = rows.join(",\n");
    Ok(format!(
        r#"INSERT INTO {t1} ({v1}, {v2}, status, created_at)
VALUES
{rows_str}
ON CONFLICT ({v1})
DO UPDATE SET
    {v2} = EXCLUDED.{v2},
    status = EXCLUDED.status,
    updated_at = NOW()
RETURNING id, {v1}, {v2};
"#
    ))
}

fn create(t1: &str, v1: &str, v2: &str, v3: &str) -> Result<String, RecordedTraceError> {
    Ok(format!(
        r#"CREATE TABLE IF NOT EXISTS {t1} (
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
"#
    ))
}

fn alter(
    r: &mut TemplateRenderer,
    t1: &str,
    t2: &str,
    v1: &str,
    v2: &str,
) -> Result<String, RecordedTraceError> {
    let col_types = ["VARCHAR(256)", "INTEGER", "BOOLEAN", "JSONB", "TIMESTAMPTZ"];
    let col_type = col_types[r.index(5)?];
    Ok(format!(
        r#"BEGIN;

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
"#
    ))
}
