// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON-response template renderers.

use super::CodingCorpusError;
use super::templates::TemplateRenderer;
use super::vocab::*;

/// Dispatch across the JSON-shaped structural variants.
pub(super) fn render(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    match r.index(3)? {
        0 => object(r),
        1 => paginated(r),
        _ => error(r),
    }
}

fn object(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let v = r.sample(VARS, 3)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);
    let cls = r.pick(CLASSES)?;
    let cls_lower = cls.to_lowercase();
    let id_suffix = r.number(1000, 9999)?;
    let num_val = r.number(0, 1000)?;
    let float_val = r.uniform(0.0, 1.0);
    let ts = r.number(1_700_000_000, 1_800_000_000)?;
    let mut items: Vec<String> = Vec::new();
    for _ in 0..3 {
        let id = r.number(1, 999)?;
        let name = r.pick(VARS)?;
        items.push(format!(r#"      {{{{"id": {id}, "name": "{name}"}}}}"#));
    }
    let items_str = items.join(",\n");

    Ok(format!(
        r#"{{{{
  "status": "ok",
  "data": {{{{
    "{v1}": "{cls_lower}_{id_suffix}",
    "{v2}": {num_val},
    "{v3}": {float_val:.4},
    "metadata": {{{{
      "action": "{m1}",
      "source": "{m2}",
      "timestamp": "{ts}"
    }}}},
    "items": [
{items_str}
    ]
  }}}}
}}}}
"#
    ))
}

fn paginated(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let v = r.sample(VARS, 2)?;
    let (v1, v2) = (v[0], v[1]);
    let cls = r.pick(CLASSES)?;
    let cls_lower = cls.to_lowercase();
    let total = r.number(50, 5000)?;
    let page = r.number(1, 20)?;
    let per_page = [10_i64, 20, 50, 100][r.index(4)?];
    let mut items: Vec<String> = Vec::new();
    for _ in 0..per_page.min(5) {
        let id_n = r.number(1000, 9999)?;
        let mod_ = r.pick(MODULES)?;
        let val = r.number(0, 100)?;
        items.push(format!(
            r#"    {{{{"id": "{cls_lower}_{id_n}", "{v1}": "{mod_}", "{v2}": {val}}}}}"#
        ));
    }
    let items_str = items.join(",\n");
    let total_pages = (total + per_page - 1) / per_page;
    let has_next = if page * per_page < total {
        "true"
    } else {
        "false"
    };
    let has_prev = if page > 1 { "true" } else { "false" };

    Ok(format!(
        r#"{{{{
  "data": [
{items_str}
  ],
  "pagination": {{{{
    "page": {page},
    "per_page": {per_page},
    "total": {total},
    "total_pages": {total_pages},
    "has_next": {has_next},
    "has_prev": {has_prev}
  }}}}
}}}}
"#
    ))
}

fn error(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let err = r.pick(ERRORS)?;
    let status = r.pick(STATUS_CODES)?;
    let code = status.split_whitespace().next().unwrap_or(status);
    let a = r.number(100_000, 999_999)?;
    let b = r.number(100_000, 999_999)?;
    let trace_id = format!("{a:06x}-{b:06x}");
    let v1 = r.pick(VARS)?;
    let cls = r.pick(CLASSES)?;

    Ok(format!(
        r#"{{{{
  "error": {{{{
    "code": {code},
    "status": "{status}",
    "message": "{err}",
    "details": [
      {{{{
        "field": "{v1}",
        "reason": "{err}",
        "type": "{cls}"
      }}}}
    ],
    "trace_id": "{trace_id}",
    "documentation_url": "https://docs.example.com/errors/{code}"
  }}}}
}}}}
"#
    ))
}
