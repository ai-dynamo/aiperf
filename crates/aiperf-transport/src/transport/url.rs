// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! URL construction. Port of `AioHttpTransport.get_url` / `_dedup_path_overlap`
//! and `BaseTransport.build_url` query merge.

use std::collections::BTreeMap;

use url::Url;

fn has_http_scheme(u: &str) -> bool {
    let l = u.to_ascii_lowercase();
    l.starts_with("http://") || l.starts_with("https://")
}

/// Join `base_path` and `sub_path`, collapsing tail/head overlap.
pub fn dedup_path_overlap(base_path: &str, sub_path: &str) -> String {
    if sub_path.is_empty() {
        return base_path.to_string();
    }
    if base_path.ends_with(&format!("/{sub_path}")) {
        return base_path.to_string();
    }
    let sub_path = if base_path.ends_with("/v1") && sub_path.starts_with("v1/") {
        sub_path.strip_prefix("v1/").unwrap()
    } else {
        sub_path
    };
    format!("{base_path}/{sub_path}")
}

/// Build a full URL from a base, an endpoint sub-path, and endpoint query params.
///
/// Returns `Err` (rather than panicking) when `base` is not a parseable URL.
pub fn build_url(
    base: &str,
    sub_path: &str,
    params: &BTreeMap<String, String>,
) -> Result<String, url::ParseError> {
    let raw = if has_http_scheme(base) {
        base.to_string()
    } else {
        format!("http://{base}")
    };
    let mut parsed = Url::parse(&raw)?;

    let base_path = parsed.path().trim_end_matches('/').to_string();
    let sub = sub_path.trim_start_matches('/');
    let new_path = dedup_path_overlap(&base_path, sub);
    parsed.set_path(&new_path);

    if !params.is_empty() {
        // Preserve existing params, then let endpoint params override.
        let existing: Vec<(String, String)> = parsed
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        let mut merged: BTreeMap<String, String> = existing.into_iter().collect();
        for (k, v) in params {
            merged.insert(k.clone(), v.clone());
        }
        parsed.query_pairs_mut().clear().extend_pairs(merged.iter());
    }
    Ok(parsed.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn no_params() -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    #[test]
    fn adds_scheme_when_missing() {
        assert_eq!(
            build_url("localhost:8000", "v1/chat/completions", &no_params()).unwrap(),
            "http://localhost:8000/v1/chat/completions"
        );
    }

    #[test]
    fn dedups_v1_prefix() {
        assert_eq!(
            dedup_path_overlap("/v1", "v1/chat/completions"),
            "/v1/chat/completions"
        );
    }

    #[test]
    fn dedups_full_suffix_already_present() {
        assert_eq!(
            dedup_path_overlap("/v1/chat/completions", "v1/chat/completions"),
            "/v1/chat/completions"
        );
    }

    #[test]
    fn empty_sub_path_returns_base() {
        assert_eq!(dedup_path_overlap("/foo", ""), "/foo");
    }

    #[test]
    fn merges_query_params_endpoint_overrides() {
        let mut p = BTreeMap::new();
        p.insert("b".to_string(), "2".to_string());
        let url = build_url("http://h/base?a=1", "sub", &p).unwrap();
        // existing a=1 preserved, endpoint b=2 added
        assert!(url.starts_with("http://h/base/sub?"));
        assert!(url.contains("a=1"));
        assert!(url.contains("b=2"));
    }
}
