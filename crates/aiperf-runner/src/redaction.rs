// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Credential-safe diagnostics for the strict runner process boundary.
//!
//! Factory errors may include authored URLs or serialized header fragments.
//! Every typed terminal/validation diagnostic passes through this module before
//! crossing stdout. A future remote coordinator can reuse the same function;
//! redaction is not coupled to the local subprocess writer.

use std::sync::LazyLock;

use regex::Regex;

const REDACTED: &str = "<redacted>";

static URL_USERINFO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)([a-z][a-z0-9+.\-]*://)[^\s'"@/?#]+@"#)
        .expect("URL-userinfo redaction regex is valid")
});

static BARE_USERINFO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(^|[\s'"(=])[^\s:@'"/?#]+:[^\s@'"/?#]+@"#)
        .expect("bare-userinfo redaction regex is valid")
});

static AUTHORIZATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)((?:proxy-)?authorization['"\s]*[:=]['"\s]*(?:bearer|basic)?\s*)[^'"}\n]+"#)
        .expect("Authorization redaction regex is valid")
});

static SENSITIVE_HEADER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)((?:x-api-key|api-key|ocp-apim-subscription-key|x-goog-api-key|x-functions-key|aeg-sas-key|x-amz-security-token)['"\s]*[:=]['"\s]*)[^\s,;}'"]+"#,
    )
    .expect("sensitive-header redaction regex is valid")
});

static SECRET_ASSIGNMENT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)\b(api[-_ ]?key|access[-_]?token|auth[-_]?token|bearer[-_]?token|token|secret|password)\s*=\s*[^&\s,;}'"]+"#,
    )
    .expect("secret-assignment redaction regex is valid")
});

static SECRET_OBJECT_FIELD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(['"]?(?:api[-_]?key|access[-_]?token|auth[-_]?token|bearer[-_]?token|token|secret|password)['"]?\s*:\s*['"]?)[^\s,;}'"]+"#,
    )
    .expect("secret-object-field redaction regex is valid")
});

/// Remove credential values from one user-visible runner diagnostic.
///
/// The transformations deliberately preserve component IDs, hosts, paths, and
/// non-sensitive header values so failures remain actionable. Applying the
/// function repeatedly is idempotent.
pub fn redact_diagnostic(value: impl AsRef<str>) -> String {
    let value = URL_USERINFO.replace_all(value.as_ref(), format!("$1{REDACTED}@"));
    let value = BARE_USERINFO.replace_all(&value, format!("$1{REDACTED}@"));
    let value = AUTHORIZATION.replace_all(&value, format!("$1{REDACTED}"));
    let value = SENSITIVE_HEADER.replace_all(&value, format!("$1{REDACTED}"));
    let value = SECRET_ASSIGNMENT.replace_all(&value, format!("$1={REDACTED}"));
    SECRET_OBJECT_FIELD
        .replace_all(&value, format!("$1{REDACTED}"))
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_urls_headers_and_structured_secret_fields() {
        let diagnostic = concat!(
            "connect https://user:password@host.test/v1; ",
            "fallback=user2:password2@other.test; ",
            "Authorization: Bearer auth-secret\n",
            "x-api-key: header-secret, api_key=query-secret&keep=yes; ",
            r#"{"token":"json-secret","requests":7}"#,
        );

        let redacted = redact_diagnostic(diagnostic);

        for secret in [
            "user:password",
            "user2:password2",
            "auth-secret",
            "header-secret",
            "query-secret",
            "json-secret",
        ] {
            assert!(!redacted.contains(secret), "leaked {secret:?}: {redacted}");
        }
        assert!(redacted.contains("https://<redacted>@host.test/v1"));
        assert!(redacted.contains("keep=yes"));
        assert!(redacted.contains(r#""requests":7"#));
    }

    #[test]
    fn preserves_actionable_non_secret_values_and_is_idempotent() {
        let diagnostic = "unknown endpoint chat; model=foo@bar; completion_tokens=12; https://host.test/users@example.com";
        let once = redact_diagnostic(diagnostic);
        assert_eq!(once, diagnostic);
        assert_eq!(redact_diagnostic(&once), once);
    }
}
