// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen advertised-format and semantic-role identity.
//!
//! The parser implementation remains typed Rust, while this canonical
//! descriptor is the durable identity bound into telemetry archive genesis.
//! Changing any accepted family, wire role, timestamp unit, EOF rule,
//! exemplar owner, created role, or non-finite policy therefore requires a new
//! descriptor fingerprint and makes exact archive resume fail closed.

/// Canonical UTF-8 JSON for the v1 parser role-validity matrix.
pub const ROLE_VALIDITY_MATRIX_V1_BYTES: &[u8] =
    include_bytes!("../descriptors/role-validity-matrix-v1.json");

/// Return the exact canonical descriptor bytes used by durable identities.
///
/// The checked-in descriptor file carries a final newline (editors and
/// final-newline tooling re-add it), but the canonical-JSON representation
/// bound into archive identity admits no trailing whitespace. Trim it here so
/// every consumer sees the exact canonical bytes regardless of the file's
/// terminator.
#[must_use]
pub const fn role_validity_matrix_v1_bytes() -> &'static [u8] {
    ROLE_VALIDITY_MATRIX_V1_BYTES.trim_ascii_end()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_canonical_json_and_covers_both_advertised_formats() {
        let text = std::str::from_utf8(ROLE_VALIDITY_MATRIX_V1_BYTES).unwrap();
        let value: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(serde_json::to_string(&value).unwrap(), text.trim_end());
        assert_eq!(value["matrix_version"], 1);
        assert!(value["formats"]["prometheus_text_0_0_4"].is_object());
        assert!(value["formats"]["openmetrics_text_1_0_0"].is_object());
    }
}
