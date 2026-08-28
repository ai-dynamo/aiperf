// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical digest computation for plugin identity records.
//!
//! Each record type feeds its fields into a BLAKE3 hasher in declaration order,
//! using length-prefixed encoding so that field boundaries are unambiguous.
//! The `canonical_digest` field itself is excluded from its own computation.

use blake3::Hasher;

/// Feeds one named field into a canonical hasher.
///
/// Encoding: 8-byte LE name-length, name bytes, 8-byte LE value-length, value bytes.
pub fn feed_field(hasher: &mut Hasher, name: &str, value: &[u8]) {
    hasher.update(&(name.len() as u64).to_le_bytes());
    hasher.update(name.as_bytes());
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

/// Feeds a string field.
pub fn feed_str(hasher: &mut Hasher, name: &str, value: &str) {
    feed_field(hasher, name, value.as_bytes());
}

/// Feeds an optional string field.
///
/// Encoding: 1-byte discriminant (`0x00` = absent, `0x01` = present),
/// followed by the standard length-prefixed field bytes when present.
/// Using a discriminant prefix ensures `None` and `Some("null")` hash
/// to different values.
pub fn feed_opt_str(hasher: &mut Hasher, name: &str, value: Option<&str>) {
    match value {
        None => {
            // Absent: discriminant only, no value bytes.
            hasher.update(&(name.len() as u64).to_le_bytes());
            hasher.update(name.as_bytes());
            hasher.update(&[0x00]);
        }
        Some(s) => {
            // Present: discriminant prefix then normal field encoding.
            hasher.update(&(name.len() as u64).to_le_bytes());
            hasher.update(name.as_bytes());
            hasher.update(&[0x01]);
            hasher.update(&(s.len() as u64).to_le_bytes());
            hasher.update(s.as_bytes());
        }
    }
}

/// Feeds a u8 field.
pub fn feed_u8(hasher: &mut Hasher, name: &str, value: u8) {
    feed_field(hasher, name, &[value]);
}

/// Feeds a u32 field (little-endian bytes).
pub fn feed_u32(hasher: &mut Hasher, name: &str, value: u32) {
    feed_field(hasher, name, &value.to_le_bytes());
}

/// Feeds an arbitrary JSON-serializable value (vecs, structs, etc.).
pub fn feed_json<T: serde::Serialize + ?Sized>(hasher: &mut Hasher, name: &str, value: &T) {
    let json = serde_json::to_vec(value).expect("record serialization cannot fail");
    feed_field(hasher, name, &json);
}

/// Finalizes the hasher and returns the lowercase hex digest string.
pub fn finalize_hex(hasher: Hasher) -> String {
    hasher.finalize().to_hex().to_string()
}
