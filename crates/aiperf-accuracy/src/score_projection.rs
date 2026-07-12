// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory-owned validation for report-safe provider score projections.
//!
//! Provider-native score trees remain restricted. A value can enter the
//! public native report only through a named validator registered by the
//! selected provider factory and bound to an immutable schema fingerprint.

use std::collections::BTreeMap;
use std::fmt::{self, Display};
use std::sync::Arc;

use crate::canonical::{
    CanonicalJson, is_sha256, redact_diagnostic, validate_no_secret_control_value,
};

/// Factory-owned validator for one reviewed public score projection schema.
pub trait PublicScoreProjectionValidator: Send + Sync {
    /// Exact immutable schema fingerprint advertised by the provider descriptor.
    fn schema_sha256(&self) -> &str;

    /// Validate one already bounded canonical provider projection.
    fn validate(&self, value: &CanonicalJson) -> Result<(), PublicScoreProjectionError>;
}

/// Stable projection ID for the reviewed object-wrapped GSM8K binary score.
pub const GSM8K_BINARY_SCORE_PROJECTION_ID: &str = "gsm8k_binary_score_v1";

/// SHA-256 of the canonical object-wrapped GSM8K binary JSON Schema.
pub const GSM8K_BINARY_SCORE_SCHEMA_SHA256: &str =
    "d156e6577305139bac7f48946996fa35d489a381a87bce4c58d18c47d8d9eeb5";

/// Executable validator for the stock NeMo/OpenBench binary score object.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gsm8kBinaryScoreProjectionValidator;

impl PublicScoreProjectionValidator for Gsm8kBinaryScoreProjectionValidator {
    fn schema_sha256(&self) -> &str {
        GSM8K_BINARY_SCORE_SCHEMA_SHA256
    }

    fn validate(&self, value: &CanonicalJson) -> Result<(), PublicScoreProjectionError> {
        let Some(object) = value.value().as_object() else {
            return Err(PublicScoreProjectionError::rejected(
                "expected the reviewed GSM8K binary score object",
            ));
        };
        let Some(number) = object
            .get("value")
            .filter(|_| object.len() == 1)
            .and_then(serde_json::Value::as_f64)
        else {
            return Err(PublicScoreProjectionError::rejected(
                "expected exactly one finite numeric value field",
            ));
        };
        if number != 0.0 && number != 1.0 {
            return Err(PublicScoreProjectionError::rejected(
                "GSM8K public score was not exactly zero or one",
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct PublicScoreProjectionRule {
    validator: Arc<dyn PublicScoreProjectionValidator>,
}

impl fmt::Debug for PublicScoreProjectionRule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicScoreProjectionRule")
            .field("schema_sha256", &self.validator.schema_sha256())
            .finish()
    }
}

/// Deterministic, factory-owned public score projection policy.
#[derive(Clone, Default)]
pub struct PublicScoreProjectionPolicy {
    rules: BTreeMap<String, PublicScoreProjectionRule>,
}

impl PublicScoreProjectionPolicy {
    /// Construct a policy under which every provider-native score stays restricted.
    pub fn restricted_only() -> Self {
        Self::default()
    }

    /// Register one exact score name and reviewed schema validator.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        validator: Arc<dyn PublicScoreProjectionValidator>,
    ) -> Result<&mut Self, PublicScoreProjectionError> {
        let name = name.into();
        if name.trim().is_empty()
            || name != name.trim()
            || name.len() > 256
            || name.chars().any(char::is_control)
            || !is_sha256(validator.schema_sha256())
            || self.rules.contains_key(&name)
        {
            return Err(PublicScoreProjectionError::Policy(
                "public score projection name/schema was empty, mutable, oversized, or duplicated"
                    .to_string(),
            ));
        }
        self.rules
            .insert(name, PublicScoreProjectionRule { validator });
        Ok(self)
    }

    /// Validate and authorize one provider-authored public projection.
    pub fn validate<'a>(
        &'a self,
        name: &str,
        value: &CanonicalJson,
    ) -> Result<&'a str, PublicScoreProjectionError> {
        let rule = self.rules.get(name).ok_or_else(|| {
            PublicScoreProjectionError::Policy(
                "provider exposed an unregistered public score projection".to_string(),
            )
        })?;
        validate_no_secret_control_value(value)
            .map_err(|error| PublicScoreProjectionError::Projection(error.to_string()))?;
        rule.validator.validate(value)?;
        Ok(rule.validator.schema_sha256())
    }

    /// Return immutable fingerprints in deterministic score-name order.
    pub fn schema_fingerprints(&self) -> BTreeMap<String, String> {
        self.rules
            .iter()
            .map(|(name, rule)| (name.clone(), rule.validator.schema_sha256().to_string()))
            .collect()
    }

    /// Prove descriptor metadata exactly matches executable validator policy.
    pub fn validate_descriptor_fingerprints(
        &self,
        fingerprints: &BTreeMap<String, String>,
    ) -> Result<(), PublicScoreProjectionError> {
        if &self.schema_fingerprints() != fingerprints {
            return Err(PublicScoreProjectionError::Policy(
                "provider public score schema descriptors did not match factory validators"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Whether no provider score is eligible for public projection.
    pub fn is_restricted_only(&self) -> bool {
        self.rules.is_empty()
    }
}

impl fmt::Debug for PublicScoreProjectionPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicScoreProjectionPolicy")
            .field("schema_fingerprints", &self.schema_fingerprints())
            .finish()
    }
}

impl PartialEq for PublicScoreProjectionPolicy {
    fn eq(&self, other: &Self) -> bool {
        self.schema_fingerprints() == other.schema_fingerprints()
    }
}

impl Eq for PublicScoreProjectionPolicy {}

/// Factory policy or provider projection validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicScoreProjectionError {
    /// Factory registration/descriptor mismatch or unknown public score name.
    Policy(String),
    /// Provider-authored value did not satisfy the registered schema.
    Projection(String),
}

impl PublicScoreProjectionError {
    /// Construct a redacted schema rejection from a validator implementation.
    pub fn rejected(message: impl AsRef<str>) -> Self {
        Self::Projection(redact_diagnostic(message.as_ref()))
    }
}

impl Display for PublicScoreProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy(message) => write!(formatter, "public score projection policy: {message}"),
            Self::Projection(message) => {
                write!(formatter, "public score projection was rejected: {message}")
            }
        }
    }
}

impl std::error::Error for PublicScoreProjectionError {}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::canonical::sha256_hex;

    struct ZeroOrOne;

    impl PublicScoreProjectionValidator for ZeroOrOne {
        fn schema_sha256(&self) -> &str {
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        }

        fn validate(&self, value: &CanonicalJson) -> Result<(), PublicScoreProjectionError> {
            match value.value().as_u64() {
                Some(0 | 1) => Ok(()),
                _ => Err(PublicScoreProjectionError::rejected(
                    "expected the reviewed zero-or-one scalar schema",
                )),
            }
        }
    }

    #[test]
    fn policy_requires_registered_schema_and_validated_secret_free_value() {
        let mut policy = PublicScoreProjectionPolicy::restricted_only();
        policy.register("accuracy", Arc::new(ZeroOrOne)).unwrap();
        let zero = CanonicalJson::new(json!(0)).unwrap();
        assert_eq!(policy.validate("accuracy", &zero).unwrap(), "a".repeat(64));
        assert!(
            policy
                .validate("accuracy", &CanonicalJson::new(json!(2)).unwrap())
                .is_err()
        );
        assert!(policy.validate("unknown", &zero).is_err());
        assert!(
            policy
                .validate(
                    "accuracy",
                    &CanonicalJson::new(json!({"token": "secret"})).unwrap(),
                )
                .is_err()
        );
    }

    #[test]
    fn descriptor_fingerprints_must_exactly_match_executable_policy() {
        let mut policy = PublicScoreProjectionPolicy::restricted_only();
        policy.register("accuracy", Arc::new(ZeroOrOne)).unwrap();
        policy
            .validate_descriptor_fingerprints(&BTreeMap::from([(
                "accuracy".to_string(),
                "a".repeat(64),
            )]))
            .unwrap();
        assert!(
            policy
                .validate_descriptor_fingerprints(&BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn object_binary_validator_matches_stock_manifest_schema_exactly() {
        let schema = CanonicalJson::new(json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "additionalProperties": false,
            "properties": {
                "value": {
                    "enum": [0, 1],
                    "type": "number",
                },
            },
            "required": ["value"],
            "type": "object",
        }))
        .unwrap();
        assert_eq!(
            sha256_hex(&schema.to_bytes()),
            GSM8K_BINARY_SCORE_SCHEMA_SHA256
        );

        let validator = Gsm8kBinaryScoreProjectionValidator;
        assert_eq!(validator.schema_sha256(), GSM8K_BINARY_SCORE_SCHEMA_SHA256);
        for value in [json!({"value": 0}), json!({"value": 1})] {
            validator
                .validate(&CanonicalJson::new(value).unwrap())
                .unwrap();
        }
        for value in [
            json!(1),
            json!({"value": 0.5}),
            json!({"value": -1}),
            json!({"value": 2}),
            json!({"value": true}),
            json!({"value": 1, "answer": "hidden"}),
        ] {
            assert!(
                validator
                    .validate(&CanonicalJson::new(value).unwrap())
                    .is_err()
            );
        }
    }

    #[test]
    fn unknown_projection_diagnostic_does_not_echo_provider_label() {
        let policy = PublicScoreProjectionPolicy::restricted_only();
        let sentinel = "private-answer-score-42";
        let error = policy
            .validate(sentinel, &CanonicalJson::new(json!({"value": 1})).unwrap())
            .unwrap_err()
            .to_string();
        assert!(!error.contains(sentinel));
    }
}
