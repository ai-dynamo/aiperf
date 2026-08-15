// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic reward parsing with `reward.json` precedence.

use std::{
    collections::BTreeMap,
    fmt::{self, Display, Formatter},
};

use serde::{Deserialize, Serialize};

/// Finite named reward metrics from a verifier artifact.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RewardDocument {
    /// Named finite reward metrics.
    pub metrics: BTreeMap<String, f64>,
}

impl RewardDocument {
    /// Parses finite metrics from `reward.json`, falling back to `reward.txt` only when absent.
    pub fn parse(
        reward_json: Option<&[u8]>,
        reward_txt: Option<&[u8]>,
    ) -> Result<Self, RewardError> {
        let metrics = match reward_json {
            Some(bytes) => serde_json::from_slice::<BTreeMap<String, f64>>(bytes)
                .map_err(|error| RewardError::InvalidJson(error.to_string()))?,
            None => parse_text_reward(reward_txt.ok_or(RewardError::Absent)?)?,
        };
        Self::new(metrics)
    }

    /// Creates a verified finite nonempty reward document.
    pub fn new(metrics: BTreeMap<String, f64>) -> Result<Self, RewardError> {
        if metrics.is_empty()
            || metrics
                .iter()
                .any(|(name, value)| name.trim().is_empty() || !value.is_finite())
        {
            return Err(RewardError::NonFiniteOrEmpty);
        }
        Ok(Self { metrics })
    }
}

fn parse_text_reward(bytes: &[u8]) -> Result<BTreeMap<String, f64>, RewardError> {
    let text =
        std::str::from_utf8(bytes).map_err(|error| RewardError::InvalidText(error.to_string()))?;
    let value = text
        .trim()
        .parse::<f64>()
        .map_err(|error| RewardError::InvalidText(error.to_string()))?;
    Ok(BTreeMap::from([("reward".to_owned(), value)]))
}

/// Malformed, missing, or non-finite verifier reward artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RewardError {
    /// Neither reward artifact existed.
    Absent,
    /// The preferred JSON document was malformed.
    InvalidJson(String),
    /// The fallback text reward was malformed.
    InvalidText(String),
    /// A reward document was empty or contained a non-finite metric.
    NonFiniteOrEmpty,
}

impl Display for RewardError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absent => formatter.write_str("verifier reward is absent"),
            Self::InvalidJson(error) => write!(formatter, "invalid reward.json: {error}"),
            Self::InvalidText(error) => write!(formatter, "invalid reward.txt: {error}"),
            Self::NonFiniteOrEmpty => {
                formatter.write_str("reward metrics must be finite and nonempty")
            }
        }
    }
}

impl std::error::Error for RewardError {}
