// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error vocabulary for the native Kubernetes client and v1 contract.

use std::fmt;

/// Failure while loading Kubernetes credentials, constructing a request, or
/// validating a native Kubernetes contract.
#[derive(Debug)]
pub enum KubeError {
    /// The supplied document names a contract not supported by this binary.
    UnsupportedContractVersion(String),
    /// A JSON document does not satisfy its versioned schema.
    ContractValidation(String),
    /// A configuration or credential source could not be read.
    Io(std::io::Error),
    /// A kubeconfig or contract document could not be decoded.
    Decode(String),
    /// A credential source is incomplete or mutually inconsistent.
    Authentication(String),
    /// TLS configuration could not be constructed.
    Tls(String),
    /// A request could not be built or delivered.
    Transport(String),
}

impl fmt::Display for KubeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedContractVersion(version) => {
                write!(
                    formatter,
                    "unsupported Kubernetes contract version {version}"
                )
            }
            Self::ContractValidation(message) => write!(
                formatter,
                "Kubernetes contract validation failed: {message}"
            ),
            Self::Io(error) => write!(formatter, "Kubernetes I/O failed: {error}"),
            Self::Decode(message) => {
                write!(formatter, "Kubernetes document decode failed: {message}")
            }
            Self::Authentication(message) => {
                write!(formatter, "Kubernetes authentication failed: {message}")
            }
            Self::Tls(message) => {
                write!(formatter, "Kubernetes TLS configuration failed: {message}")
            }
            Self::Transport(message) => write!(formatter, "Kubernetes transport failed: {message}"),
        }
    }
}

impl std::error::Error for KubeError {}

impl From<std::io::Error> for KubeError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
