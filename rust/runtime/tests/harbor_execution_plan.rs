// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Pure contracts for normalized Harbor benchmark execution plans.

use aiperf_runtime::eval::{EnvBinding, NetworkPolicy};

#[test]
fn normalizes_equivalent_allowlist_spelling_deterministically() {
    let policy =
        NetworkPolicy::allowlist(["EXAMPLE.com", "*.Example.ORG", "2001:DB8::1", "10.0.0.0/24"])
            .unwrap();

    assert_eq!(
        policy,
        NetworkPolicy::Allowlist {
            allowed_hosts: vec![
                "*.example.org".to_owned(),
                "10.0.0.0/24".to_owned(),
                "2001:db8::1".to_owned(),
                "example.com".to_owned(),
            ],
        }
    );
}

#[test]
fn environment_binding_never_captures_the_host_secret_value() {
    let binding = EnvBinding::parse("${HOST_API_TOKEN}").unwrap();

    assert_eq!(
        binding,
        EnvBinding::SecretReference("HOST_API_TOKEN".to_owned())
    );
    assert!(!format!("{binding:?}").contains("super-secret"));
}
