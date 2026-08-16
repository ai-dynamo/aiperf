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
        policy
            .allowed_hosts()
            .unwrap()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["*.example.org", "10.0.0.0/24", "2001:db8::1", "example.com"]
    );
}

#[test]
fn environment_binding_never_captures_the_host_secret_value() {
    let binding = EnvBinding::parse("${HOST_API_TOKEN}").unwrap();

    assert_eq!(binding.secret_reference(), Some("HOST_API_TOKEN"));
    assert!(!format!("{binding:?}").contains("super-secret"));
}
