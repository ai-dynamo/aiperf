// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distribution census: the assembler turns a candidate generation into an
//! authenticated inventory document whose packages and digests are exactly the
//! ones the generation declared.
//!
//! These tests drive the same code path the `assemble-plugin-distribution`
//! binary drives, so the census a release candidate publishes is the census
//! the tests here pin.

use std::path::Path;

use aiperf_plugin_packaging_tests::assemble::{
    CandidateFixture, INVENTORY_FILE_NAME, assemble_distribution,
};
use aiperf_plugin_packaging_tests::inventory::PluginInventoryV1;

/// Path to the synthetic candidate generation checked in beside this suite.
fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/candidate-generation/fixture.toml")
}

/// Parse the fixture and materialize its synthetic artifacts into `dir`.
fn staged_fixture(dir: &Path) -> CandidateFixture {
    let text = std::fs::read_to_string(fixture_path()).expect("fixture is readable");
    let fixture = CandidateFixture::parse(&text).expect("fixture parses");
    fixture
        .materialize_synthetic_artifacts(dir)
        .expect("synthetic artifacts materialize");
    fixture
}

#[test]
fn distribution_json_is_valid_inventory() {
    let artifacts = tempfile::tempdir().expect("tempdir");
    let output = tempfile::tempdir().expect("tempdir");
    let fixture = staged_fixture(artifacts.path());

    let written = assemble_distribution(&fixture, artifacts.path(), output.path())
        .expect("the candidate generation assembles");
    assert_eq!(written, output.path().join(INVENTORY_FILE_NAME));

    // The published document must survive the host's own no-follow,
    // digest-authenticated read: assembly that cannot be verified is not a
    // distribution.
    let loaded = PluginInventoryV1::load_and_verify(&written).expect("published inventory verifies");
    assert_eq!(loaded.schema_version, 1);
    assert_eq!(loaded.generation, fixture.generation);
}

#[test]
fn distribution_census_has_expected_packages() {
    let artifacts = tempfile::tempdir().expect("tempdir");
    let output = tempfile::tempdir().expect("tempdir");
    let fixture = staged_fixture(artifacts.path());

    let written = assemble_distribution(&fixture, artifacts.path(), output.path())
        .expect("the candidate generation assembles");
    let loaded = PluginInventoryV1::load_and_verify(&written).expect("published inventory verifies");

    let census: Vec<&str> = loaded.packages.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(census, vec!["nvidia/export-basic", "nvidia/transport-http"]);

    // Every digest is the assembler's own hash of the staged bytes, not a value
    // copied from the fixture: a candidate cannot declare a digest its artifact
    // does not have.
    for declared in &fixture.packages {
        let published = loaded
            .packages
            .iter()
            .find(|p| p.id == declared.id)
            .unwrap_or_else(|| panic!("{} is published", declared.id));
        assert_eq!(published.version, declared.version);
        assert_eq!(published.build_id.as_deref(), Some(fixture.build_id.as_str()));

        let artifact_bytes =
            std::fs::read(artifacts.path().join(&declared.artifact)).expect("artifact staged");
        let expected = format!("blake3:{}", blake3::hash(&artifact_bytes).to_hex());
        assert_eq!(published.artifact_digest, expected);
    }
}
