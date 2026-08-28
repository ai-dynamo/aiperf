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
    CandidateFixture, INVENTORY_FILE_NAME, assemble_distribution, next_generation,
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
    let loaded =
        PluginInventoryV1::load_and_verify(&written).expect("published inventory verifies");
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
    let loaded =
        PluginInventoryV1::load_and_verify(&written).expect("published inventory verifies");

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
        assert_eq!(
            published.build_id.as_deref(),
            Some(fixture.build_id.as_str())
        );

        let artifact_bytes =
            std::fs::read(artifacts.path().join(&declared.artifact)).expect("artifact staged");
        let expected = format!("blake3:{}", blake3::hash(&artifact_bytes).to_hex());
        assert_eq!(published.artifact_digest, expected);
    }
}

#[test]
fn synthetic_materialization_refuses_to_replace_a_staged_artifact() {
    let artifacts = tempfile::tempdir().expect("tempdir");
    let text = std::fs::read_to_string(fixture_path()).expect("fixture is readable");
    let fixture = CandidateFixture::parse(&text).expect("fixture parses");

    // A real build product is already staged where the synthetic bytes would go.
    let staged = artifacts.path().join(&fixture.packages[0].artifact);
    std::fs::write(&staged, b"real-build-product").expect("stage a build product");

    let refusal = fixture
        .materialize_synthetic_artifacts(artifacts.path())
        .expect_err("synthetic bytes never replace a staged artifact");
    assert!(
        refusal.to_string().contains("refuses to overwrite"),
        "unexpected refusal: {refusal}"
    );
    assert_eq!(
        std::fs::read(&staged).expect("staged bytes survive"),
        b"real-build-product"
    );
}

#[cfg(unix)]
#[test]
fn a_symlinked_artifact_is_refused_rather_than_hashed_through() {
    let artifacts = tempfile::tempdir().expect("tempdir");
    let outside = tempfile::tempdir().expect("tempdir");
    let output = tempfile::tempdir().expect("tempdir");
    let fixture = staged_fixture(artifacts.path());

    // A plain, legal file name that is actually a link out of the artifacts
    // directory: the name check passes, so only the no-follow open can refuse.
    let secret = outside.path().join("secret.bin");
    std::fs::write(&secret, b"outside-the-artifacts-dir").expect("write outside file");
    let planted = artifacts.path().join(&fixture.packages[0].artifact);
    std::fs::remove_file(&planted).expect("clear the staged artifact");
    std::os::unix::fs::symlink(&secret, &planted).expect("plant the symlink");

    assemble_distribution(&fixture, artifacts.path(), output.path())
        .expect_err("a symlinked artifact is never hashed");
}

#[test]
fn auto_generation_advances_but_refuses_an_unverifiable_prior_inventory() {
    let artifacts = tempfile::tempdir().expect("tempdir");
    let output = tempfile::tempdir().expect("tempdir");
    let fixture = staged_fixture(artifacts.path());

    // Nothing published yet: the declared generation stands.
    assert_eq!(
        next_generation(output.path(), fixture.generation)
            .expect("an absent prior is not an error"),
        fixture.generation
    );

    let written = assemble_distribution(&fixture, artifacts.path(), output.path())
        .expect("the candidate generation assembles");
    assert_eq!(
        next_generation(output.path(), fixture.generation).expect("a verifiable prior advances"),
        fixture.generation + 1
    );

    // A tampered document must surface as a refusal. Falling back to the
    // fixture's own generation here would republish a lower generation, which
    // the install side accepts as a downgrade instead of an integrity failure.
    std::fs::write(&written, b"{not-an-inventory").expect("tamper with the prior document");
    let refusal = next_generation(output.path(), fixture.generation)
        .expect_err("an unverifiable prior is never answered with a lower generation");
    assert!(
        refusal.to_string().contains("cannot be verified"),
        "unexpected refusal: {refusal}"
    );
}
