// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use aiperf_runtime::eval::{
    ArtifactDigest, HarborImporter, HarborSource, ImportDisposition, SourceAcquirer,
};

#[derive(Default)]
struct MemoryAcquirer {
    packages: BTreeMap<String, Vec<u8>>,
}

impl SourceAcquirer for MemoryAcquirer {
    fn acquire(&self, source: &HarborSource) -> Result<Vec<u8>, aiperf_runtime::eval::HarborImportError> {
        self.packages
            .get(source.location())
            .cloned()
            .ok_or_else(|| aiperf_runtime::eval::HarborImportError::Unavailable(source.location().to_owned()))
    }
}

#[test]
fn local_import_preserves_source_digest_and_normalizes_task() {
    let bytes = br#"{
        "id":"repair-1",
        "instruction":"Fix the failing test",
        "environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    }"#
    .to_vec();
    let source = HarborSource::local("fixtures/repair-1").unwrap();
    let mut acquirer = MemoryAcquirer::default();
    acquirer.packages.insert(source.location().to_owned(), bytes.clone());

    let imported = HarborImporter::new(&acquirer).import(&source).unwrap();

    assert_eq!(imported.report.source_digest, ArtifactDigest::from_bytes(&bytes));
    assert_eq!(imported.report.disposition, ImportDisposition::LosslessNormalized);
    assert_eq!(imported.task.id.as_str(), "repair-1");
}

#[test]
fn unsupported_semantics_return_report_before_provisioning() {
    let source = HarborSource::local("fixtures/unsupported").unwrap();
    let mut acquirer = MemoryAcquirer::default();
    acquirer.packages.insert(
        source.location().to_owned(),
        br#"{"id":"repair-1","unsupported_semantics":"sidecar"}"#.to_vec(),
    );

    let refusal = HarborImporter::new(&acquirer).import(&source).unwrap_err();

    assert_eq!(refusal.disposition(), Some(ImportDisposition::Unsupported));
}
