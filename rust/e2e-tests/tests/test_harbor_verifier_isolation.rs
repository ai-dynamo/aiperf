// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::collections::BTreeMap;

use aiperf_runtime::eval::{
    ArtifactDigest, DeclaredArtifactTransfer, VerifierExecutionError, VerifierMode,
    VerifierSandboxFactory, prepare_verifier,
};

fn digest(seed: char) -> ArtifactDigest {
    ArtifactDigest::parse(format!("blake3:{}", seed.to_string().repeat(64))).unwrap()
}

struct IsolatedVerifier {
    files: RefCell<BTreeMap<String, ArtifactDigest>>,
}

impl VerifierSandboxFactory for IsolatedVerifier {
    fn prepare(
        &self,
        mode: VerifierMode,
        artifacts: &[(String, ArtifactDigest)],
    ) -> Result<(), VerifierExecutionError> {
        if mode != VerifierMode::Separate {
            return Err(VerifierExecutionError::PreparationFailed(
                "verifier must be separately provisioned".to_owned(),
            ));
        }
        let mut files = self.files.borrow_mut();
        files.clear();
        files.extend(artifacts.iter().cloned());
        Ok(())
    }
}

#[test]
fn separate_verifier_receives_only_declared_artifacts_at_exact_paths() {
    let verifier = IsolatedVerifier {
        files: RefCell::new(BTreeMap::from([("/agent/secret".to_owned(), digest('s'))])),
    };
    let transfer = DeclaredArtifactTransfer::new(vec![
        ("/results/patch.diff", digest('a')),
        ("/results/report.json", digest('b')),
    ])
    .unwrap();

    prepare_verifier(&verifier, VerifierMode::Separate, &transfer).unwrap();

    assert_eq!(
        *verifier.files.borrow(),
        BTreeMap::from([
            ("/results/patch.diff".to_owned(), digest('a')),
            ("/results/report.json".to_owned(), digest('b')),
        ])
    );
    assert!(!verifier.files.borrow().contains_key("/agent/secret"));
}
