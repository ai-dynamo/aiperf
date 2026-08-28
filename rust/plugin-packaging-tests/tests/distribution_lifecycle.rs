// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distribution lifecycle: the published inventory is authenticated, is
//! replaced atomically, and is never read through a symlink.

use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use aiperf_plugin_host::inventory::{InventoryPackageV1, PluginInventoryV1};

fn package(id: &str) -> InventoryPackageV1 {
    InventoryPackageV1 {
        id: id.to_string(),
        version: "1.2.3".to_string(),
        artifact_digest: format!("blake3:{}", "a".repeat(64)),
        manifest_digest: format!("blake3:{}", "b".repeat(64)),
        build_id: Some("build-7".to_string()),
    }
}

#[test]
fn inventory_digest_must_match() {
    let inventory = PluginInventoryV1::new(1, vec![package("aiperf.example")]);
    inventory.verify_digest().expect("freshly minted digest verifies");

    let mut tampered = inventory.clone();
    tampered.packages[0].version = "9.9.9".to_string();
    assert!(tampered.verify_digest().is_err(), "tampered payload rejected");

    let mut forged = inventory.clone();
    forged.inventory_digest = "c".repeat(64);
    assert!(forged.verify_digest().is_err(), "forged digest rejected");

    let mut malformed = inventory;
    malformed.inventory_digest = "not-a-digest".to_string();
    assert!(malformed.verify_digest().is_err(), "malformed digest rejected");
}

#[test]
fn inventory_atomic_publish() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("inventory.json");

    let old = PluginInventoryV1::new(1, vec![package("aiperf.old")]);
    let new = PluginInventoryV1::new(2, vec![package("aiperf.new")]);
    old.publish(&path).expect("publish gen 1");

    let stop = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let path = path.clone();
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    // Every observation must be a complete, verified document:
                    // either the old generation or the new one, never a partial
                    // write of the final path.
                    let loaded =
                        PluginInventoryV1::load_and_verify(&path).expect("readers never see a partial file");
                    assert!(loaded.generation == 1 || loaded.generation == 2);
                }
            })
        })
        .collect();

    for _ in 0..200 {
        new.publish(&path).expect("publish gen 2");
        old.publish(&path).expect("publish gen 1");
    }
    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().expect("reader thread");
    }
}

#[test]
#[cfg(unix)]
fn inventory_symlink_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let real = dir.path().join("real.json");
    PluginInventoryV1::new(1, vec![package("aiperf.example")])
        .publish(&real)
        .expect("publish");

    let link = dir.path().join("inventory.json");
    std::os::unix::fs::symlink(&real, &link).expect("symlink");

    assert!(
        PluginInventoryV1::load_and_verify(&link).is_err(),
        "an inventory reached through a symlink is refused"
    );
    // Sanity: the real path still loads.
    assert!(PluginInventoryV1::load_and_verify(&real).is_ok());
    drop(fs::metadata(&real));
}
