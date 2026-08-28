// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Catalog resolution tests (Task 13): ID normalization, path dedup, ordering.

use std::{collections::HashMap, path::PathBuf};

use aiperf_plugin_host::{
    catalog::resolve_catalog,
    discovery::{DiscoveredPackage, DiscoverySourceId},
};

fn manifest_yaml(id: &str, priority: i32) -> Vec<u8> {
    format!(
        "schema_version: \"2.0\"\n\
         packages:\n\
         \x20 - id: \"{id}\"\n\
         \x20   version: \"1.0.0\"\n\
         \x20   priority: {priority}\n\
         \x20   categories:\n\
         \x20     - category: exporter\n\
         \x20       id: \"demo-exporter\"\n"
    )
    .into_bytes()
}

fn discovered(path: &str, kind_ordinal: u8, authored_index: u32) -> DiscoveredPackage {
    DiscoveredPackage {
        manifest_path: PathBuf::from(path),
        source_id: DiscoverySourceId {
            kind_ordinal,
            authored_index,
        },
        priority: 0,
    }
}

#[test]
fn ids_differing_only_by_case_contest_the_same_bucket() {
    let mut bytes: HashMap<PathBuf, Vec<u8>> = HashMap::new();
    bytes.insert(
        PathBuf::from("/a/plugin.manifest.yaml"),
        manifest_yaml("Foo", 0),
    );
    bytes.insert(
        PathBuf::from("/b/plugin.manifest.yaml"),
        manifest_yaml("foo", 5),
    );

    let catalog = resolve_catalog(
        vec![
            discovered("/a/plugin.manifest.yaml", 3, 0),
            discovered("/b/plugin.manifest.yaml", 3, 1),
        ],
        bytes,
    );

    assert!(catalog.quarantined.is_empty(), "no quarantine expected");
    assert_eq!(
        catalog.winners.len(),
        1,
        "`Foo` and `foo` must be one package, not two: {:?}",
        catalog.winners
    );
    assert_eq!(catalog.winners[0].package_id, "foo");
    assert_eq!(catalog.shadows.len(), 1, "the loser must be shadowed");
    assert!(catalog.ambiguous.is_empty());
}

#[test]
fn a_repeated_search_path_does_not_make_a_package_ambiguous() {
    // `AIPERF_PLUGIN_PATH=/a:/a` discovers the same manifest twice.
    let mut bytes: HashMap<PathBuf, Vec<u8>> = HashMap::new();
    bytes.insert(
        PathBuf::from("/a/plugin.manifest.yaml"),
        manifest_yaml("foo", 0),
    );

    let catalog = resolve_catalog(
        vec![
            discovered("/a/plugin.manifest.yaml", 3, 0),
            discovered("/a/plugin.manifest.yaml", 3, 0),
        ],
        bytes,
    );

    assert!(
        catalog.ambiguous.is_empty(),
        "a duplicated path must not be ambiguous against itself: {:?}",
        catalog.ambiguous
    );
    assert_eq!(catalog.winners.len(), 1);
    assert!(catalog.shadows.is_empty());
}

#[test]
fn winner_order_is_deterministic_by_package_id() {
    let mut bytes: HashMap<PathBuf, Vec<u8>> = HashMap::new();
    for id in ["zeta", "alpha", "mid"] {
        bytes.insert(
            PathBuf::from(format!("/{id}/plugin.manifest.yaml")),
            manifest_yaml(id, 0),
        );
    }

    let discovered_pkgs: Vec<_> = ["zeta", "alpha", "mid"]
        .iter()
        .enumerate()
        .map(|(i, id)| discovered(&format!("/{id}/plugin.manifest.yaml"), 3, i as u32))
        .collect();

    let catalog = resolve_catalog(discovered_pkgs, bytes);
    let ids: Vec<_> = catalog
        .winners
        .iter()
        .map(|w| w.package_id.as_str())
        .collect();
    assert_eq!(ids, vec!["alpha", "mid", "zeta"]);
}
