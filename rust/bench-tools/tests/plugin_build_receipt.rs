// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavioral contract for the parity runner's canonical build receipt.

use std::{collections::BTreeMap, path::Path, process::Command};

fn script() -> &'static str {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../scripts/run-plugin-parity.sh"
    )
}

fn decode_hex(value: &str) -> Vec<u8> {
    assert_eq!(value.len() % 2, 0);
    value
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).expect("hex is ASCII"), 16)
                .expect("valid hex byte")
        })
        .collect()
}

fn run(target: &Path, receipt: &Path, argument: &str) -> std::process::Output {
    let mut command = Command::new("sh");
    command
        .arg(script())
        .arg(target)
        .arg(receipt)
        .arg("--")
        .arg("/bin/true")
        .arg(argument)
        .env("RUSTC_WRAPPER", "")
        .env("SCCACHE_DIR", "cache\nwith-trailing-newline\n")
        .env("CC", "clang\u{1f}wrapped")
        .env("CXX", "clang++\nwrapped")
        .env("PATH", std::env::var_os("PATH").expect("test PATH"))
        .env_remove("SCCACHE_CACHE_SIZE")
        .env_remove("SCCACHE_IDLE_TIMEOUT");
    command.output().expect("parity script executes")
}

#[test]
fn receipt_round_trips_unset_empty_newline_separator_and_argv_bytes() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let target = temp.path().join("target");
    let receipt = temp.path().join("receipts/build.json");
    let argument = "arg\nwith\u{1f}separator\n";
    let output = run(&target, &receipt, argument);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = std::fs::read(&receipt).expect("receipt");
    assert_eq!(bytes.last(), Some(&b'\n'));
    let value: serde_json::Value = serde_json::from_slice(&bytes).expect("canonical JSON receipt");
    let mut canonical = serde_json_canonicalizer::to_vec(&value).expect("canonicalize receipt");
    canonical.push(b'\n');
    assert_eq!(bytes, canonical);
    assert_eq!(value["schema_version"], 1);
    let environment = value["environment"]
        .as_array()
        .expect("environment array")
        .iter()
        .map(|entry| {
            (
                entry["name"].as_str().expect("name"),
                (
                    entry["state"].as_str().expect("state"),
                    entry["value_hex"].as_str(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(environment["RUSTC_WRAPPER"], ("set", Some("")));
    assert_eq!(environment["SCCACHE_CACHE_SIZE"], ("unset", None));
    assert_eq!(
        decode_hex(environment["SCCACHE_DIR"].1.expect("set value")),
        b"cache\nwith-trailing-newline\n"
    );
    assert_eq!(
        decode_hex(environment["CC"].1.expect("set value")),
        b"clang\x1fwrapped"
    );
    assert_eq!(
        decode_hex(environment["CXX"].1.expect("set value")),
        b"clang++\nwrapped"
    );
    let argv = value["argv_hex"].as_array().expect("argv array");
    assert_eq!(
        decode_hex(argv.last().expect("argument").as_str().expect("hex argv")),
        argument.as_bytes()
    );

    let second = temp.path().join("receipts/build-again.json");
    let output = run(&target, &second, argument);
    assert!(output.status.success());
    assert_eq!(bytes, std::fs::read(second).expect("second receipt"));
}

#[test]
fn receipt_path_must_be_supplied_outside_the_target_tree() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let target = temp.path().join("target");
    let inside = target.join("receipts/build.json");
    let output = run(&target, &inside, "argument");
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("outside TARGET_PATH"));
    assert!(!inside.exists());

    let output = run(&target, &target, "argument");
    assert_eq!(output.status.code(), Some(2));
}
