// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `cargo aiperf-plugin` subcommand — hermetic plugin build and validation.

use std::path::PathBuf;

fn main() {
    // cargo passes the subcommand name as argv[1] when invoked as
    // `cargo aiperf-plugin`; skip it.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args = if args.first().map(|s| s.as_str()) == Some("aiperf-plugin") {
        &args[1..]
    } else {
        &args[..]
    };

    match args.first().map(|s| s.as_str()) {
        Some("build") => cmd_build(&args[1..]),
        Some("--help") | Some("-h") | None => print_help(),
        Some(other) => {
            eprintln!("unknown subcommand: {other}");
            std::process::exit(1);
        }
    }
}

fn print_help() {
    println!("cargo aiperf-plugin <COMMAND>");
    println!();
    println!("Commands:");
    println!("  build   Build a plugin cdylib in a hermetic sandbox");
    println!();
    println!("Options:");
    println!("  --help  Print this help message");
}

fn cmd_build(args: &[String]) {
    let mut plugin_dir: Option<PathBuf> = None;
    let mut sdk_dir: Option<PathBuf> = None;
    let mut release = false;
    let mut target: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--release" => release = true,
            "--sdk" => {
                i += 1;
                sdk_dir = args.get(i).map(PathBuf::from);
            }
            "--target" => {
                i += 1;
                target = args.get(i).cloned();
            }
            "--help" | "-h" => {
                println!("cargo aiperf-plugin build [OPTIONS] [PLUGIN_DIR]");
                println!();
                println!("Options:");
                println!("  --release        Build in release mode (enforces panic=abort)");
                println!("  --sdk <DIR>      Path to alternate SDK directory");
                println!("  --target <TRIPLE> Cross-compilation target triple");
                return;
            }
            arg if !arg.starts_with('-') => {
                plugin_dir = Some(PathBuf::from(arg));
            }
            other => {
                eprintln!("unknown option: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let plugin_dir = plugin_dir.unwrap_or_else(|| std::env::current_dir().unwrap());
    let config = aiperf_plugin_sdk::build::BuildConfig {
        plugin_dir,
        sdk_dir,
        release,
        target,
    };

    match aiperf_plugin_sdk::build::build_plugin(&config) {
        Ok(artifact) => {
            println!("Built: {}", artifact.display());
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
