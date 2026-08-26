// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authoring regression tests for opt-in cumulative streaming usage.

use aiperf_cli::flags::ProfileFlags;
use aiperf_cli::load::resolve_inputs;

fn parse(args: &[&str]) -> ProfileFlags {
    let args = args.iter().map(|value| value.to_string()).collect::<Vec<_>>();
    ProfileFlags::parse_from_args(&args).expect("profile flags should parse")
}

#[test]
fn per_chunk_usage_flag_projects_and_defaults_false() {
    let authored = parse(&[
        "--endpoint-type",
        "chat",
        "--streaming",
        "--use-server-token-count",
        "--per-chunk-usage",
    ]);
    assert_eq!(authored.per_chunk_usage, Some(true));
    assert!(resolve_inputs(&authored).expect("resolve authored flags").per_chunk_usage);

    let defaults = parse(&[]);
    assert_eq!(defaults.per_chunk_usage, None);
    assert!(!resolve_inputs(&defaults).expect("resolve defaults").per_chunk_usage);
}

#[test]
fn per_chunk_usage_explicit_false_is_retained() {
    let flags = parse(&["--per-chunk-usage=false"]);
    assert_eq!(flags.per_chunk_usage, Some(false));
    assert!(!resolve_inputs(&flags).expect("resolve explicit false").per_chunk_usage);
}
