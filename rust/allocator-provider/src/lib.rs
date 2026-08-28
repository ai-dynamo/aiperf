// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared allocator provider for native AIPerf plugins.
//!
//! `aiperf-allocator-provider` is a cdylib that statically embeds mimalloc v3
//! and re-exports exactly the `mi_*` symbols listed in `build.rs::EXPORTED_SYMBOLS`
//! through a platform-specific linker version script.  No Rust-level public
//! items are defined; all exported symbols come from the statically linked
//! mimalloc archive or from `src/options.c`.
//!
//! Because both a host process and every plugin cdylib that links against this
//! shared library see the same in-process instance, all allocation and
//! deallocation cross the host/plugin boundary safely: memory allocated in the
//! host can be freed in a plugin and vice versa.
//!
//! ## Option helpers
//!
//! `src/options.c` exports `mi_aiperf_option_purge_delay()`, which returns the
//! `mi_option_purge_delay` index resolved against the exact mimalloc header
//! version linked into this provider.  The CLI execute-mode uses this index
//! together with the exported `mi_option_set` to disable purge syscalls for
//! short-lived benchmark runs.
//!
//! Set `MIMALLOC_ARENA_EAGER_COMMIT=0` in the environment to disable eager
//! arena commit without a pre-constructor hook.
//!
//! See [`aiperf_allocator_shim`] for the `GlobalAlloc` implementation.

// No Rust-level public items are needed.  All exported symbols come from the
// statically linked mimalloc (via libmimalloc-sys) and are made visible by the
// version script / exported-symbols list emitted in build.rs.
