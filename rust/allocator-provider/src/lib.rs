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
//! ## Option initialization
//!
//! `src/options.c` installs a priority-100 process constructor (GCC/Clang;
//! before mimalloc's own priority-101 constructor) that sets
//! `mi_option_arena_eager_commit = 0`, preventing mimalloc from committing
//! physical pages for the full initial arena on first allocation.  Consumers
//! do not need to perform any option initialization themselves.
//!
//! `mi_aiperf_option_purge_delay()` is exported for the CLI execute-mode,
//! which sets `mi_option_purge_delay = -1` to skip purge syscalls during
//! short-lived benchmark runs.  `mi_option_set` is exported for the same
//! purpose.
//!
//! See [`aiperf_allocator_shim`] for the `GlobalAlloc` implementation.

// No Rust-level public items are needed.  All exported symbols come from the
// statically linked mimalloc (via libmimalloc-sys) and are made visible by the
// version script / exported-symbols list emitted in build.rs.
