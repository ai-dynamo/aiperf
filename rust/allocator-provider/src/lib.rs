// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared allocator provider for native AIPerf plugins.
//!
//! `aiperf-allocator-provider` is the canonical mimalloc shared library that
//! both the AIPerf host process and every native plugin cdylib link against.
//! Because only one copy of mimalloc is loaded per OS process, all allocation
//! and deallocation cross the host/plugin boundary safely: memory allocated in
//! the host can be freed in a plugin and vice versa.
//!
//! The provider exports the symbols listed in `build.rs::EXPORTED_SYMBOLS`.
//! These are the only symbols the shim (`aiperf-allocator-shim`) imports.  No
//! AIPerf wrapper, dispatch table, or lazy-loading mechanism is permitted; the
//! shim imports `mi_*` directly with eager binding.
//!
//! # Usage
//!
//! The provider is a cdylib loaded at process start as a mandatory, non-delay
//! shared library dependency of the `aiperf` binary.  Plugins also carry it as
//! a mandatory dependency.  The host verifies the mapped provider and its
//! relocation origin before discovery begins.
//!
//! See [`aiperf_allocator_shim`] for the `GlobalAlloc` implementation.

// No Rust-level public items are needed.  All exported symbols come from the
// statically linked mimalloc (via libmimalloc-sys) and are made visible by the
// version script / exported-symbols list emitted in build.rs.
