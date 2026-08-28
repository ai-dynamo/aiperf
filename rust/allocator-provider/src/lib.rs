// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared allocator provider for native AIPerf plugins.
//!
//! `aiperf-allocator-provider` is a cdylib that statically embeds mimalloc v3
//! and re-exports exactly the `mi_*` symbols listed in `build.rs::EXPORTED_SYMBOLS`
//! through a platform-specific linker version script.  No Rust-level public
//! items are defined; all exported symbols come from the statically linked
//! mimalloc archive.
//!
//! Because both a host process and every plugin cdylib that links against this
//! shared library see the same in-process instance, all allocation and
//! deallocation cross the host/plugin boundary safely: memory allocated in the
//! host can be freed in a plugin and vice versa.
//!
//! The provider exports the symbols listed in `build.rs::EXPORTED_SYMBOLS`.
//! These are the only symbols the shim (`aiperf-allocator-shim`) imports.  No
//! AIPerf wrapper, dispatch table, or lazy-loading mechanism is permitted; the
//! shim imports `mi_*` directly with eager binding.
//!
//! See [`aiperf_allocator_shim`] for the `GlobalAlloc` implementation.

// No Rust-level public items are needed.  All exported symbols come from the
// statically linked mimalloc (via libmimalloc-sys) and are made visible by the
// version script / exported-symbols list emitted in build.rs.
