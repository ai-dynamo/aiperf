// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Allocator candidate plugin fixture cdylib.
//!
//! Exports three functions used by the host conformance fixture:
//!
//! - `plugin_subproc_id()` — returns `mi_subproc_main()` so the host can
//!   verify the shared allocator instance.
//! - `plugin_alloc(n)` — allocates `n` bytes through the shim; the host
//!   frees the returned pointer to test cross-boundary deallocation.
//! - `plugin_free(ptr, n)` — frees a pointer allocated by the host to test
//!   the other cross-boundary direction.

use aiperf_allocator_shim::{MiMallocShim, MiSubprocId, mi_subproc_main};
use std::alloc::{GlobalAlloc, Layout};

#[global_allocator]
static GLOBAL: MiMallocShim = MiMallocShim;

/// Returns the process-global mimalloc subprocess handle.
///
/// The host compares this with its own `mi_subproc_main()` result to confirm
/// that host and plugin share one allocator instance.
#[no_mangle]
pub extern "C" fn plugin_subproc_id() -> MiSubprocId {
    // SAFETY: mi_subproc_main is exported by the provider cdylib and is safe
    // to call after the process has initialized.
    unsafe { mi_subproc_main() }
}

/// Allocates `n` bytes through the shim and returns the raw pointer.
///
/// The caller (host fixture) is responsible for freeing the returned pointer,
/// which tests cross-boundary deallocation.
#[no_mangle]
pub extern "C" fn plugin_alloc(n: usize) -> *mut u8 {
    let layout = Layout::array::<u8>(n).expect("plugin_alloc: invalid layout");
    // SAFETY: layout is valid; MiMallocShim satisfies the GlobalAlloc contract.
    unsafe { MiMallocShim.alloc(layout) }
}

/// Frees a pointer allocated by the caller with size `n`.
///
/// Used to test the host→plugin deallocation direction.
#[no_mangle]
pub extern "C" fn plugin_free(ptr: *mut u8, n: usize) {
    if ptr.is_null() {
        return;
    }
    let layout = Layout::array::<u8>(n).expect("plugin_free: invalid layout");
    // SAFETY: ptr was allocated by the host through MiMallocShim, which uses
    // the same provider cdylib, so the pointer is valid for mi_free.
    unsafe { MiMallocShim.dealloc(ptr, layout) }
}
