// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `GlobalAlloc` shim that imports `mi_*` directly from the provider cdylib.
//!
//! # How it works
//!
//! `aiperf-allocator-shim` declares the `mi_*` allocation functions as
//! `extern "C"` symbols.  These are satisfied at final link time by
//! `aiperf-allocator-provider`, which is a shared library containing the
//! canonical mimalloc instance.  Because both the host binary and every plugin
//! cdylib carry the same provider as a mandatory, non-delay dependency, all
//! allocation and deallocation goes through a single allocator instance in each
//! OS process.
//!
//! # What is NOT in this crate
//!
//! - No AIPerf wrapper around any `mi_*` function.
//! - No dispatch table, selector, or lazy-loading mechanism.
//! - No lock, metadata, or indirection between `GlobalAlloc` and `mi_*`.
//! - No dependency on `libmimalloc-sys` or any embedded mimalloc build.
//!   The `mi_*` symbols are satisfied only by the provider cdylib.
//!
//! # Installation
//!
//! ```rust,ignore
//! use aiperf_allocator_shim::MiMallocShim;
//!
//! #[global_allocator]
//! static GLOBAL: MiMallocShim = MiMallocShim;
//! ```
//!
//! The binary must also link against `aiperf-allocator-provider` to satisfy
//! the `mi_*` symbol references at link time.

use std::alloc::{GlobalAlloc, Layout};

/// Direct-import `GlobalAlloc` shim backed by the provider cdylib.
///
/// Imports `mi_malloc`, `mi_zalloc`, `mi_malloc_aligned`, `mi_zalloc_aligned`,
/// `mi_realloc`, `mi_realloc_aligned`, and `mi_free` directly from the provider
/// cdylib.  No wrapper, no indirection, no lazy loading.
pub struct MiMallocShim;

// Safety: these symbols are exported by the provider cdylib and satisfy the
// full allocator API contract.  mi_free correctly handles null pointers as
// required by the GlobalAlloc contract for dealloc.
unsafe extern "C" {
    fn mi_malloc(size: usize) -> *mut u8;
    fn mi_zalloc(size: usize) -> *mut u8;
    fn mi_malloc_aligned(size: usize, alignment: usize) -> *mut u8;
    fn mi_zalloc_aligned(size: usize, alignment: usize) -> *mut u8;
    fn mi_realloc(p: *mut u8, new_size: usize) -> *mut u8;
    fn mi_realloc_aligned(p: *mut u8, new_size: usize, alignment: usize) -> *mut u8;
    fn mi_free(p: *mut u8);

    /// Returns the mimalloc version as an integer: `major*10000 + minor*100 + patch`.
    pub fn mi_version() -> i32;

    /// Returns the process-global main subprocess handle.
    ///
    /// Comparing the returned opaque pointer between host and plugin confirms
    /// that both processes share the same mimalloc instance.
    pub fn mi_subproc_main() -> MiSubprocId;
}

/// Opaque subprocess handle returned by `mi_subproc_main`.
///
/// The single pointer field is the internal mimalloc subprocess pointer.
/// Identical values in host and plugin confirm a shared allocator instance.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MiSubprocId {
    /// Internal mimalloc subprocess pointer.  Not dereferenceable by callers.
    pub ptr: *const u8,
}

// SAFETY: mi_subproc_main returns a globally stable pointer that lives for
// the duration of the process.  There is no aliasing concern.
unsafe impl Send for MiSubprocId {}
unsafe impl Sync for MiSubprocId {}

// Standard allocation alignment threshold.  mimalloc's alignment guarantee
// for the basic (non-aligned) allocation path is 8 bytes on 64-bit platforms.
const BASIC_ALIGN: usize = 8;

// SAFETY: every method calls a matching mi_* function from the provider
// cdylib.  mimalloc's contract matches the GlobalAlloc contract:
//
// - alloc/alloc_zeroed: returns a properly aligned pointer or null on
//   allocation failure.
// - dealloc: mi_free accepts null (no-op) and handles any mi_malloc pointer.
// - realloc: mi_realloc(null, n) == mi_malloc(n); returns null on failure
//   without freeing the original pointer.  Layout size is the previous
//   allocation size, which mimalloc ignores in favor of its own tracking.
unsafe impl GlobalAlloc for MiMallocShim {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= BASIC_ALIGN {
            unsafe { mi_malloc(layout.size()) }
        } else {
            unsafe { mi_malloc_aligned(layout.size(), layout.align()) }
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= BASIC_ALIGN {
            unsafe { mi_zalloc(layout.size()) }
        } else {
            unsafe { mi_zalloc_aligned(layout.size(), layout.align()) }
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        // mi_free handles null correctly (no-op), but the GlobalAlloc contract
        // guarantees ptr is non-null.  Either way this is safe.
        unsafe { mi_free(ptr) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.align() <= BASIC_ALIGN {
            unsafe { mi_realloc(ptr, new_size) }
        } else {
            unsafe { mi_realloc_aligned(ptr, new_size, layout.align()) }
        }
    }
}
