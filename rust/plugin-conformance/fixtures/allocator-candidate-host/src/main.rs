// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Allocator candidate host fixture.
//!
//! Validates the shared-provider topology by:
//!
//! 1. Checking `mi_version()` is a v3.x value.
//! 2. Loading the candidate plugin cdylib via `libloading`.
//! 3. Calling the plugin's exported `plugin_subproc_id()` and comparing the
//!    returned pointer with the host's own `mi_subproc_main()` result.
//! 4. Exercising allocation operations through the shim in both host and plugin.
//! 5. Verifying cross-boundary free: allocate in plugin, free in host.
//!
//! Exits 0 on success.  Any assertion failure prints a message to stderr and
//! exits non-zero.

use aiperf_allocator_shim::{MiMallocShim, MiSubprocId, mi_subproc_main, mi_version};
use std::alloc::{GlobalAlloc, Layout};
use std::env;
use std::rc::Rc;
use std::sync::Arc;

#[global_allocator]
static GLOBAL: MiMallocShim = MiMallocShim;

fn main() {
    let plugin_path = env::args()
        .nth(1)
        .expect("usage: allocator-candidate-host <plugin-cdylib-path>");

    // ── 1. Version check ──────────────────────────────────────────────────
    let version = unsafe { mi_version() };
    assert!(
        version >= 30000 && version < 40000,
        "mi_version() returned {version}; expected v3.x (30000..40000)"
    );
    eprintln!("host: mi_version() = {version}");

    // ── 2. Allocation operations in the host ──────────────────────────────
    let s = String::from("hello from host");
    let v: Vec<u64> = (0..256).collect();
    let b: Box<u64> = Box::new(0xDEAD_BEEF);
    let arc: Arc<u64> = Arc::new(0xCAFE_BABE);
    let rc: Rc<u64> = Rc::new(0x1234_5678);
    assert_eq!(s, "hello from host");
    assert_eq!(v.len(), 256);
    assert_eq!(*b, 0xDEAD_BEEF);
    assert_eq!(*arc, 0xCAFE_BABE);
    assert_eq!(*rc, 0x1234_5678);
    eprintln!("host: basic allocation operations passed");

    // ── 3. Load the plugin cdylib ─────────────────────────────────────────
    // SAFETY: we are loading a trusted test fixture in an isolated subprocess.
    let lib = unsafe {
        libloading::Library::new(&plugin_path)
            .unwrap_or_else(|e| panic!("failed to load plugin {plugin_path}: {e}"))
    };
    eprintln!("host: loaded plugin {plugin_path}");

    // ── 4. Plugin subproc check (shared instance) ─────────────────────────
    // The plugin exports `plugin_subproc_id() -> MiSubprocId`.
    // SAFETY: the symbol type matches the exported function signature.
    let plugin_subproc_id: libloading::Symbol<unsafe extern "C" fn() -> MiSubprocId> =
        unsafe { lib.get(b"plugin_subproc_id\0") }
            .expect("plugin must export `plugin_subproc_id`");

    let host_id = unsafe { mi_subproc_main() };
    let plugin_id = unsafe { plugin_subproc_id() };

    assert_eq!(
        host_id.ptr, plugin_id.ptr,
        "host and plugin must share the same allocator instance: \
         host mi_subproc_main={:p}, plugin mi_subproc_main={:p}",
        host_id.ptr, plugin_id.ptr
    );
    eprintln!("host: shared allocator instance confirmed (ptr = {:p})", host_id.ptr);

    // ── 5. Cross-boundary free: allocate in plugin, free in host ─────────
    // The plugin exports `plugin_alloc(n: usize) -> *mut u8`.
    // SAFETY: symbol type matches.
    let plugin_alloc: libloading::Symbol<unsafe extern "C" fn(usize) -> *mut u8> =
        unsafe { lib.get(b"plugin_alloc\0") }
            .expect("plugin must export `plugin_alloc`");

    let plugin_free: libloading::Symbol<unsafe extern "C" fn(*mut u8, usize)> =
        unsafe { lib.get(b"plugin_free\0") }
            .expect("plugin must export `plugin_free`");

    let layout = Layout::array::<u8>(128).unwrap();

    // Allocate in plugin, free in host (cross-boundary free).
    let ptr = unsafe { plugin_alloc(128) };
    assert!(!ptr.is_null(), "plugin_alloc returned null");
    unsafe { std::ptr::write_bytes(ptr, 0xCC, 128) };
    unsafe { MiMallocShim.dealloc(ptr, layout) };
    eprintln!("host: cross-boundary free (plugin→host) passed");

    // Allocate in host, free in plugin (cross-boundary free, other direction).
    let ptr2 = unsafe { MiMallocShim.alloc(layout) };
    assert!(!ptr2.is_null(), "host alloc returned null");
    unsafe { std::ptr::write_bytes(ptr2, 0xDD, 128) };
    unsafe { plugin_free(ptr2, 128) };
    eprintln!("host: cross-boundary free (host→plugin) passed");

    // ── 6. Trait object and boxed future allocation ───────────────────────
    let obj: Box<dyn std::fmt::Display> = Box::new(42u64);
    assert_eq!(format!("{obj}"), "42");

    use std::future::Future;
    use std::pin::Pin;
    let fut: Pin<Box<dyn Future<Output = u64>>> = Box::pin(async { 99u64 });
    drop(fut);
    eprintln!("host: trait object and boxed future allocation passed");

    // ── 7. Trait object error allocation ─────────────────────────────────
    let err: Box<dyn std::error::Error> = Box::new(std::fmt::Error);
    drop(err);

    eprintln!("host: all allocator conformance checks passed");
}
