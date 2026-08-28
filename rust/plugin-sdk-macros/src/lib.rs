// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Procedural macro crate for native AIPerf plugins.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Marks the plugin's initialization function and generates the required
/// `aiperf_plugin_entry_v1` native entry symbol.
///
/// # Usage
///
/// ```ignore
/// use aiperf_plugin_sdk::declaration::PluginDeclarationV1;
/// use aiperf_plugin_sdk_macros::aiperf_plugin;
///
/// #[aiperf_plugin]
/// fn my_plugin() -> PluginDeclarationV1 {
///     PluginDeclarationV1 {
///         name: "my-plugin",
///         version: env!("CARGO_PKG_VERSION"),
///         aiperf_sdk_version: "0.13.0",
///         capabilities: &[],
///     }
/// }
/// ```
///
/// The macro preserves the original function and generates a
/// `#[no_mangle] pub extern "Rust" fn aiperf_plugin_entry_v1()` that returns
/// a `&'static PluginDeclarationV1` backed by a `OnceLock`.
#[proc_macro_attribute]
pub fn aiperf_plugin(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let inner_name = &input.sig.ident;
    let inner_block = &input.block;
    let inner_inputs = &input.sig.inputs;
    let inner_vis = &input.vis;

    let expanded = quote! {
        #inner_vis fn #inner_name(#inner_inputs)
            -> ::aiperf_plugin_sdk::declaration::PluginDeclarationV1
        #inner_block

        #[no_mangle]
        pub extern "Rust" fn aiperf_plugin_entry_v1()
            -> &'static ::aiperf_plugin_sdk::declaration::PluginDeclarationV1
        {
            static DECL: ::std::sync::OnceLock<
                ::aiperf_plugin_sdk::declaration::PluginDeclarationV1
            > = ::std::sync::OnceLock::new();
            DECL.get_or_init(|| #inner_name())
        }
    };

    TokenStream::from(expanded)
}
