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
/// static PKG: std::sync::LazyLock<aiperf_plugin_api::descriptor::PluginPackageDescriptor> =
///     std::sync::LazyLock::new(|| {
///         aiperf_plugin_api::descriptor::PluginPackageDescriptor::from_authored(
///             "my-plugin", env!("CARGO_PKG_VERSION"), "My plugin"
///         ).unwrap()
///     });
///
/// struct MyExtension;
/// impl aiperf_plugin_api::extension::AIPerfExtension for MyExtension {
///     fn register(&self, _r: &mut aiperf_plugin_api::extension::PluginRegistrar<'_>)
///         -> Result<(), aiperf_plugin_api::error::ExtensionError> { Ok(()) }
/// }
/// static EXT: MyExtension = MyExtension;
///
/// #[aiperf_plugin]
/// fn my_plugin() -> PluginDeclarationV1 {
///     PluginDeclarationV1 { package: &*PKG, extension: &EXT }
/// }
/// ```
///
/// The macro preserves the original function and generates a
/// `#[no_mangle] pub unsafe fn aiperf_plugin_entry_v1()` that returns a
/// `PluginDeclarationV1` by value, backed by a `OnceLock`.
///
/// The generated function has type `unsafe fn() -> PluginDeclarationV1`,
/// matching the host-authoritative `PluginEntryV1` alias exactly.
#[proc_macro_attribute]
pub fn aiperf_plugin(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let inner_name = &input.sig.ident;
    let inner_block = &input.block;
    let inner_inputs = &input.sig.inputs;
    let inner_vis = &input.vis;

    let expanded = quote! {
        #inner_vis fn #inner_name(#inner_inputs)
            -> ::aiperf_plugin_api::extension::PluginDeclarationV1
        #inner_block

        /// # Safety
        ///
        /// The host may call this only after manifest and build-identity
        /// validation confirms the library's universe digest matches the host
        /// ABI universe. The returned `PluginDeclarationV1` is `'static` and
        /// must not be freed by the host.
        #[no_mangle]
        pub unsafe fn aiperf_plugin_entry_v1()
            -> ::aiperf_plugin_api::extension::PluginDeclarationV1
        {
            static DECL: ::std::sync::OnceLock<
                ::aiperf_plugin_api::extension::PluginDeclarationV1
            > = ::std::sync::OnceLock::new();
            *DECL.get_or_init(|| #inner_name())
        }
    };

    TokenStream::from(expanded)
}
