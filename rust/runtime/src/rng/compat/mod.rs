// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python and numpy random-number compatibility shims.
//!
//! Byte-exact re-implementations of CPython's Mersenne Twister and numpy's
//! PCG64 generator, used to reproduce Python-derived sampling sequences.

pub mod numpy_generator;
pub mod numpy_pcg64;
pub mod python_mt;
pub mod python_random;
pub(crate) mod ziggurat_constants;
