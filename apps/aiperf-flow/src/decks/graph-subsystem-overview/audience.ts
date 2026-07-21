/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared audience discriminant for the Graph Subsystem deck. Ports the source canvas's
//! manager/developer toggle: developer view reveals code symbols, file references, and
//! on-disk store internals; manager view keeps the concepts only.

export type Audience = "manager" | "developer";
