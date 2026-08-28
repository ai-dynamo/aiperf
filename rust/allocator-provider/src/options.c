// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Provider-owned mimalloc option-index helpers.
 *
 * Exported functions return mimalloc option enum indices resolved against
 * the exact mimalloc version statically linked into this provider.  Consumers
 * must call these functions rather than duplicating numeric values, which can
 * change across mimalloc releases.
 *
 * Note on arena_eager_commit initialization: the original CLI set
 * mi_option_arena_eager_commit=0 in a priority-100 .init_array entry (before
 * mimalloc's priority-101 constructor).  Constructor priorities 0–100 are
 * reserved for the implementation (GCC/Clang -Wprio-ctor-dtor), so this
 * approach is not portable to all toolchains.  Set MIMALLOC_ARENA_EAGER_COMMIT=0
 * in the environment to achieve the same effect without a constructor.
 */

#include <mimalloc.h>

/*
 * Return the `mi_option_purge_delay` enum index from this provider's exact
 * mimalloc header.  Callers must not hard-code the numeric value.
 */
int mi_aiperf_option_purge_delay(void) {
    return (int)mi_option_purge_delay;
}
