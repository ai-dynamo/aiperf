// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Provider-owned mimalloc option initialization and option-index helpers.
 *
 * This translation unit is compiled into the provider cdylib so that option
 * constants and initialization are resolved against the exact mimalloc version
 * statically linked here, not re-derived from a separate header in a consumer.
 */

#include <mimalloc.h>

/*
 * Priority-100 constructor: runs before mimalloc's own priority-101
 * constructor, setting defaults before the initial arena is committed.
 *
 * Disabling arena_eager_commit prevents mimalloc from committing physical
 * pages for the full initial arena on the first allocation, reducing startup
 * RSS on workloads that do not immediately saturate the arena.
 *
 * GCC/Clang constructor priorities are supported on Linux and macOS; the
 * initializer is skipped on MSVC (Windows) where DllMain ordering achieves
 * the equivalent effect via the mimalloc Windows initializer.
 */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor(100)))
static void aiperf_alloc_init_options(void) {
    mi_option_set_default(mi_option_arena_eager_commit, 0);
}
#endif

/*
 * Return the `mi_option_purge_delay` enum index from this provider's exact
 * mimalloc header.  Consumers must call this function rather than hard-coding
 * the numeric value, which can change between mimalloc releases.
 */
int mi_aiperf_option_purge_delay(void) {
    return (int)mi_option_purge_delay;
}
