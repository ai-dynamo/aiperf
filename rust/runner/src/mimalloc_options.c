// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <mimalloc.h>

int aiperf_mi_option_arena_eager_commit(void) {
  return (int)mi_option_arena_eager_commit;
}

int aiperf_mi_option_purge_delay(void) {
  return (int)mi_option_purge_delay;
}
