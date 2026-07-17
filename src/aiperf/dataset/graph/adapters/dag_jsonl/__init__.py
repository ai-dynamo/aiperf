# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native ``dag_jsonl`` graph adapter.

Reuses the legacy ``dag_jsonl`` loader and expands its Conversation output into
per-root instanced trees (:mod:`.tree`), then lowers those trees onto the
unified-segment-store graph IR (:mod:`.lowering`). Consumers import the leaf
modules (:mod:`.tree`, :mod:`.lowering`) directly rather than the legacy loader
internals.
"""
