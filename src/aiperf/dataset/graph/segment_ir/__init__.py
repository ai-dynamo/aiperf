# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Format-agnostic segment-trie IR: the content-addressed prefix-segment store
and its interned builder + ordinal scheme.

Any adapter can target this IR by returning a ``ParsedGraph`` whose dispatchable
``LlmNode``s carry ``metadata["trie"]["prompt_segment_ids"]`` (an ordered path
into a ``SegmentPool``) with ``ParsedGraph.segment_pool`` set. Everything here is
independent of any specific recorded trace format (weka, dynamo, ...).
"""
