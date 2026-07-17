# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Format-agnostic helpers shared across graph adapters.

Cross-cutting utilities (corpus-backed content synthesis) used by more than
one adapter family. Import the concrete module directly, e.g.
``from aiperf.dataset.graph.adapters.shared.content import ...``.

Workload-format detection lives in ``aiperf.dataset.graph.parser``
(``detect_format``): ``parser`` was its only importer, so it was never
adapter-shared.
"""
