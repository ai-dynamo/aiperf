<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Current-State Agent Documentation

## Goal

Make the shared agent instructions describe only the repository as it exists now.

## Scope

Update `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, and
`.cursor/rules/python.mdc` together.

Retain current architecture, supported behavior, commands, coding constraints,
verification requirements, and documentation-maintenance rules.

Remove:

- migration and porting narratives;
- former names and deleted systems;
- comparisons with retired implementations;
- implementation provenance and historical chronology;
- aspirational designs, future work, and known gaps.

Rewrite necessary sentences in direct present tense instead of deleting useful
current-state facts that happen to share a paragraph with historical context.

## Verification

Run `python tools/check_agent_files_sync.py` and scan the resulting shared body
for historical, migration, porting, aspirational, and gap-oriented language.
