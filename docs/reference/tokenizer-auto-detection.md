<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pre-Flight Tokenizer Auto Detection

AIPerf resolves tokenizer names **before** spawning services via lightweight Hub API calls. This pre-flight check catches ambiguous or unknown names immediately without delaying startup: it does **not** download or load the tokenizer. Full tokenizer loading happens later inside each service, where errors like gated repos or missing files are caught and displayed with context-aware panels.

## How It Works

1. **Determine names**: Uses `--tokenizer` if specified, otherwise each `--model` name.
2. **Resolve aliases**: Lightweight Hub API calls to resolve aliases to canonical repository IDs (e.g., `qwen3-0.6b` → `Qwen/Qwen3-0.6B`).
3. **Fail fast on ambiguity**: If no exact or suffix match, displays top matches by downloads and exits.
4. **Cache results**: Resolved names are passed to all services so they skip re-resolution.

Pre-flight is skipped when `--use-server-token-count` is set with a non-synthetic dataset, or when the endpoint type doesn't require tokenization.

## Alias Resolution

1. **Local paths**: Absolute, `./`, `../`, or existing directories are used as-is.
2. **Offline mode**: If `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE` is set, names are used as-is.
3. **Direct lookup**: `model_info()` API call. Returns canonical `model.id` if found.
4. **Search fallback**: If direct lookup fails (`RepositoryNotFoundError` or `HfHubHTTPError`), searches with `list_models(search=name, limit=50)`:
   - **Exact match**: Result ID matches input (case-insensitive).
   - **Suffix match**: Result ends with `/<name>`, picks highest downloads.
   - **Ambiguous**: No match found, returns top 5 suggestions.

Set `HF_TOKEN` for gated or private models.

## Output Examples

**Successful resolution:**
```
INFO     ✓ Tokenizer Qwen/Qwen3-0.6B detected for qwen3-0.6b
INFO     1 tokenizer validated • 1 resolved • 0.3s
```

**Ambiguous name:**
```
╭──────────────────────────────── Ambiguous Tokenizer Name ─────────────────────────────────╮
│                                                                                           │
│  'llama-3' matched multiple HuggingFace tokenizers                                        │
│                                                                                           │
│  AIPerf needs a tokenizer for accurate client-side token counting and synthetic prompt    │
│  generation.                                                                              │
│                                                                                           │
│  Did you mean one of these?                                                               │
│    • meta-llama/Llama-3.1-8B-Instruct (8.4M downloads)                                    │
│    • meta-llama/Llama-3.2-1B-Instruct (2.9M downloads)                                    │
│    • meta-llama/Llama-3.2-1B (2.4M downloads)                                             │
│    • meta-llama/Meta-Llama-3-8B (1.8M downloads)                                          │
│    • meta-llama/Llama-3.2-3B-Instruct (1.6M downloads)                                    │
│                                                                                           │
│  Suggested Fixes:                                                                         │
│    • Specify explicitly: --tokenizer meta-llama/Llama-3.1-8B-Instruct                     │
│    • Skip tokenizer (non-synthetic data only): --use-server-token-count                   │
│                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```

**Gated repository error (runtime):**
```
╭───────────────────────────────── Gated Repository ──────────────────────────────────╮
│                                                                                     │
│  Failed to load tokenizer 'tiiuae/falcon-180B'                                      │
│                                                                                     │
│  AIPerf needs a tokenizer for accurate client-side token counting and synthetic     │
│  prompt generation.                                                                 │
│                                                                                     │
│  Possible Causes:                                                                   │
│    • Model is gated - requires accepting terms on HuggingFace                       │
│                                                                                     │
│  Investigation Steps:                                                               │
│    1. Visit huggingface.co/<model> to request access                                │
│                                                                                     │
│  Suggested Fixes:                                                                   │
│    • Accept terms, then: huggingface-cli login                                      │
│    • Skip tokenizer (non-synthetic data only): --use-server-token-count             │
│                                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```

## Runtime Error Panels

If a tokenizer fails during service initialization, AIPerf walks the `__cause__` chain to show a context-aware panel. Duplicate errors across services are shown once.

| Exception Type | Panel Title | Fix |
|---|---|---|
| `GatedRepoError` | Gated Repository | Accept terms, then: `huggingface-cli login` |
| `RepositoryNotFoundError` | Repository Not Found | Use full ID: `--tokenizer org-name/model-name` |
| `RevisionNotFoundError` | Invalid Git Revision | Remove `--tokenizer-revision` or use `--tokenizer-revision main` |
| `EntryNotFoundError` | Missing Tokenizer Files | Use a different tokenizer that matches your model |
| `LocalEntryNotFoundError` | Offline - Files Not Cached | Pre-download online, then: `export HF_HUB_OFFLINE=1` |
| `HfHubHTTPError` | HuggingFace Hub Error | Check network connectivity |
| `ModuleNotFoundError` | Missing Python Package | Install: `pip install <package>` |
| `PermissionError` | Cache Permission Error | Fix: `chmod -R u+rw ~/.cache/huggingface/` |
| `TimeoutError` | Network Timeout | Pre-download and use: `--tokenizer ./local-path` |
| `OSError` | Tokenizer Load Error | Clear cache and retry |

## CLI Options

| Option | Description |
|---|---|
| `--tokenizer <name-or-path>` | Explicit tokenizer name or local path. If omitted, model names are used. |
| `--tokenizer-revision <rev>` | Git revision for the tokenizer repo. Default: `main`. |
| `--tokenizer-trust-remote-code` | Allow execution of custom tokenizer code from the repo. |
| `--use-server-token-count` | Skip client-side tokenization. Skips pre-flight validation with non-synthetic data. |

## See Also

- [Using Local Tokenizers](../tutorials/local-tokenizer.md)
