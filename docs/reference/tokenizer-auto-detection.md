---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Pre-Flight Tokenizer Auto Detection
---

# Pre-Flight Tokenizer Auto Detection

AIPerf resolves tokenizer names **before** spawning services via lightweight Hub API calls. This pre-flight check catches ambiguous or unknown names immediately without delaying startup: it does **not** download or load the tokenizer. Full tokenizer loading happens later inside each service, where errors like gated repos or missing files are caught and displayed with context-aware panels.

## How It Works

1. **Determine names**: Uses `--tokenizer` if specified, otherwise each `--model` name.
2. **Resolve aliases**: Lightweight Hub API calls to resolve aliases to canonical repository IDs (e.g., `qwen3-0.6b` → `Qwen/Qwen3-0.6B`).
3. **Fail fast on ambiguity**: If no exact or suffix match, displays top matches by downloads and exits.
4. **Cache results**: Resolved names are passed to all services so they skip re-resolution.

Pre-flight is skipped when `--use-server-token-count` is set with a non-synthetic dataset, or when the endpoint type doesn't require tokenization.

## Built-in Tokenizer

Pass `--tokenizer builtin` to use a zero-network-access tokenizer backed by [tiktoken](https://github.com/openai/tiktoken) with the `o200k_base` encoding (GPT-4o / o1 / o3, 200k vocabulary). This skips all HuggingFace Hub alias resolution and downloads.

Use this when you don't need a model-specific tokenizer and just want token counts for performance metrics. The encoding data is downloaded once on first use and cached locally by tiktoken -- subsequent runs require no network access.

```bash
aiperf profile --tokenizer builtin ...
```

## Placeholder Model Name Detection

When `--tokenizer` is **not** set, AIPerf normally derives the tokenizer name from `--model`. If the model name looks like an obvious LLM-hallucinated placeholder (e.g. `mock-model`, `test-model`, `fake-llama`), AIPerf substitutes `builtin` for that model and emits a warning instead of attempting an HF Hub lookup that would fail. This avoids a confusing tokenizer error when the user is iterating against a mock or test server.

The check fires when the model name (case-insensitive, with `_` normalized to `-`) is not path-like (no `/`, `\`, leading `.`, or leading `~`) **and** matches either:

| Match type | Values |
|---|---|
| Exact name | `test`, `mock`, `fake`, `dummy`, `example`, `sample`, `placeholder` |
| Substring | `mock-`, `-mock`, `fake-`, `-fake`, `test-model`, `-test-model`, `your-model`, `my-model`, `model-name`, `model-id` |

Examples that trigger the fallback: `mock-model`, `Test-Model-v2`, `MOCK_LLAMA`, `placeholder`, `my-model`. Examples that do **not** trigger it: `meta-llama/Llama-3-test-finetune` (path-like), `gpt2`, `Qwen/Qwen3-0.6B`.

**Sample output:**
```
WARNING  Model name 'mock-llama' looks like a placeholder; defaulting tokenizer to 'builtin' (tiktoken o200k_base). Pass --tokenizer <name> to override.
```

**Opt out** by passing `--tokenizer <name>` explicitly. Any explicit value wins, even one that looks placeholder-ish — the check only runs when the tokenizer would otherwise default from `--model`. If a model with a placeholder-shaped name is real on your inference server, set `--tokenizer` to a real HF repo (or to `builtin` yourself to suppress the warning).

## Automatic Cache Detection

When a HuggingFace tokenizer has been previously downloaded, AIPerf detects it in the local HF cache and loads directly without any network calls. This applies to both alias resolution and tokenizer loading -- no `model_info()` API call, no ETag update check. First run downloads as normal; every subsequent run is fully offline.

## Alias Resolution

1. **Local paths**: Absolute, `./`, `../`, or existing directories are used as-is.
2. **Cached locally**: If the model directory exists in the HF cache, the name is used as-is (no network).
3. **Offline mode**: If `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE` is set, names are used as-is.
4. **Direct lookup**: `model_info()` API call. Returns canonical `model.id` if found.
5. **Search fallback**: If direct lookup fails (`RepositoryNotFoundError` or `HfHubHTTPError`), searches with `list_models(search=name, limit=50)`:
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
| `--tokenizer <name-or-path>` | Explicit tokenizer name, local path, or `builtin` for tiktoken. If omitted, model names are used. |
| `--tokenizer-revision <rev>` | Git revision for the tokenizer repo. Default: `main`. |
| `--tokenizer-trust-remote-code` | Allow execution of custom tokenizer code from the repo. |
| `--use-server-token-count` | Skip client-side tokenization. Skips pre-flight validation with non-synthetic data. |

## See Also

- [Using Local Tokenizers](../tutorials/local-tokenizer.md)
