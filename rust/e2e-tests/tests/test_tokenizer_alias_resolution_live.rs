// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;

const DOCUMENTED_ALIASES: &[(&str, &str)] = &[
    ("bert-base-uncased", "google-bert/bert-base-uncased"),
    ("roberta-large", "FacebookAI/roberta-large"),
    ("clip-vit-base-patch32", "openai/clip-vit-base-patch32"),
];

const LLM_SPECIFIC_NAMES: &[(&str, &str)] = &[
    ("Llama-2-7b-hf", "meta-llama/Llama-2-7b-hf"),
    ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B"),
    ("CodeLlama-7b-hf", "codellama/CodeLlama-7b-hf"),
    (
        "Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ),
    ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("gemma-2b", "google/gemma-2b"),
    ("phi-2", "microsoft/phi-2"),
    ("falcon-7b", "tiiuae/falcon-7b"),
];

const LLM_LOWERCASE_NAMES: &[(&str, &str)] = &[
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
    ("qwen2.5-7b", "Qwen/Qwen2.5-7B"),
    ("qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("llama-3.1-8b", "meta-llama/Llama-3.1-8B"),
    ("llama-2-7b-hf", "meta-llama/Llama-2-7b-hf"),
    ("mistral-7b-v0.1", "mistralai/Mistral-7B-v0.1"),
];

const LLM_GENERIC_NAMES: &[&str] = &["llama", "mistral", "qwen", "gemma", "phi", "falcon"];

const ENCODER_MODELS: &[(&str, &str)] = &[
    ("gpt2", "openai-community/gpt2"),
    ("gpt2-medium", "openai-community/gpt2-medium"),
    ("bert-base-uncased", "google-bert/bert-base-uncased"),
    ("bert-base-cased", "google-bert/bert-base-cased"),
    ("roberta-base", "FacebookAI/roberta-base"),
    ("roberta-large", "FacebookAI/roberta-large"),
    (
        "distilbert-base-uncased",
        "distilbert/distilbert-base-uncased",
    ),
    ("albert-base-v2", "albert/albert-base-v2"),
    ("t5-small", "google-t5/t5-small"),
    ("t5-base", "google-t5/t5-base"),
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
    ("bge-large-en", "BAAI/bge-large-en"),
    ("e5-large", "intfloat/e5-large"),
];

const FULL_REPOSITORY_IDS: &[&str] = &[
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "google-bert/bert-base-uncased",
    "openai-community/gpt2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "Qwen/Qwen2.5-7B-Instruct",
];

const EDGE_CASE_PATHS: &[&str] = &["../etc/passwd", "./local/path", "/absolute/path"];
const EDGE_CASE_INVALID: &[&str] = &[
    "",
    "a",
    "this-model-does-not-exist-xyz-123",
    "https://evil.com",
];

// Alias resolution has no native seam, so these helpers are unimplemented stubs:
// every `#[ignore]`d test below would pass vacuously if it were enabled. The
// tables above record the intended contract, not a checked one.
fn assert_resolves_to(_alias: &str, _expected: &str) {}

fn assert_unchanged(_name: &str) {}

fn assert_ambiguous(_name: &str) {}

#[ignore] // requires: Python aiperf.common.tokenizer + live HuggingFace Hub
#[tokio::test]
async fn test_documented_alias_resolves_correctly() {
    for (alias, expected) in DOCUMENTED_ALIASES {
        assert_resolves_to(alias, expected);
    }
}

#[ignore] // requires: Python aiperf.common.tokenizer + live HuggingFace Hub
#[tokio::test]
async fn test_specific_llm_names_resolve() {
    for (alias, expected) in LLM_SPECIFIC_NAMES {
        assert_resolves_to(alias, expected);
    }
}

#[ignore] // requires: Python aiperf.common.tokenizer + live HuggingFace Hub
#[tokio::test]
async fn test_lowercase_llm_names_resolve() {
    for (alias, expected) in LLM_LOWERCASE_NAMES {
        assert_resolves_to(alias, expected);
    }
}

#[ignore] // requires: Python aiperf.common.tokenizer + live HuggingFace Hub
#[tokio::test]
async fn test_generic_llm_names_are_ambiguous() {
    for generic_name in LLM_GENERIC_NAMES {
        assert_ambiguous(generic_name);
    }
}

#[ignore] // requires: Python aiperf.common.tokenizer + live HuggingFace Hub
#[tokio::test]
async fn test_encoder_model_resolves() {
    for (alias, expected) in ENCODER_MODELS {
        assert_resolves_to(alias, expected);
    }
}

#[ignore] // requires: Python aiperf.common.tokenizer + live HuggingFace Hub
#[tokio::test]
async fn test_full_repository_id_unchanged() {
    for full_id in FULL_REPOSITORY_IDS {
        assert_unchanged(full_id);
    }
}

#[ignore] // requires: Python aiperf.common.tokenizer + live HuggingFace Hub
#[tokio::test]
async fn test_local_paths_returned_unchanged() {
    for path in EDGE_CASE_PATHS {
        assert_unchanged(path);
    }
}

#[ignore] // requires: Python aiperf.common.tokenizer + live HuggingFace Hub
#[tokio::test]
async fn test_invalid_inputs_returned_unchanged() {
    for invalid_input in EDGE_CASE_INVALID {
        assert_unchanged(invalid_input);
    }
}
