# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit Python (transformers) reference token ids for tokenizer byte-parity.

Runs an adversarial battery of encode inputs (and chat-template message sets for
chat models) through `transformers.AutoTokenizer` and writes the resolved
tokenizer directory plus the reference ids to JSON. A Rust comparator
(`rust/runtime/examples/tokenizer_parity.rs`) loads the SAME directories through
`aiperf_runtime::dataset::HuggingFaceTokenizer` and asserts byte-equality.

Offline: set HF_HUB_OFFLINE=1; every model below is expected in the local cache.
"""

import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer  # noqa: E402

# (repo id, has_chat_template)
MODELS = [
    ("gpt2", False),
    ("Qwen/Qwen3-0.6B", True),
    ("mistralai/Mistral-7B-Instruct-v0.3", True),
    ("meta-llama/Llama-3.1-8B-Instruct", True),
]

# Adversarial encode inputs: (name, text).
ENCODE_INPUTS = [
    ("empty", ""),
    ("single_space", " "),
    ("many_spaces", "     "),
    ("tabs_newlines", "\t\n\r\n"),
    ("blank_lines", "\n\n\n"),
    ("leading_ws", "   leading"),
    ("trailing_ws", "trailing   "),
    ("both_ws", "  both  "),
    ("ascii_baseline", "Hello world"),
    ("accents", "café naïve résumé"),
    ("cjk", "日本語のテキストです"),
    ("emoji_zwj_family", "🎉🚀👨‍👩‍👧‍👦"),
    ("combining", "éàô"),
    ("zero_width", "zero​width​space"),
    ("rtl_arabic", "مرحبا بالعالم"),
    ("special_token_literal", "<|endoftext|>"),
    ("chat_markup_literal", "<s></s>[INST] hi [/INST]"),
    ("im_markup_literal", "<|im_start|>user\nhi<|im_end|>"),
    ("long_repeat", "a" * 777),
    ("digits", "1234567890" * 3),
    ("punct_run", "!@#$%^&*()_+-=[]{}|;':\",./<>?"),
    ("null_and_controls", "x\x00y\x07z\x1bw"),
    ("mixed_script", "The quick 狐 jumps over 🦊 the lazy dog\n\tTab."),
    ("code", "def f(x):\n    return x**2  # comment"),
    ("json_ish", '{"key": "value", "n": 42, "nested": {"a": [1, 2, 3]}}'),
    ("surrogate_ish_high_unicode", "𝕳𝖊𝖑𝖑𝖔 𝓦𝓸𝓻𝓵𝓭 𐍈"),
    ("repeated_newlines_text", "line1\n\n\nline2\n\nline3"),
]

# Adversarial chat message sets: (name, messages, add_generation_prompt).
CHAT_INPUTS = [
    ("user_only", [{"role": "user", "content": "Hello!"}], True),
    ("user_only_no_gen", [{"role": "user", "content": "Hello!"}], False),
    (
        "system_user",
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"},
        ],
        True,
    ),
    (
        "multi_turn",
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And times 3?"},
        ],
        True,
    ),
    ("empty_content", [{"role": "user", "content": ""}], True),
    (
        "unicode_and_newlines",
        [{"role": "user", "content": "café\n日本語\n🚀 line3"}],
        True,
    ),
    (
        "markup_in_content",
        [{"role": "user", "content": "print('<|im_end|> </s> [INST]')"}],
        True,
    ),
]


def resolve_dir(tok, repo: str) -> str:
    """Materialize a self-contained dir holding tokenizer.json + config.

    `save_pretrained` re-serializes the fast tokenizer's backend, so the written
    `tokenizer.json` is content-equivalent to the source (identical vocab/merges/
    normalizer) and carries the chat template in `tokenizer_config.json`. Rust
    loads this exact directory, so both sides tokenize the same artifact.
    """
    safe = repo.replace("/", "__")
    directory = os.path.join("/tmp/tok_parity_dirs", safe)
    os.makedirs(directory, exist_ok=True)
    tok.save_pretrained(directory)
    if not os.path.isfile(os.path.join(directory, "tokenizer.json")):
        raise RuntimeError(f"{repo}: save_pretrained produced no tokenizer.json")
    return directory


def main() -> int:
    out = {}
    for repo, has_chat in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {repo}: {exc}", file=sys.stderr)
            continue
        directory = resolve_dir(tok, repo)
        entry = {"repo": repo, "dir": directory, "encode": [], "chat": []}
        for name, text in ENCODE_INPUTS:
            ids = tok.encode(text, add_special_tokens=False)
            entry["encode"].append({"name": name, "text": text, "ids": ids})
        if has_chat and tok.chat_template:
            for name, messages, add_gen in CHAT_INPUTS:
                try:
                    ids = tok.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=add_gen,
                        return_dict=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  chat {repo}/{name} failed: {exc}", file=sys.stderr)
                    continue
                if hasattr(ids, "input_ids"):
                    ids = ids["input_ids"]
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                ids = [int(x) for x in ids]
                entry["chat"].append(
                    {
                        "name": name,
                        "messages": messages,
                        "add_generation_prompt": add_gen,
                        "ids": ids,
                    }
                )
        out[repo] = entry
        print(
            f"OK {repo}: {len(entry['encode'])} encode, {len(entry['chat'])} chat",
            file=sys.stderr,
        )

    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tokenizer_parity_golden.json"
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
