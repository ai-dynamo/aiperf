// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokenizer seam and native Hugging Face / tiktoken implementations.
//!
//! Composition calls this trait before interning messages, making token IDs the
//! authoritative message identity and input-length source. No dispatch path
//! tokenizes again.

use std::fmt::{self, Display};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use bytes::Bytes;
use dynamo_renderer::{ChatTemplate, ContextMixins, OAIChatLikeRequest, PromptFormatter};
use dynamo_tokenizers::HuggingFaceTokenizer as DynamoHuggingFaceTokenizer;
use dynamo_tokenizers::traits::{Decoder as _, Encoder as _};
use minijinja::Value as JinjaValue;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tiktoken_rs::{
    CoreBPE, cl100k_base_singleton, o200k_base_singleton, o200k_harmony_singleton,
    p50k_base_singleton, p50k_edit_singleton, r50k_base_singleton,
};
// tiktoken-rs 0.9's `CoreBPE::new` wants `rustc_hash` 1.x maps; this bridged pin
// (renamed `rustc-hash` 1.x) matches its hasher type, because the workspace
// `rustc_hash` is 2.x whose `FxHashMap` is a distinct, incompatible type.
use tiktoken_rustc_hash::FxHashMap as TiktokenFxHashMap;
use tokenizers::Tokenizer as HfTokenizer;
use tokenizers::tokenizer::PostProcessor as _;

use crate::clock::{Clock, RealClock};
use crate::dataset::error::{DatasetError, Result};
use crate::transport::core::Response;
use crate::transport::http::config::ClientConfig;
use crate::transport::http::models::RequestConfig;
use crate::transport::http::transport::http_transport::HttpTransport;

/// Tokenization contract used by composers and input accounting.
pub trait TextTokenizer: Send + Sync {
    /// Encode text without automatically adding model special tokens.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Encode an ordered batch without automatically adding model special tokens.
    ///
    /// A successful result contains exactly one token vector per input, and result
    /// element `i` is semantically identical to `encode(texts[i])`. Empty input
    /// returns an empty vector without invoking a tokenizer backend.
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        texts.par_iter().map(|text| self.encode(text)).collect()
    }

    /// Decode token IDs while retaining explicit special tokens.
    fn decode(&self, token_ids: &[u32]) -> Result<String>;

    /// Decode token IDs, replacing any invalid UTF-8 with U+FFFD instead of
    /// failing.
    ///
    /// Byte-level BPE tokenizers can emit a token window that begins or ends
    /// mid-codepoint (e.g. a corpus window sampled at an arbitrary byte offset),
    /// which is not valid UTF-8. Python's `tiktoken` decodes with
    /// `errors="replace"` by default, so recorded-content reconstruction must do
    /// the same to stay byte-identical. The default implementation defers to
    /// [`decode`](Self::decode) for tokenizers whose decode is already lossy.
    fn decode_lossy(&self, token_ids: &[u32]) -> Result<String> {
        self.decode(token_ids)
    }

    /// Configured beginning-of-sequence token, if known.
    fn bos_token_id(&self) -> Option<u32>;

    /// Configured end-of-sequence token, if known.
    fn eos_token_id(&self) -> Option<u32>;

    /// Vocabulary cardinality used to keep replacement token IDs in range.
    ///
    /// Implementations that cannot expose this value may return `None`; raw
    /// prompt generation then selects a known non-EOS corpus token instead.
    fn vocab_size(&self) -> Option<u32> {
        None
    }

    /// Optional filtered token domain for synthetic random generation.
    ///
    /// Tokenizers that can distinguish ordinary vocabulary ids from
    /// special/control/otherwise-unsuitable ids should return the subset that is
    /// safe to sample directly. `None` means callers should fall back to the
    /// full `0..vocab_size` range.
    fn allowed_random_token_ids(&self) -> Option<Arc<[u32]>> {
        None
    }

    /// Number of tokens the endpoint tokenizer adds around bare input text.
    /// Native encoders default to zero because [`Self::encode`] never adds them.
    fn num_special_tokens_to_add(&self) -> usize {
        0
    }

    /// BOS, then EOS, used as a synthetic block separator when available.
    fn block_separation_token_id(&self) -> Option<u32> {
        self.bos_token_id().or_else(|| self.eos_token_id())
    }

    /// Stable diagnostic name.
    fn name(&self) -> &str;

    /// Count encoded tokens without retaining the vector.
    fn count(&self, text: &str) -> Result<usize> {
        self.encode(text).map(|tokens| tokens.len())
    }

    /// Render and tokenize chat messages when this tokenizer owns a template.
    ///
    /// `None` means chat-template accounting is unavailable and callers must
    /// use the ordinary bare-text path.
    fn apply_chat_template(
        &self,
        _messages: &[Value],
        _add_generation_prompt: bool,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }
}

/// Supported built-in tiktoken encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiktokenEncoding {
    /// GPT-4o / GPT-4.1 / GPT-5 family encoding and AIPerf's `builtin` default.
    O200kBase,
    /// GPT-OSS harmony encoding.
    O200kHarmony,
    /// GPT-4 / GPT-3.5 family encoding.
    Cl100kBase,
    /// Codex/text-davinci encoding.
    P50kBase,
    /// Edit-model encoding.
    P50kEdit,
    /// GPT-2 / GPT-3 encoding.
    R50kBase,
}

impl TiktokenEncoding {
    /// Canonical encoding name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::O200kBase => "o200k_base",
            Self::O200kHarmony => "o200k_harmony",
            Self::Cl100kBase => "cl100k_base",
            Self::P50kBase => "p50k_base",
            Self::P50kEdit => "p50k_edit",
            Self::R50kBase => "r50k_base",
        }
    }

    fn bpe(self) -> &'static CoreBPE {
        match self {
            Self::O200kBase => o200k_base_singleton(),
            Self::O200kHarmony => o200k_harmony_singleton(),
            Self::Cl100kBase => cl100k_base_singleton(),
            Self::P50kBase => p50k_base_singleton(),
            Self::P50kEdit => p50k_edit_singleton(),
            Self::R50kBase => r50k_base_singleton(),
        }
    }

    const fn vocab_size(self) -> u32 {
        match self {
            Self::O200kBase => 200_019,
            Self::O200kHarmony => 201_088,
            Self::Cl100kBase => 100_277,
            Self::P50kBase => 50_281,
            Self::P50kEdit => 50_284,
            Self::R50kBase => 50_257,
        }
    }
}

impl Display for TiktokenEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for TiktokenEncoding {
    type Err = DatasetError;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().replace('-', "_").as_str() {
            "builtin" | "o200k_base" => Ok(Self::O200kBase),
            "o200k_harmony" => Ok(Self::O200kHarmony),
            "cl100k_base" => Ok(Self::Cl100kBase),
            "p50k_base" => Ok(Self::P50kBase),
            "p50k_edit" => Ok(Self::P50kEdit),
            "r50k_base" => Ok(Self::R50kBase),
            other => Err(DatasetError::Tokenizer(format!(
                "unknown tiktoken encoding {other:?}"
            ))),
        }
    }
}

/// Zero-network built-in tokenizer backed by `tiktoken-rs`.
pub struct TiktokenTokenizer {
    encoding: TiktokenEncoding,
    bpe: &'static CoreBPE,
    eos_token_id: Option<u32>,
}

impl TiktokenTokenizer {
    /// Construct a tokenizer for one built-in encoding.
    pub fn new(encoding: TiktokenEncoding) -> Self {
        let bpe = encoding.bpe();
        let eos_token_id = bpe
            .encode_with_special_tokens("<|endoftext|>")
            .into_iter()
            .next();
        Self {
            encoding,
            bpe,
            eos_token_id,
        }
    }

    /// Construct AIPerf's `builtin` `o200k_base` tokenizer.
    pub fn builtin() -> Self {
        Self::new(TiktokenEncoding::O200kBase)
    }
}

impl Default for TiktokenTokenizer {
    fn default() -> Self {
        Self::builtin()
    }
}

impl TextTokenizer for TiktokenTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.bpe.encode_with_special_tokens(text))
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.bpe
            .decode(token_ids.to_vec())
            .map_err(|error| DatasetError::Tokenizer(error.to_string()))
    }

    fn decode_lossy(&self, token_ids: &[u32]) -> Result<String> {
        // Matches Python `tiktoken`'s default `errors="replace"`: reassemble the
        // raw per-token bytes into one buffer, then substitute maximal ill-formed
        // subsequences with U+FFFD. Recorded corpus windows begin at arbitrary
        // byte offsets and are usually *not* valid UTF-8, so a strict-decode-first
        // probe would fail and redo the work on the common path; go straight to
        // the byte path and size the buffer once.
        let mut bytes = Vec::with_capacity(token_ids.len().saturating_mul(4));
        for token in self.bpe._decode_native_and_split(token_ids.to_vec()) {
            bytes.extend_from_slice(&token);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn bos_token_id(&self) -> Option<u32> {
        None
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn vocab_size(&self) -> Option<u32> {
        Some(self.encoding.vocab_size())
    }

    fn name(&self) -> &str {
        self.encoding.as_str()
    }
}

/// Candidate `tiktoken.model` filenames shipped by tiktoken-vocab repositories.
///
/// Ordered by prevalence: Kimi K2 / DeepSeek ship `tiktoken.model`, Llama-3 and
/// some Qwen checkpoints ship `tokenizer.model` (a base64-rank BPE file, not a
/// SentencePiece proto), and older Qwen ships a repo-named `*.tiktoken`. The
/// discovery walks these names; a `*.tiktoken` is matched by extension.
const TIKTOKEN_MODEL_FILENAMES: &[&str] = &["tiktoken.model", "tokenizer.model"];

/// `cl100k_base` pre-tokenization regex (GPT-4, and the default for tiktoken
/// vocab models such as Qwen), matching HuggingFace `TikTokenConverter`.
const CL100K_BASE_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Kimi K2 pre-tokenization regex from `tokenization_kimi.py`. Kimi/DeepSeek-V3
/// tiktoken vocabs need this Han-aware split to reproduce upstream token ids.
const KIMI_PATTERN: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Fallback reserved special-token slot count when `config.json` has no
/// `vocab_size`. 256 is the `num_reserved_special_tokens` used by Kimi K2 and
/// Llama-3, the common convention for modern tiktoken-vocab HF tokenizers.
const FALLBACK_NUM_RESERVED_SPECIAL_TOKENS: u32 = 256;

/// Locate a native `tiktoken.model` / `tokenizer.model` / `*.tiktoken` BPE vocab
/// file in a model directory, or `None` when the directory has none.
///
/// A repository that ships a `tokenizer.json` is loaded through the HuggingFace
/// fast path instead; this is only consulted when that file is absent.
pub fn find_tiktoken_model_file(directory: &Path) -> Option<PathBuf> {
    for name in TIKTOKEN_MODEL_FILENAMES {
        let candidate = directory.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    // Fall back to any `*.tiktoken` (older Qwen ships a repo-named one, e.g.
    // `qwen.tiktoken`).
    std::fs::read_dir(directory).ok().and_then(|entries| {
        entries
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                path.is_file()
                    && path
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("tiktoken"))
            })
    })
}

/// One `added_tokens_decoder` entry from `tokenizer_config.json`.
#[derive(Debug, Clone, Deserialize)]
struct TiktokenAddedToken {
    content: String,
    /// HuggingFace marks placeholder/control tokens `"special": true`. Defaults
    /// to `false` (matching the `AddedToken` default) when the field is absent.
    #[serde(default)]
    #[allow(dead_code)]
    special: bool,
}

/// Minimal `tokenizer_config.json` view for the tiktoken loader.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct TiktokenTokenizerConfig {
    added_tokens_decoder: FxHashMap<u32, TiktokenAddedToken>,
    bos_token: Option<Value>,
    eos_token: Option<Value>,
}

/// Minimal `config.json` view for the tiktoken loader.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct TiktokenModelConfig {
    model_type: Option<String>,
    vocab_size: Option<u32>,
}

/// Native `tiktoken.model` tokenizer for Kimi/Qwen/DeepSeek-class models.
///
/// Loads the base64-rank BPE vocabulary (one `<base64-token-bytes> <rank>` pair
/// per line — OpenAI's tiktoken format, also emitted by DeepSeek and Kimi K2),
/// registers the model's special tokens over the reserved id range that follows
/// the BPE vocabulary, and builds a working [`CoreBPE`]. This is the pure-Rust
/// client-side equivalent of loading a `trust_remote_code` Python tokenizer over
/// a tiktoken vocab, so AIPerf tokenizes these models without a server round-trip.
pub struct NativeTiktokenTokenizer {
    bpe: CoreBPE,
    name: String,
    /// Exclusive upper bound on decodable ids (base BPE tokens + reserved
    /// special slots), used for `vocab_size` range checks.
    vocab_upper_bound: u32,
    /// Dense `id -> raw bytes` map for lossy decode. Base tokens hold their BPE
    /// bytes; reserved/special slots hold the token string bytes. Empty entries
    /// are ids no source named.
    decoder: Vec<Vec<u8>>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    allowed_random_token_ids: Arc<[u32]>,
}

impl NativeTiktokenTokenizer {
    /// Load from a model directory that contains a `tiktoken.model` /
    /// `tokenizer.model` / `*.tiktoken` file, reading sibling
    /// `tokenizer_config.json` and `config.json` for special tokens, vocab size,
    /// and the pre-tokenization regex.
    pub fn from_directory(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let model_path = find_tiktoken_model_file(directory).ok_or_else(|| {
            DatasetError::Tokenizer(format!(
                "no tiktoken model file ({}) found in {}",
                TIKTOKEN_MODEL_FILENAMES.join(", "),
                directory.display()
            ))
        })?;
        Self::from_model_file(&model_path, directory)
    }

    /// Load from an explicit BPE vocab `model_path`, reading optional sibling
    /// config files from `directory`.
    pub fn from_model_file(model_path: &Path, directory: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(model_path).map_err(|error| {
            DatasetError::Tokenizer(format!(
                "reading tiktoken model {}: {error}",
                model_path.display()
            ))
        })?;

        let mut encoder: TiktokenFxHashMap<Vec<u8>, u32> =
            TiktokenFxHashMap::with_capacity_and_hasher(
                content.lines().count(),
                Default::default(),
            );
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let (Some(token_b64), Some(rank_str)) = (parts.next(), parts.next()) else {
                continue;
            };
            let token_bytes = base64::engine::general_purpose::STANDARD
                .decode(token_b64)
                .map_err(|error| {
                    DatasetError::Tokenizer(format!(
                        "invalid base64 token in {}: {error}",
                        model_path.display()
                    ))
                })?;
            let rank: u32 = rank_str.parse().map_err(|error| {
                DatasetError::Tokenizer(format!(
                    "invalid rank in {}: {error}",
                    model_path.display()
                ))
            })?;
            encoder.insert(token_bytes, rank);
        }
        if encoder.is_empty() {
            return Err(DatasetError::Tokenizer(format!(
                "tiktoken model {} contains no tokens",
                model_path.display()
            )));
        }

        // Ranks are token ids and need not be contiguous, so the base-vocab
        // boundary is `max_rank + 1`, not `encoder.len()`.
        let num_base_tokens = encoder
            .values()
            .copied()
            .max()
            .map_or(0, |max| max.saturating_add(1));

        let tokenizer_config: TiktokenTokenizerConfig =
            read_optional_json(&directory.join("tokenizer_config.json")).unwrap_or_default();
        let model_config: TiktokenModelConfig =
            read_optional_json(&directory.join("config.json")).unwrap_or_default();

        let added_by_id = &tokenizer_config.added_tokens_decoder;
        let max_added_id = added_by_id.keys().copied().max().unwrap_or(0);
        let reserved_end = model_config
            .vocab_size
            .unwrap_or_else(|| num_base_tokens.saturating_add(FALLBACK_NUM_RESERVED_SPECIAL_TOKENS))
            .max(num_base_tokens)
            .max(max_added_id.saturating_add(1));

        // Build the special-token encoder over the reserved id range. Named
        // entries come from `added_tokens_decoder`; unnamed reserved slots get
        // Python-compatible `<|reserved_token_{id}|>` placeholders so any
        // sampled id in range still decodes (matching `tokenization_kimi.py`).
        let mut special_tokens_encoder: TiktokenFxHashMap<String, u32> =
            TiktokenFxHashMap::default();
        for id in num_base_tokens..reserved_end {
            let name = match added_by_id.get(&id) {
                Some(token) => token.content.clone(),
                None => format!("<|reserved_token_{id}|>"),
            };
            special_tokens_encoder.insert(name, id);
        }
        let mut special_token_ids = vec![false; reserved_end as usize];
        for &id in special_tokens_encoder.values() {
            if let Some(slot) = special_token_ids.get_mut(id as usize) {
                *slot = true;
            }
        }

        // Dense id -> bytes decoder for lossy reconstruction.
        let mut decoder = vec![Vec::new(); reserved_end as usize];
        for (bytes, &id) in &encoder {
            if let Some(slot) = decoder.get_mut(id as usize) {
                *slot = bytes.clone();
            }
        }
        for (name, &id) in &special_tokens_encoder {
            if let Some(slot) = decoder.get_mut(id as usize) {
                *slot = name.as_bytes().to_vec();
            }
        }

        // Resolve BOS/EOS by name through the special-token map.
        let name_to_id: FxHashMap<&str, u32> = special_tokens_encoder
            .iter()
            .map(|(name, &id)| (name.as_str(), id))
            .collect();
        let bos_token_id = special_token_name(tokenizer_config.bos_token.as_ref())
            .and_then(|n| name_to_id.get(n.as_str()).copied());
        let eos_token_id = special_token_name(tokenizer_config.eos_token.as_ref())
            .and_then(|n| name_to_id.get(n.as_str()).copied());

        let pattern = match model_config.model_type.as_deref() {
            Some("kimi" | "kimi_k2" | "kimi_k25" | "deepseek_v3") => KIMI_PATTERN,
            _ => CL100K_BASE_PATTERN,
        };

        let bpe = CoreBPE::new(encoder, special_tokens_encoder, pattern).map_err(|error| {
            DatasetError::Tokenizer(format!(
                "building tiktoken BPE from {}: {error}",
                model_path.display()
            ))
        })?;
        let allowed_random_token_ids =
            native_allowed_random_token_ids(&decoder, &special_token_ids);

        Ok(Self {
            bpe,
            name: directory.display().to_string(),
            vocab_upper_bound: reserved_end,
            decoder,
            bos_token_id,
            eos_token_id,
            allowed_random_token_ids,
        })
    }
}

/// Read and parse an optional JSON metadata file, returning `None` when it is
/// missing or unparsable (best-effort enrichment, never fatal).
fn read_optional_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Filter a local tiktoken vocabulary down to ordinary decodable token ids.
///
/// Random prompt generation samples only ids whose standalone bytes are valid
/// UTF-8 and are not registered special/reserved tokens.
fn native_allowed_random_token_ids(decoder: &[Vec<u8>], special_token_ids: &[bool]) -> Arc<[u32]> {
    decoder
        .iter()
        .enumerate()
        .filter_map(|(id, bytes)| {
            (!special_token_ids.get(id).copied().unwrap_or(false)
                && !bytes.is_empty()
                && std::str::from_utf8(bytes).is_ok())
            .then_some(id as u32)
        })
        .collect::<Vec<_>>()
        .into()
}

/// Extract the string content of a `bos_token`/`eos_token` field, tolerating
/// either a bare string or an `AddedToken`-style `{ "content": "…" }` object.
fn special_token_name(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(token) => Some(token.clone()),
        Value::Object(object) => object.get("content")?.as_str().map(str::to_owned),
        _ => None,
    }
}

impl TextTokenizer for NativeTiktokenTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // `encode_with_special_tokens` recognizes the registered special tokens
        // in the input, matching Python `encode(allowed_special="all")`.
        Ok(self.bpe.encode_with_special_tokens(text))
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.decode_lossy(token_ids)
    }

    fn decode_lossy(&self, token_ids: &[u32]) -> Result<String> {
        // Reassemble raw per-id bytes and substitute ill-formed subsequences with
        // U+FFFD, matching Python `tiktoken`'s default `errors="replace"`. Ids at
        // or above the vocabulary bound have no bytes and are skipped.
        let mut bytes = Vec::with_capacity(token_ids.len().saturating_mul(4));
        for &id in token_ids {
            if let Some(token_bytes) = self.decoder.get(id as usize) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn vocab_size(&self) -> Option<u32> {
        Some(self.vocab_upper_bound)
    }

    fn allowed_random_token_ids(&self) -> Option<Arc<[u32]>> {
        Some(self.allowed_random_token_ids.clone())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Local Hugging Face `tokenizer.json` implementation.
///
/// Encoding and decoding run through `dynamo-tokenizers` (`ai-dynamo/frontend-crates`),
/// which wraps the HF `tokenizers` library — token ids are byte-identical to a
/// direct `tokenizers` load. Chat templates render through `dynamo-renderer`'s
/// [`PromptFormatter`] (minijinja over `tokenizer_config.json`). Special-token ids
/// and vocabulary size, which `dynamo-tokenizers` does not expose, are read from a
/// parallel [`HfTokenizer`] loaded once from the same `tokenizer.json`.
pub struct HuggingFaceTokenizer {
    inner: DynamoHuggingFaceTokenizer,
    formatter: Option<PromptFormatter>,
    name: String,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    vocab_size: Option<u32>,
    allowed_random_token_ids: Option<Arc<[u32]>>,
    num_special_tokens_to_add: usize,
}

impl HuggingFaceTokenizer {
    /// Load a standalone `tokenizer.json` file.
    ///
    /// A bare `tokenizer.json` carries no special-token map or chat template, so
    /// BOS/EOS resolve to `None` and no chat-template formatter is built; use
    /// [`Self::from_directory`] to also read `tokenizer_config.json`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let (inner, introspect) = load_tokenizer(path)?;
        Ok(Self {
            inner,
            formatter: None,
            name: path.display().to_string(),
            bos_token_id: None,
            eos_token_id: None,
            vocab_size: vocab_size_of(&introspect),
            allowed_random_token_ids: allowed_random_token_ids_of(&introspect),
            num_special_tokens_to_add: num_special_tokens_to_add_of(&introspect),
        })
    }

    /// Load `tokenizer.json` plus BOS/EOS declarations and the chat template from a
    /// model directory.
    pub fn from_directory(path: impl AsRef<Path>) -> Result<Self> {
        let directory = path.as_ref();
        let tokenizer_path = directory.join("tokenizer.json");
        let (inner, introspect) = load_tokenizer(&tokenizer_path)?;
        let mut tokenizer = Self {
            inner,
            formatter: None,
            name: directory.display().to_string(),
            bos_token_id: None,
            eos_token_id: None,
            vocab_size: vocab_size_of(&introspect),
            allowed_random_token_ids: allowed_random_token_ids_of(&introspect),
            num_special_tokens_to_add: num_special_tokens_to_add_of(&introspect),
        };
        let config_path = directory.join("tokenizer_config.json");
        let tokenizer_config = if config_path.is_file() {
            Some(serde_json::from_slice::<Value>(&std::fs::read(
                &config_path,
            )?)?)
        } else {
            None
        };
        tokenizer.formatter = build_prompt_formatter(directory, tokenizer_config.as_ref());
        tokenizer.bos_token_id =
            resolve_special_token(directory, &introspect, tokenizer_config.as_ref(), "bos");
        tokenizer.eos_token_id =
            resolve_special_token(directory, &introspect, tokenizer_config.as_ref(), "eos");
        Ok(tokenizer)
    }

    /// Override special token IDs when model metadata is supplied separately.
    pub fn with_special_token_ids(mut self, bos: Option<u32>, eos: Option<u32>) -> Self {
        self.bos_token_id = bos;
        self.eos_token_id = eos;
        self
    }

    /// Return the source path used for diagnostics.
    pub fn source_path(&self) -> PathBuf {
        PathBuf::from(&self.name)
    }
}

/// Load one `tokenizer.json` into both the dynamo encode/decode wrapper and a
/// parallel [`HfTokenizer`] used only for special-token / vocab introspection.
///
/// `dynamo-tokenizers` owns no accessor for its inner tokenizer, so the second
/// load is how AIPerf reads `vocab_size` / `token_to_id`. It is a one-time cost at
/// composition, off every dispatch path.
fn load_tokenizer(path: &Path) -> Result<(DynamoHuggingFaceTokenizer, HfTokenizer)> {
    let path_text = path.to_str().ok_or_else(|| {
        DatasetError::Tokenizer(format!(
            "tokenizer path is not valid UTF-8: {}",
            path.display()
        ))
    })?;
    let inner = DynamoHuggingFaceTokenizer::from_file(path_text)
        .map_err(|error| DatasetError::Tokenizer(error.to_string()))?;
    let introspect = HfTokenizer::from_file(path).map_err(|error| {
        DatasetError::Tokenizer(format!("loading tokenizer {path_text:?}: {error}"))
    })?;
    Ok((inner, introspect))
}

/// Vocabulary cardinality including added tokens, clamped into `u32`.
fn vocab_size_of(tokenizer: &HfTokenizer) -> Option<u32> {
    u32::try_from(tokenizer.get_vocab_size(true)).ok()
}

/// Exclude Hugging Face added special tokens from direct random sampling.
fn allowed_random_token_ids_of(tokenizer: &HfTokenizer) -> Option<Arc<[u32]>> {
    let vocab_size = vocab_size_of(tokenizer)?;
    let special_ids: std::collections::HashSet<u32> = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter_map(|(id, token)| token.special.then_some(id))
        .collect();
    Some(
        (0..vocab_size)
            .filter(|id| !special_ids.contains(id))
            .collect::<Vec<_>>()
            .into(),
    )
}

fn num_special_tokens_to_add_of(tokenizer: &HfTokenizer) -> usize {
    tokenizer
        .get_post_processor()
        .map_or(0, |processor| processor.added_tokens(false))
}

/// Build a chat-template formatter for a model directory.
///
/// The template lives either inline in `tokenizer_config.json` (`chat_template`)
/// or, in the newer HF layout, in a sibling `chat_template.jinja`; the `.jinja`
/// file is merged into the config when the key is absent. Returns `None` when no
/// template is found or parsed; callers then use the bare-text path.
fn build_prompt_formatter(directory: &Path, config: Option<&Value>) -> Option<PromptFormatter> {
    let mut config = config
        .cloned()
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
    if config.get("chat_template").is_none() {
        let jinja = std::fs::read_to_string(directory.join("chat_template.jinja")).ok()?;
        config
            .as_object_mut()?
            .insert("chat_template".to_string(), Value::String(jinja));
    }
    let chat_template = ChatTemplate::deserialize(&config).ok()?;
    PromptFormatter::from_parts(chat_template, ContextMixins::default(), false).ok()
}

impl TextTokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text)
            .map(|encoding| encoding.token_ids().to_vec())
            .map_err(|error| DatasetError::Tokenizer(error.to_string()))
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.inner
            .encode_batch(texts)
            .map(|encodings| {
                encodings
                    .into_iter()
                    .map(|encoding| encoding.token_ids().to_vec())
                    .collect()
            })
            .map_err(|error| DatasetError::Tokenizer(error.to_string()))
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.inner
            .decode(token_ids, false)
            .map(Into::into)
            .map_err(|error| DatasetError::Tokenizer(error.to_string()))
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn vocab_size(&self) -> Option<u32> {
        self.vocab_size
    }

    fn allowed_random_token_ids(&self) -> Option<Arc<[u32]>> {
        self.allowed_random_token_ids.clone()
    }

    fn num_special_tokens_to_add(&self) -> usize {
        self.num_special_tokens_to_add
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn apply_chat_template(
        &self,
        messages: &[Value],
        add_generation_prompt: bool,
    ) -> Result<Option<Vec<u32>>> {
        let Some(PromptFormatter::OAI(formatter)) = self.formatter.as_ref() else {
            return Ok(None);
        };
        let request = ChatLikeRequest {
            messages: JinjaValue::from_serialize(messages),
            add_generation_prompt,
        };
        let rendered = match formatter.render(&request) {
            Ok(rendered) => rendered,
            Err(_) => return Ok(None),
        };
        match self.encode(&rendered) {
            Ok(tokens) => Ok(Some(tokens)),
            Err(_) => Ok(None),
        }
    }
}

/// Minimal [`OAIChatLikeRequest`] over AIPerf's `serde_json` message array.
///
/// `dynamo-renderer`'s formatter reads only the message value and the
/// generation-prompt flag; every other request facet defaults to absent.
struct ChatLikeRequest {
    messages: JinjaValue,
    add_generation_prompt: bool,
}

impl OAIChatLikeRequest for ChatLikeRequest {
    fn model(&self) -> String {
        String::new()
    }

    fn messages(&self) -> JinjaValue {
        self.messages.clone()
    }

    fn should_add_generation_prompt(&self) -> bool {
        self.add_generation_prompt
    }
}

/// Resolve a model's BOS or EOS token id from the HuggingFace metadata files.
///
/// Uses Hugging Face precedence: a `{kind}_token` string in
/// `tokenizer_config.json`, then in
/// `special_tokens_map.json`, then a numeric `{kind}_token_id` in `config.json` or
/// `generation_config.json` — where GPT-2-style repos (whose `tokenizer_config.json`
/// is only `{"model_max_length": …}`) declare theirs. `None` when no source names one.
fn resolve_special_token(
    directory: &Path,
    introspect: &HfTokenizer,
    tokenizer_config: Option<&Value>,
    kind: &str,
) -> Option<u32> {
    let string_field = format!("{kind}_token");
    let id_field = format!("{kind}_token_id");

    if let Some(config) = tokenizer_config
        && let Some(id) = special_token_id(introspect, config, &string_field)
    {
        return Some(id);
    }
    if let Some(map) = read_json_file(&directory.join("special_tokens_map.json"))
        && let Some(id) = special_token_id(introspect, &map, &string_field)
    {
        return Some(id);
    }
    for file in ["config.json", "generation_config.json"] {
        if let Some(config) = read_json_file(&directory.join(file))
            && let Some(id) = config.get(&id_field).and_then(numeric_token_id)
        {
            return Some(id);
        }
    }
    None
}

/// Read and parse a best-effort JSON metadata file, ignoring a missing or invalid one.
fn read_json_file(path: &Path) -> Option<Value> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// Extract a numeric token id, tolerating the array form some
/// `generation_config.json` files use for `eos_token_id`.
fn numeric_token_id(value: &Value) -> Option<u32> {
    match value {
        Value::Array(items) => items.iter().find_map(numeric_token_id),
        _ => u32::try_from(value.as_u64()?).ok(),
    }
}

/// Resolve a special-token *string* declaration (`tokenizer_config.json` /
/// `special_tokens_map.json`) to its id.
///
/// Accepts either a bare string token or an `AddedToken`-style object with a
/// `content` field, then maps it through the introspection tokenizer.
fn special_token_id(tokenizer: &HfTokenizer, config: &Value, field: &str) -> Option<u32> {
    let value = config.get(field)?;
    let token = match value {
        Value::String(token) => token.as_str(),
        Value::Object(object) => object.get("content")?.as_str()?,
        _ => return None,
    };
    tokenizer.token_to_id(token)
}

/// Tokenizer that offloads encoding and decoding to an inference server's own
/// `/tokenize` and `/detokenize` HTTP endpoints.
///
/// This variant lets a run be tokenized by the exact vocabulary the server uses
/// even when that tokenizer is not available locally: composition asks the
/// server for the token ids of each segment and, where a decode is required,
/// asks it to reconstruct text from ids.
///
/// Every call goes through AIPerf's shared native [`HttpTransport`] — the same
/// Clock-injected client the dataset/hub loader and server-metrics scraper use
/// for control-plane HTTP — so it inherits that stack's direct, proxy-free
/// connections (loopback benchmarking must never route through an ambient
/// proxy) and its Clock-enforced connect/request deadlines.
///
/// `HttpTransport` is `!Send` and drives on a current-thread `LocalSet`, but
/// this tokenizer is wrapped in a `Send + Sync` `Arc<dyn TextTokenizer>` and is
/// exercised on the online hot path (per-response `count`, per-request chat
/// templating). So a single long-lived worker thread owns one runtime, one
/// `LocalSet`, and one `HttpTransport` — its connection pool and keep-alive
/// persist across calls — and this handle holds only the channel sender. Each
/// synchronous `encode`/`decode` hands one request to that worker and blocks on
/// a reply channel: this cannot re-enter the ambient composition runtime or
/// nested-`block_on`-panic, and the worker's own reactor drives the I/O while
/// the caller waits the network round-trip. Calls are serviced serially — the
/// worker processes one request to completion before the next — so a stalled
/// request delays any others queued behind it, each bounded by the per-round-trip
/// deadline. The worker shuts down cleanly when this handle drops (the request
/// channel closes and the loop exits).
///
/// Only `http`/`https` origins are accepted; any other scheme is rejected at
/// construction. BOS/EOS and vocabulary size are not exposed by the two
/// endpoints, so they resolve to `None` and raw prompt generation falls back to
/// a corpus token.
pub struct ServerTokenizer {
    /// Absolute URL of the tokenize endpoint (default `<base>/tokenize`).
    tokenize_url: String,
    /// Absolute URL of the detokenize endpoint (default `<base>/detokenize`).
    detokenize_url: String,
    /// Model identity forwarded so the server selects the matching tokenizer.
    ///
    /// Absent when the server hosts a single tokenizer and needs no selector.
    model: Option<String>,
    /// Stable diagnostic name (`server:<base-url>`).
    name: String,
    /// Sender to the long-lived transport worker. `None` only transiently during
    /// [`Drop`], which closes the channel to signal shutdown.
    request_tx: Option<tokio::sync::mpsc::UnboundedSender<TransportRequest>>,
    /// Join handle for the worker thread, taken and joined on [`Drop`].
    worker_handle: Option<std::thread::JoinHandle<()>>,
}

/// One POST job handed to the transport worker with a channel for its reply.
struct TransportRequest {
    /// Absolute endpoint URL to POST to.
    url: String,
    /// Serialized JSON request body.
    payload: Vec<u8>,
    /// One-shot reply channel carrying the response body bytes or an error.
    reply: std::sync::mpsc::Sender<Result<Vec<u8>>>,
}

impl ServerTokenizer {
    /// Default Clock-enforced connect/request/total deadline per round-trip.
    const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

    /// Build a server tokenizer targeting `base_url` (e.g. `http://host:8000`).
    ///
    /// `model` is forwarded verbatim in every request body when present. The
    /// tokenize and detokenize endpoints default to `<base>/tokenize` and
    /// `<base>/detokenize`. The long-lived transport worker thread is spawned
    /// here and lives until this handle drops.
    pub fn new(base_url: &str, model: Option<String>) -> Result<Self> {
        let parsed = url::Url::parse(base_url).map_err(|error| {
            DatasetError::Tokenizer(format!(
                "invalid server tokenizer URL {base_url:?}: {error}"
            ))
        })?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(DatasetError::Tokenizer(format!(
                "server tokenizer supports only http/https origins, got {:?} in {base_url:?}",
                parsed.scheme()
            )));
        }
        if parsed.host_str().is_none() {
            return Err(DatasetError::Tokenizer(format!(
                "server tokenizer URL {base_url:?} has no host"
            )));
        }
        let base = base_url.trim_end_matches('/');
        let timeout_ns = i64::try_from(Self::DEFAULT_TIMEOUT.as_nanos()).unwrap_or(i64::MAX);
        let (request_tx, worker_handle) = spawn_transport_worker(timeout_ns)?;
        Ok(Self {
            tokenize_url: format!("{base}/tokenize"),
            detokenize_url: format!("{base}/detokenize"),
            model,
            name: format!("server:{base_url}"),
            request_tx: Some(request_tx),
            worker_handle: Some(worker_handle),
        })
    }

    /// POST a JSON body to `url` and decode the reply.
    fn post_json<B: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
        body: &B,
    ) -> Result<R> {
        let payload = serde_json::to_vec(body)?;
        let raw = self.exchange(url, payload)?;
        serde_json::from_slice(&raw).map_err(|error| {
            DatasetError::Tokenizer(format!(
                "server tokenizer {url} returned undecodable JSON: {error}"
            ))
        })
    }

    /// Hand one POST to the long-lived transport worker and block on its reply.
    ///
    /// Blocking is on a plain channel (not `block_on`), so it is safe from
    /// within the ambient composition runtime and never re-enters it; the
    /// worker's own reactor drives the request while this thread waits the
    /// network round-trip.
    fn exchange(&self, url: &str, payload: Vec<u8>) -> Result<Vec<u8>> {
        let (reply, reply_rx) = std::sync::mpsc::channel();
        let request_tx = self.request_tx.as_ref().ok_or_else(|| {
            DatasetError::Tokenizer("server tokenizer worker is shutting down".to_string())
        })?;
        request_tx
            .send(TransportRequest {
                url: url.to_string(),
                payload,
                reply,
            })
            .map_err(|_| {
                DatasetError::Tokenizer("server tokenizer worker is not running".to_string())
            })?;
        reply_rx.recv().map_err(|_| {
            DatasetError::Tokenizer(
                "server tokenizer worker dropped the request before replying".to_string(),
            )
        })?
    }
}

impl Drop for ServerTokenizer {
    fn drop(&mut self) {
        // Close the request channel so the worker's `recv().await` yields `None`
        // and its runtime tears down, then join the now-exiting thread.
        self.request_tx = None;
        if let Some(worker_handle) = self.worker_handle.take() {
            let _ = worker_handle.join();
        }
    }
}

/// Spawn the long-lived transport worker: one thread owning one current-thread
/// runtime, one `LocalSet`, and one [`HttpTransport`] whose connection pool and
/// keep-alive persist for the tokenizer's lifetime.
///
/// Returns the request sender and the thread's join handle. Construction blocks
/// until the worker signals that its runtime built successfully, so a runtime
/// failure surfaces as a clear construction error rather than a later send error.
fn spawn_transport_worker(
    timeout_ns: i64,
) -> Result<(
    tokio::sync::mpsc::UnboundedSender<TransportRequest>,
    std::thread::JoinHandle<()>,
)> {
    // Unbounded is safe despite no backpressure: each caller blocks on its own
    // reply channel before it can enqueue another job, so the queue depth is
    // capped by the number of concurrent callers, not by producer speed.
    let (request_tx, mut request_rx) = tokio::sync::mpsc::unbounded_channel::<TransportRequest>();
    let (ready, ready_rx) = std::sync::mpsc::channel::<std::result::Result<(), String>>();
    let worker = std::thread::Builder::new()
        .name("aiperf-server-tokenizer".to_string())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let _ = ready.send(Ok(()));
            let local = tokio::task::LocalSet::new();
            runtime.block_on(local.run_until(async move {
                let clock: Rc<dyn Clock> = RealClock::new();
                let client = ClientConfig {
                    connect_timeout_ns: Some(timeout_ns),
                    request_timeout_ns: Some(timeout_ns),
                    total_timeout_ns: Some(timeout_ns),
                    ..ClientConfig::default()
                };
                let transport =
                    HttpTransport::new(clock, client).with_user_agent("aiperf-tokenizer/0");
                // Await requests on the async channel so the reactor keeps
                // polling the connection pool (keep-alive) between jobs; a
                // blocking `recv` would park the reactor and stall idle sockets.
                // The loop exits when the sender drops (tokenizer shutdown).
                while let Some(request) = request_rx.recv().await {
                    let result = dispatch_once(&transport, &request.url, request.payload).await;
                    let _ = request.reply.send(result);
                }
            }));
        })
        .map_err(DatasetError::Io)?;
    match ready_rx.recv() {
        Ok(Ok(())) => Ok((request_tx, worker)),
        Ok(Err(message)) => Err(DatasetError::Tokenizer(format!(
            "server tokenizer worker failed to start: {message}"
        ))),
        Err(_) => Err(DatasetError::Tokenizer(
            "server tokenizer worker exited before signaling readiness".to_string(),
        )),
    }
}

/// Dispatch one POST over the shared transport and classify the terminal record.
///
/// A non-2xx status is surfaced with a bounded body snippet; a missing status
/// (connect failure or Clock-enforced deadline) is surfaced as a transport error.
async fn dispatch_once(transport: &HttpTransport, url: &str, payload: Vec<u8>) -> Result<Vec<u8>> {
    let record = transport
        .send_request_bytes(
            &RequestConfig::new(url.to_string()),
            Bytes::from(payload),
            false,
            |_| {},
        )
        .await;

    let body = record
        .responses
        .into_iter()
        .find_map(|response| match response {
            Response::Text(text) => Some(text.text),
            Response::Sse(_) => None,
        });

    match record.status {
        Some(status) if (200..300).contains(&status) => Ok(body.unwrap_or_default().into_bytes()),
        Some(status) => {
            let snippet: String = body.unwrap_or_default().chars().take(200).collect();
            Err(DatasetError::Tokenizer(format!(
                "server tokenizer {url} returned HTTP {status}: {snippet}"
            )))
        }
        None => {
            let detail = record
                .error
                .map(|error| error.message)
                .unwrap_or_else(|| "no response".to_string());
            Err(DatasetError::Tokenizer(format!(
                "server tokenizer request to {url} failed: {detail}"
            )))
        }
    }
}

impl TextTokenizer for ServerTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let reply: EncodeReply = self.post_json(
            &self.tokenize_url,
            &EncodeQuery {
                model: self.model.as_deref(),
                prompt: text,
                add_special_tokens: false,
            },
        )?;
        reply
            .tokens
            .into_iter()
            .map(|id| {
                u32::try_from(id).map_err(|_| {
                    DatasetError::Tokenizer(format!(
                        "server tokenizer returned out-of-range token id {id}"
                    ))
                })
            })
            .collect()
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let reply: DecodeReply = self.post_json(
            &self.detokenize_url,
            &DecodeQuery {
                model: self.model.as_deref(),
                tokens: token_ids,
            },
        )?;
        reply.text().ok_or_else(|| {
            DatasetError::Tokenizer(
                "server detokenize reply had neither a `prompt` nor `text` field".to_string(),
            )
        })
    }

    fn bos_token_id(&self) -> Option<u32> {
        None
    }

    fn eos_token_id(&self) -> Option<u32> {
        None
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Tokenize request body sent to `/tokenize`.
///
/// Field names follow the server's wire contract; `add_special_tokens` is fixed
/// to `false` so the counts mirror [`TextTokenizer::encode`]'s no-special-token
/// contract used everywhere else in composition.
#[derive(Serialize)]
struct EncodeQuery<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    prompt: &'a str,
    add_special_tokens: bool,
}

/// Tokenize reply parsed from `/tokenize`.
#[derive(Deserialize)]
struct EncodeReply {
    tokens: Vec<u64>,
}

/// Detokenize request body sent to `/detokenize`.
#[derive(Serialize)]
struct DecodeQuery<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    tokens: &'a [u32],
}

/// Detokenize reply parsed from `/detokenize`.
///
/// Servers name the reconstructed text either `prompt` or `text`; accept both.
#[derive(Deserialize)]
struct DecodeReply {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

impl DecodeReply {
    /// The reconstructed text under whichever field the server populated.
    fn text(self) -> Option<String> {
        self.prompt.or(self.text)
    }
}

/// Test tokenizer that encodes to a fixed token run and refuses to decode.
///
/// Shared by the raw-token composition/generation tests (in `dataset::prompt`
/// and `dataset::loader::synthetic`) that must never fall back through a decode
/// path.
#[cfg(test)]
pub(crate) struct NoDecodeTokenizer;

#[cfg(test)]
impl TextTokenizer for NoDecodeTokenizer {
    fn encode(&self, _text: &str) -> Result<Vec<u32>> {
        Ok(vec![9, 10, 9, 11])
    }

    fn decode(&self, _token_ids: &[u32]) -> Result<String> {
        panic!("raw-token composition must not decode")
    }

    fn bos_token_id(&self) -> Option<u32> {
        None
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(9)
    }

    fn vocab_size(&self) -> Option<u32> {
        Some(12)
    }

    fn name(&self) -> &str {
        "no-decode"
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use super::*;

    struct ScalarOnlyTokenizer;

    impl TextTokenizer for ScalarOnlyTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(&self, _token_ids: &[u32]) -> Result<String> {
            Err(DatasetError::Tokenizer(
                "decode is not used by this test".into(),
            ))
        }

        fn bos_token_id(&self) -> Option<u32> {
            None
        }

        fn eos_token_id(&self) -> Option<u32> {
            None
        }

        fn name(&self) -> &str {
            "scalar-only"
        }
    }

    #[test]
    fn text_tokenizer_default_batch_is_ordered_and_empty_safe() {
        let tokenizer = ScalarOnlyTokenizer;
        let texts = ["second", "first", "third value"];
        let expected = texts
            .iter()
            .map(|text| tokenizer.encode(text))
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(tokenizer.encode_batch(&texts).unwrap(), expected);
        assert_eq!(tokenizer.encode_batch(&[]).unwrap(), Vec::<Vec<u32>>::new());
    }

    #[test]
    fn hugging_face_batch_encoding_matches_scalar_order() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("tokenizer.json"),
            r#"{
  "version":"1.0",
  "truncation":null,
  "padding":null,
  "added_tokens":[
    {"id":0,"content":"[UNK]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
    {"id":1,"content":"<s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
    {"id":2,"content":"</s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
  ],
  "normalizer":null,
  "pre_tokenizer":{"type":"Whitespace"},
  "post_processor":null,
  "decoder":null,
  "model":{"type":"WordLevel","vocab":{"[UNK]":0,"<s>":1,"</s>":2,"user":3,"hello":4,"assistant":5},"unk_token":"[UNK]"}
}"#,
        )
        .unwrap();
        let tokenizer =
            HuggingFaceTokenizer::from_file(directory.path().join("tokenizer.json")).unwrap();
        let texts = ["assistant hello", "", "user hello assistant", "hello"];
        let scalar = texts
            .iter()
            .map(|text| tokenizer.encode(text))
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(tokenizer.encode_batch(&texts).unwrap(), scalar);
        assert_eq!(tokenizer.encode_batch(&[]).unwrap(), Vec::<Vec<u32>>::new());
    }

    /// Write a minimal but format-faithful `tiktoken.model`: single-byte tokens
    /// for bytes 0..=255 (ranks 0..=255) plus a couple of multi-byte merges, so
    /// the base64-rank parser and BPE round-trip run without any pretrained asset.
    fn write_synthetic_tiktoken_model(dir: &Path) -> PathBuf {
        let engine = base64::engine::general_purpose::STANDARD;
        let mut content = String::new();
        for byte in 0u8..=255 {
            content.push_str(&format!("{} {}\n", engine.encode([byte]), byte as u32));
        }
        // A couple of higher-rank merges so encode produces multi-byte tokens.
        content.push_str(&format!("{} 256\n", engine.encode(b"hello")));
        content.push_str(&format!("{} 257\n", engine.encode(b" world")));
        let path = dir.join("tiktoken.model");
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn native_tiktoken_round_trips_and_maps_special_tokens() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_tiktoken_model(dir.path());
        // Kimi-shaped config: special tokens declared in the reserved range that
        // follows the 258-token base vocab (max rank 257 -> num_base_tokens 258).
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{
  "bos_token": "[BOS]",
  "eos_token": "[EOS]",
  "added_tokens_decoder": {
    "258": {"content": "[BOS]", "special": true},
    "259": {"content": "[EOS]", "special": true}
  }
}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "kimi_k2", "vocab_size": 512}"#,
        )
        .unwrap();

        let tokenizer = NativeTiktokenTokenizer::from_directory(dir.path()).unwrap();

        // Base-text round-trip through the parsed BPE.
        let text = "hello world";
        let ids = tokenizer.encode(text).unwrap();
        assert!(!ids.is_empty());
        assert_eq!(tokenizer.decode(&ids).unwrap(), text);

        // Special tokens resolve to their reserved ids and BOS/EOS are wired.
        assert_eq!(tokenizer.bos_token_id(), Some(258));
        assert_eq!(tokenizer.eos_token_id(), Some(259));
        assert_eq!(tokenizer.encode("[BOS]").unwrap(), vec![258]);
        assert_eq!(tokenizer.encode("[EOS]").unwrap(), vec![259]);
        // vocab_size honors config.json vocab_size (reserved upper bound).
        assert_eq!(tokenizer.vocab_size(), Some(512));
        // Token counts are stable and match the encoded length (no network).
        let repeated = "hello world hello";
        assert_eq!(
            tokenizer.count(repeated).unwrap(),
            tokenizer.encode(repeated).unwrap().len()
        );
        assert_eq!(
            tokenizer.encode(repeated).unwrap(),
            tokenizer.encode(repeated).unwrap()
        );
        assert_eq!(
            tokenizer
                .decode(&tokenizer.encode(repeated).unwrap())
                .unwrap(),
            repeated
        );
    }

    #[test]
    fn native_tiktoken_allowed_random_tokens_exclude_reserved_and_invalid_utf8_ids() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_tiktoken_model(dir.path());
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{
  "bos_token": "[BOS]",
  "eos_token": "[EOS]",
  "added_tokens_decoder": {
    "258": {"content": "[BOS]", "special": true},
    "259": {"content": "[EOS]", "special": true}
  }
}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "kimi_k2", "vocab_size": 512}"#,
        )
        .unwrap();

        let tokenizer = NativeTiktokenTokenizer::from_directory(dir.path()).unwrap();
        let allowed = tokenizer
            .allowed_random_token_ids()
            .expect("local tiktoken should expose an allowed token subset");

        assert!(allowed.contains(&65), "ASCII token should be allowed");
        assert!(
            allowed.contains(&256),
            "merged UTF-8 token should be allowed"
        );
        assert!(
            !allowed.contains(&128),
            "invalid standalone UTF-8 byte should be excluded"
        );
        assert!(
            !allowed.contains(&258),
            "reserved BOS token should be excluded"
        );
        assert!(
            !allowed.contains(&259),
            "reserved EOS token should be excluded"
        );
    }

    #[test]
    fn native_tiktoken_defaults_reserved_slots_without_config() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_tiktoken_model(dir.path());
        // No tokenizer_config.json / config.json: fall back to 256 reserved slots.
        let tokenizer = NativeTiktokenTokenizer::from_directory(dir.path()).unwrap();
        // num_base_tokens = 258 (max rank 257 + 1); + 256 reserved = 514.
        assert_eq!(tokenizer.vocab_size(), Some(514));
        assert_eq!(tokenizer.bos_token_id(), None);
        assert_eq!(
            tokenizer.decode(&tokenizer.encode("hi").unwrap()).unwrap(),
            "hi"
        );
        // An unnamed reserved slot decodes to its placeholder token.
        assert_eq!(
            tokenizer.encode("<|reserved_token_300|>").unwrap(),
            vec![300]
        );
    }

    #[test]
    fn find_tiktoken_model_file_discovers_named_variants() {
        let dir = tempfile::tempdir().unwrap();
        assert!(find_tiktoken_model_file(dir.path()).is_none());
        std::fs::write(dir.path().join("qwen.tiktoken"), "AAAA 0\n").unwrap();
        assert!(
            find_tiktoken_model_file(dir.path())
                .unwrap()
                .extension()
                .is_some_and(|ext| ext == "tiktoken")
        );
    }

    /// Deterministic in-process keep-alive `/tokenize` + `/detokenize` server for
    /// the [`ServerTokenizer`] tests.
    ///
    /// Maps each Unicode scalar to its code point as a token id and back, so a
    /// round-trip is byte-exact and token counts come verifiably from the server,
    /// not a local encoding. Serves `requests` keep-alive requests total
    /// (responses omit `Connection: close`), accepting connections in a loop until
    /// every expected request has been served. A client that reuses one pooled
    /// connection is accepted exactly once; a regression that opens a fresh socket
    /// per call is accepted again, so the returned counter distinguishes the two
    /// crisply. Each accepted socket carries a short read timeout, so a regressed
    /// client whose first socket goes idle after one request trips a fast timeout
    /// and re-enters `accept()` (incrementing the counter) instead of hanging until
    /// the transport's 30s deadline. Binds `127.0.0.1` explicitly (never
    /// `localhost`) so the client's direct connection cannot resolve to an IPv6
    /// loopback the listener does not own.
    fn spawn_keepalive_tokenize_server(
        requests: usize,
    ) -> (String, Arc<AtomicUsize>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let base_url = format!("http://127.0.0.1:{}", address.port());
        let connections = Arc::new(AtomicUsize::new(0));
        let counter = connections.clone();
        let handle = thread::spawn(move || {
            let mut served = 0usize;
            while served < requests {
                let (stream, _) = listener.accept().unwrap();
                counter.fetch_add(1, Ordering::SeqCst);
                // Bound the idle wait so a client that fails to reuse this socket
                // trips a fast timeout and falls back to accept() rather than
                // stalling the join() until the 30s transport deadline.
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut writer = stream.try_clone().unwrap();
                let mut reader = BufReader::new(stream);

                while served < requests {
                    let mut request_line = String::new();
                    match reader.read_line(&mut request_line) {
                        Ok(0) => break, // client closed this socket
                        Ok(_) => {}
                        Err(_) => break, // idle read timeout: re-enter accept()
                    }
                    let path = request_line.split_whitespace().nth(1).unwrap().to_string();

                    let mut content_length = 0usize;
                    loop {
                        let mut header = String::new();
                        reader.read_line(&mut header).unwrap();
                        if header == "\r\n" || header.is_empty() {
                            break;
                        }
                        if let Some(value) =
                            header.to_ascii_lowercase().strip_prefix("content-length:")
                        {
                            content_length = value.trim().parse().unwrap();
                        }
                    }
                    let mut body = vec![0u8; content_length];
                    reader.read_exact(&mut body).unwrap();
                    let request: Value = serde_json::from_slice(&body).unwrap();

                    let reply = if path == "/tokenize" {
                        let prompt = request["prompt"].as_str().unwrap();
                        let tokens: Vec<u32> = prompt.chars().map(|c| c as u32).collect();
                        serde_json::json!({ "count": tokens.len(), "tokens": tokens })
                    } else if path == "/detokenize" {
                        let text: String = request["tokens"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|id| char::from_u32(id.as_u64().unwrap() as u32).unwrap())
                            .collect();
                        serde_json::json!({ "prompt": text })
                    } else {
                        serde_json::json!({ "error": "unknown path" })
                    };

                    let payload = serde_json::to_vec(&reply).unwrap();
                    // No `Connection: close`: the connection stays keep-alive so a
                    // pooling client reuses it for every subsequent request.
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                        payload.len()
                    );
                    writer.write_all(response.as_bytes()).unwrap();
                    writer.write_all(&payload).unwrap();
                    writer.flush().unwrap();
                    served += 1;
                }
            }
        });
        (base_url, connections, handle)
    }

    // End-to-end: the server tokenizer must round-trip text through the live
    // `/tokenize` and `/detokenize` endpoints, and its token count must be the
    // one the server reports (never a local encoding). The three calls must be
    // served over a SINGLE accepted connection, proving the long-lived worker's
    // HttpTransport (and its pool/keep-alive) is reused across calls rather than
    // rebuilt per call.
    #[test]
    fn server_tokenizer_round_trips_and_reuses_one_connection() {
        let (base_url, connections, handle) = spawn_keepalive_tokenize_server(3);
        let tokenizer = ServerTokenizer::new(&base_url, Some("test-model".to_string())).unwrap();

        let text = "abc€";
        let tokens = tokenizer.encode(text).unwrap();
        // 'a','b','c' are one scalar each; '€' is U+20AC.
        assert_eq!(tokens, vec![97, 98, 99, 0x20AC]);
        // `count` re-encodes (second request); `decode` is the third — all three
        // over the one pooled connection.
        assert_eq!(tokenizer.count(text).unwrap(), 4);
        assert_eq!(tokenizer.decode(&tokens).unwrap(), text);
        assert_eq!(tokenizer.name(), format!("server:{base_url}"));

        handle.join().unwrap();
        assert_eq!(
            connections.load(Ordering::SeqCst),
            1,
            "three tokenizer calls must reuse a single transport connection"
        );

        // Dropping the tokenizer shuts the worker down cleanly.
        drop(tokenizer);
    }

    // An unsupported scheme is rejected at construction with a clear message.
    #[test]
    fn server_tokenizer_rejects_unsupported_scheme() {
        let error = match ServerTokenizer::new("ftp://host:21", None) {
            Ok(_) => panic!("non-http/https origin must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("only http/https origins"));
    }

    #[test]
    fn builtin_is_exact_and_round_trips_unicode() {
        let tokenizer = TiktokenTokenizer::builtin();
        let text = "hello, 世界";
        let tokens = tokenizer.encode(text).unwrap();
        assert!(!tokens.is_empty());
        assert_eq!(tokenizer.decode(&tokens).unwrap(), text);
        assert_eq!(tokenizer.name(), "o200k_base");
        assert!(tokenizer.eos_token_id().is_some());
    }

    #[test]
    fn encoding_names_follow_python_aliases() {
        assert_eq!(
            "builtin".parse::<TiktokenEncoding>().unwrap(),
            TiktokenEncoding::O200kBase
        );
        assert_eq!(
            "CL100K-BASE".parse::<TiktokenEncoding>().unwrap(),
            TiktokenEncoding::Cl100kBase
        );
        assert!("unknown".parse::<TiktokenEncoding>().is_err());
    }

    #[test]
    fn hugging_face_template_is_rendered_and_tokenized_natively() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("tokenizer.json"),
            r#"{
  "version":"1.0",
  "truncation":null,
  "padding":null,
  "added_tokens":[
    {"id":0,"content":"[UNK]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
    {"id":1,"content":"<s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
    {"id":2,"content":"</s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
  ],
  "normalizer":null,
  "pre_tokenizer":{"type":"Whitespace"},
  "post_processor":null,
  "decoder":null,
  "model":{"type":"WordLevel","vocab":{"[UNK]":0,"<s>":1,"</s>":2,"user":3,"hello":4,"assistant":5},"unk_token":"[UNK]"}
}"#,
        )
        .unwrap();
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{
  "bos_token":"<s>",
  "eos_token":"</s>",
  "chat_template":"{{ bos_token }} {% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
}"#,
        )
        .unwrap();

        let tokenizer = HuggingFaceTokenizer::from_directory(directory.path()).unwrap();
        let bare = tokenizer.encode("hello").unwrap();
        let templated = tokenizer
            .apply_chat_template(
                &[serde_json::json!({"role":"user","content":"hello"})],
                true,
            )
            .unwrap()
            .unwrap();

        assert_eq!(bare.len(), 1);
        assert_eq!(templated.len(), 4);
        assert_eq!(tokenizer.bos_token_id(), Some(1));
        assert_eq!(tokenizer.eos_token_id(), Some(2));
    }

    #[test]
    fn hugging_face_allowed_random_tokens_exclude_added_special_ids() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("tokenizer.json"),
            r#"{
  "version":"1.0",
  "truncation":null,
  "padding":null,
  "added_tokens":[
    {"id":0,"content":"[UNK]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
    {"id":1,"content":"<s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
    {"id":2,"content":"</s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
  ],
  "normalizer":null,
  "pre_tokenizer":{"type":"Whitespace"},
  "post_processor":null,
  "decoder":null,
  "model":{"type":"WordLevel","vocab":{"[UNK]":0,"<s>":1,"</s>":2,"user":3,"hello":4,"assistant":5},"unk_token":"[UNK]"}
}"#,
        )
        .unwrap();

        let tokenizer =
            HuggingFaceTokenizer::from_file(directory.path().join("tokenizer.json")).unwrap();
        let allowed = tokenizer
            .allowed_random_token_ids()
            .expect("HF tokenizer should expose an allowed token subset");

        assert_eq!(allowed.as_ref(), &[3, 4, 5]);
    }

    // The newer HF layout stores the chat template in a sibling
    // `chat_template.jinja` rather than inline in `tokenizer_config.json`
    // (what `save_pretrained` emits, and what many recent repos ship). The
    // formatter must still build from that file.
    #[test]
    fn chat_template_jinja_sidecar_is_used() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("tokenizer.json"),
            r#"{"version":"1.0","truncation":null,"padding":null,
  "added_tokens":[{"id":1,"content":"<s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
  "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,
  "model":{"type":"WordLevel","vocab":{"<s>":1,"user":3,"hello":4,"assistant":5},"unk_token":"<s>"}}"#,
        )
        .unwrap();
        // tokenizer_config.json carries NO chat_template.
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{"bos_token":"<s>"}"#,
        )
        .unwrap();
        std::fs::write(
            directory.path().join("chat_template.jinja"),
            "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}",
        )
        .unwrap();

        let tokenizer = HuggingFaceTokenizer::from_directory(directory.path()).unwrap();
        let templated = tokenizer
            .apply_chat_template(
                &[serde_json::json!({"role":"user","content":"hello"})],
                true,
            )
            .unwrap()
            .expect("sidecar chat_template.jinja must build a formatter");
        // "user hello assistant" -> [3, 4, 5].
        assert_eq!(templated, vec![3, 4, 5]);
    }

    // Validates the downloaded tokenizer against known GPT-2 BPE token IDs.
    // Network-gated because it fetches `gpt2` from the Hugging Face hub.
    #[test]
    #[ignore = "downloads the gpt2 tokenizer from the Hugging Face hub"]
    fn gpt2_download_encodes_to_known_ids() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let directory = runtime
            .block_on(crate::dataset::hf_hub::download_hugging_face_tokenizer(
                "gpt2",
            ))
            .expect("download gpt2 tokenizer");

        let tokenizer = HuggingFaceTokenizer::from_directory(&directory).unwrap();
        // Canonical GPT-2 BPE encoding of "Hello world".
        let tokens = tokenizer.encode("Hello world").unwrap();
        assert_eq!(tokens, vec![15496, 995]);
        assert_eq!(tokenizer.decode(&tokens).unwrap(), "Hello world");
        assert_eq!(tokenizer.vocab_size(), Some(50257));
        // gpt2 declares `<|endoftext|>` (id 50256) as both bos and eos.
        assert_eq!(tokenizer.eos_token_id(), Some(50256));
    }
}
