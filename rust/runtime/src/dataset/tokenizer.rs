// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokenizer seam and native Hugging Face / tiktoken implementations.
//!
//! Composition calls this trait before interning messages, making token IDs the
//! authoritative message identity and input-length source. No dispatch path
//! tokenizes again.

use std::fmt::{self, Display};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use base64::Engine as _;
use dynamo_renderer::{ChatTemplate, ContextMixins, OAIChatLikeRequest, PromptFormatter};
use dynamo_tokenizers::HuggingFaceTokenizer as DynamoHuggingFaceTokenizer;
use dynamo_tokenizers::traits::{Decoder as _, Encoder as _};
use minijinja::Value as JinjaValue;
use rustc_hash::FxHashMap;
use serde::Deserialize;
// tiktoken-rs 0.9's `CoreBPE::new` wants `rustc_hash` 1.x maps; alias the bridged
// pin so the two maps handed to it match its hasher type (the workspace
// `rustc_hash` is 2.x, a distinct `FxHashMap`).
use tiktoken_rustc_hash::FxHashMap as TiktokenFxHashMap;
use serde_json::Value;
use tiktoken_rs::{
    CoreBPE, cl100k_base_singleton, o200k_base_singleton, o200k_harmony_singleton,
    p50k_base_singleton, p50k_edit_singleton, r50k_base_singleton,
};
use tokenizers::Tokenizer as HfTokenizer;

use crate::dataset::error::{DatasetError, Result};

/// Tokenization contract used by composers and input accounting.
pub trait TextTokenizer: Send + Sync {
    /// Encode text without automatically adding model special tokens.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

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
            TiktokenFxHashMap::with_capacity_and_hasher(content.lines().count(), Default::default());
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
        let bos_token_id =
            special_token_name(tokenizer_config.bos_token.as_ref()).and_then(|n| name_to_id.get(n.as_str()).copied());
        let eos_token_id =
            special_token_name(tokenizer_config.eos_token.as_ref()).and_then(|n| name_to_id.get(n.as_str()).copied());

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

        Ok(Self {
            bpe,
            name: directory.display().to_string(),
            vocab_upper_bound: reserved_end,
            decoder,
            bos_token_id,
            eos_token_id,
        })
    }
}

/// Read and parse an optional JSON metadata file, returning `None` when it is
/// missing or unparsable (best-effort enrichment, never fatal).
fn read_optional_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
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
    use super::*;

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
        // Stable token count for a fixed input (no network).
        assert_eq!(tokenizer.count("hello world hello").unwrap(), ids.len() + 2);
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
        assert_eq!(tokenizer.decode(&tokenizer.encode("hi").unwrap()).unwrap(), "hi");
        // An unnamed reserved slot decodes to its placeholder token.
        assert_eq!(tokenizer.encode("<|reserved_token_300|>").unwrap(), vec![300]);
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
