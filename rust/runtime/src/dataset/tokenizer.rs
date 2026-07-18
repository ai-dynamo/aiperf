// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokenizer seam and native Hugging Face / tiktoken implementations.
//!
//! Composition calls this trait before interning messages, making token IDs the
//! authoritative message identity and input-length source. No dispatch path
//! tokenizes again.

use std::fmt::{self, Display};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use dynamo_renderer::{ChatTemplate, ContextMixins, OAIChatLikeRequest, PromptFormatter};
use dynamo_tokenizers::HuggingFaceTokenizer as DynamoHuggingFaceTokenizer;
use dynamo_tokenizers::traits::{Decoder as _, Encoder as _};
use minijinja::Value as JinjaValue;
use serde::{Deserialize, Serialize};
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

/// Tokenizer that offloads encoding and decoding to an inference server's own
/// `/tokenize` and `/detokenize` HTTP endpoints.
///
/// This variant lets a run be tokenized by the exact vocabulary the server uses
/// even when that tokenizer is not available locally: composition asks the
/// server for the token ids of each segment and, where a decode is required,
/// asks it to reconstruct text from ids. Every call is a single blocking
/// round-trip over a fresh direct connection, so no ambient proxy is consulted
/// (loopback benchmarking must never route through one) and no async runtime is
/// borrowed on the tokenizing path.
///
/// Only `http` origins are supported; an `https` base URL is rejected at
/// construction rather than silently downgraded. BOS/EOS and vocabulary size are
/// not exposed by the two endpoints, so they resolve to `None` and raw prompt
/// generation falls back to a corpus token.
pub struct ServerTokenizer {
    /// Direct-connect coordinates parsed once from the base URL.
    origin: ServerOrigin,
    /// Absolute path of the tokenize endpoint (default `/tokenize`).
    tokenize_path: String,
    /// Absolute path of the detokenize endpoint (default `/detokenize`).
    detokenize_path: String,
    /// Model identity forwarded so the server selects the matching tokenizer.
    ///
    /// Absent when the server hosts a single tokenizer and needs no selector.
    model: Option<String>,
    /// Per-call socket timeout for connect, write, and read.
    timeout: Duration,
    /// Stable diagnostic name (`server:<base-url>`).
    name: String,
}

/// Direct-connection coordinates for the tokenize endpoints.
///
/// Held pre-parsed so each call opens a plain `TcpStream` to `host:port` without
/// consulting `HTTP_PROXY`/`ALL_PROXY`, matching the project rule that loopback
/// traffic is never proxied.
struct ServerOrigin {
    /// Host used both for the TCP connection and the `Host` header.
    host: String,
    /// Resolved TCP port (explicit, or 80 for `http`).
    port: u16,
}

impl ServerTokenizer {
    /// Default per-call socket timeout.
    const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

    /// Build a server tokenizer targeting `base_url` (e.g. `http://host:8000`).
    ///
    /// `model` is forwarded verbatim in every request body when present. The
    /// tokenize and detokenize paths default to `/tokenize` and `/detokenize`
    /// under the base URL's origin; a base URL path is ignored so a bare
    /// endpoint origin resolves correctly.
    pub fn new(base_url: &str, model: Option<String>) -> Result<Self> {
        let parsed = url::Url::parse(base_url).map_err(|error| {
            DatasetError::Tokenizer(format!(
                "invalid server tokenizer URL {base_url:?}: {error}"
            ))
        })?;
        if parsed.scheme() != "http" {
            return Err(DatasetError::Tokenizer(format!(
                "server tokenizer supports only http origins, got {:?} in {base_url:?}",
                parsed.scheme()
            )));
        }
        let host = parsed
            .host_str()
            .ok_or_else(|| {
                DatasetError::Tokenizer(format!("server tokenizer URL {base_url:?} has no host"))
            })?
            .to_string();
        let port = parsed.port().unwrap_or(80);
        Ok(Self {
            origin: ServerOrigin { host, port },
            tokenize_path: "/tokenize".to_string(),
            detokenize_path: "/detokenize".to_string(),
            model,
            timeout: Self::DEFAULT_TIMEOUT,
            name: format!("server:{base_url}"),
        })
    }

    /// Override the tokenize and detokenize paths (defaults `/tokenize`,
    /// `/detokenize`).
    pub fn with_paths(mut self, tokenize_path: &str, detokenize_path: &str) -> Self {
        self.tokenize_path = tokenize_path.to_string();
        self.detokenize_path = detokenize_path.to_string();
        self
    }

    /// Override the per-call socket timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// POST a JSON body to `path` on the endpoint origin and decode the reply.
    ///
    /// Uses a one-shot `Connection: close` exchange over a direct socket so the
    /// blocking hot path stays trivial and proxy-free.
    fn post_json<B: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<R> {
        let payload = serde_json::to_vec(body)?;
        let raw = self.exchange(path, &payload)?;
        serde_json::from_slice(&raw).map_err(|error| {
            DatasetError::Tokenizer(format!(
                "server tokenizer {path} returned undecodable JSON: {error}"
            ))
        })
    }

    /// Perform one blocking HTTP/1.1 request/response and return the body bytes.
    fn exchange(&self, path: &str, payload: &[u8]) -> Result<Vec<u8>> {
        let address = (self.origin.host.as_str(), self.origin.port);
        let mut stream = TcpStream::connect(address).map_err(|error| {
            DatasetError::Tokenizer(format!(
                "server tokenizer could not connect to {}:{}: {error}",
                self.origin.host, self.origin.port
            ))
        })?;
        stream.set_read_timeout(Some(self.timeout))?;
        stream.set_write_timeout(Some(self.timeout))?;
        stream.set_nodelay(true).ok();

        let request = format!(
            "POST {path} HTTP/1.1\r\n\
             Host: {host}:{port}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {len}\r\n\
             Connection: close\r\n\
             Accept: application/json\r\n\
             \r\n",
            host = self.origin.host,
            port = self.origin.port,
            len = payload.len(),
        );
        stream.write_all(request.as_bytes())?;
        stream.write_all(payload)?;
        stream.flush()?;

        let mut response = Vec::new();
        stream.read_to_end(&mut response)?;
        parse_http_response(&response)
    }
}

impl TextTokenizer for ServerTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let reply: EncodeReply = self.post_json(
            &self.tokenize_path,
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
            &self.detokenize_path,
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

/// Split a raw HTTP/1.1 response into status validation and body bytes.
///
/// Handles both `Content-Length`-framed and `Transfer-Encoding: chunked`
/// bodies; a `Connection: close` request means the body otherwise runs to EOF.
/// A non-2xx status is surfaced with a bounded snippet of the body for context.
fn parse_http_response(response: &[u8]) -> Result<Vec<u8>> {
    let separator = b"\r\n\r\n";
    let header_end = response
        .windows(separator.len())
        .position(|window| window == separator)
        .ok_or_else(|| {
            DatasetError::Tokenizer(
                "server tokenizer response had no header terminator".to_string(),
            )
        })?;
    let head = &response[..header_end];
    let body = &response[header_end + separator.len()..];

    let head_text = String::from_utf8_lossy(head);
    let mut lines = head_text.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| {
            DatasetError::Tokenizer(format!(
                "server tokenizer returned an unparsable status line {status_line:?}"
            ))
        })?;

    let chunked = lines.any(|line| {
        let lower = line.to_ascii_lowercase();
        lower.starts_with("transfer-encoding:") && lower.contains("chunked")
    });
    let decoded = if chunked {
        dechunk_body(body)?
    } else {
        body.to_vec()
    };

    if !(200..300).contains(&status) {
        let snippet = String::from_utf8_lossy(&decoded);
        let snippet = snippet.chars().take(200).collect::<String>();
        return Err(DatasetError::Tokenizer(format!(
            "server tokenizer returned HTTP {status}: {snippet}"
        )));
    }
    Ok(decoded)
}

/// Decode a `Transfer-Encoding: chunked` body into its contiguous payload.
fn dechunk_body(mut body: &[u8]) -> Result<Vec<u8>> {
    let mut decoded = Vec::with_capacity(body.len());
    loop {
        let line_end = body
            .windows(2)
            .position(|window| window == b"\r\n")
            .ok_or_else(|| {
                DatasetError::Tokenizer("truncated chunk size in server response".to_string())
            })?;
        let size_text = String::from_utf8_lossy(&body[..line_end]);
        let size = usize::from_str_radix(size_text.trim(), 16).map_err(|_| {
            DatasetError::Tokenizer(format!(
                "invalid chunk size {size_text:?} in server response"
            ))
        })?;
        body = &body[line_end + 2..];
        if size == 0 {
            break;
        }
        if body.len() < size {
            return Err(DatasetError::Tokenizer(
                "truncated chunk payload in server response".to_string(),
            ));
        }
        decoded.extend_from_slice(&body[..size]);
        // Skip the payload and its trailing CRLF.
        body = &body[(size + 2).min(body.len())..];
    }
    Ok(decoded)
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
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::thread;

    use super::*;

    /// Deterministic in-process `/tokenize` + `/detokenize` server for the
    /// [`ServerTokenizer`] round-trip test.
    ///
    /// Maps each Unicode scalar to its code point as a token id and back, so a
    /// round-trip is byte-exact and token counts come verifiably from the
    /// server, not a local encoding. Serves `connections` requests then exits.
    /// Binds `127.0.0.1` explicitly (never `localhost`) so the client's direct
    /// connection cannot resolve to an IPv6 loopback the listener does not own.
    fn spawn_tokenize_server(connections: usize) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let base_url = format!("http://127.0.0.1:{}", address.port());
        let handle = thread::spawn(move || {
            for _ in 0..connections {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(&mut stream);

                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let path = request_line.split_whitespace().nth(1).unwrap().to_string();

                let mut content_length = 0usize;
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    if header == "\r\n" || header.is_empty() {
                        break;
                    }
                    if let Some(value) = header.to_ascii_lowercase().strip_prefix("content-length:")
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
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    payload.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.write_all(&payload).unwrap();
                stream.flush().unwrap();
            }
        });
        (base_url, handle)
    }

    // End-to-end: the server tokenizer must round-trip text through the live
    // `/tokenize` and `/detokenize` endpoints, and its token count must be the
    // one the server reports (never a local encoding).
    #[test]
    fn server_tokenizer_round_trips_over_http() {
        let (base_url, handle) = spawn_tokenize_server(3);
        let tokenizer = ServerTokenizer::new(&base_url, Some("test-model".to_string())).unwrap();

        let text = "abc€";
        let tokens = tokenizer.encode(text).unwrap();
        // 'a','b','c' are one scalar each; '€' is U+20AC.
        assert_eq!(tokens, vec![97, 98, 99, 0x20AC]);
        assert_eq!(tokenizer.count(text).unwrap(), 4);
        assert_eq!(tokenizer.decode(&tokens).unwrap(), text);
        assert_eq!(tokenizer.name(), format!("server:{base_url}"));

        handle.join().unwrap();
    }

    // A non-http origin is rejected at construction with a clear message.
    #[test]
    fn server_tokenizer_rejects_non_http_scheme() {
        let error = match ServerTokenizer::new("https://host:8443", None) {
            Ok(_) => panic!("https origin must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("only http origins"));
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
