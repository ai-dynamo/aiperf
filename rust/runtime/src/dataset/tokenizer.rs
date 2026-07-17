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

use llm_tokenizer::chat_template::ChatTemplateParams;
use llm_tokenizer::traits::{
    Decoder as LlmDecoder, Encoder as LlmEncoder, Tokenizer as LlmTokenizer,
};
use serde_json::Value;
use tiktoken_rs::{
    CoreBPE, cl100k_base_singleton, o200k_base_singleton, o200k_harmony_singleton,
    p50k_base_singleton, p50k_edit_singleton, r50k_base_singleton,
};

use crate::dataset::error::{DatasetError, Result};

/// Tokenization contract used by composers and input accounting.
pub trait TextTokenizer: Send + Sync {
    /// Encode text without automatically adding model special tokens.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode token IDs while retaining explicit special tokens.
    fn decode(&self, token_ids: &[u32]) -> Result<String>;

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
    /// `None` means that chat-template accounting is unavailable. Callers must
    /// then use the ordinary bare-text path, matching Python AIPerf's
    /// best-effort `apply_chat_template` policy.
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
    /// Legacy edit-model encoding.
    P50kEdit,
    /// GPT-2 / legacy GPT-3 encoding.
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
pub struct HuggingFaceTokenizer {
    tokenizer: llm_tokenizer::HuggingFaceTokenizer,
    name: String,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

impl HuggingFaceTokenizer {
    /// Load a standalone `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let path_text = path.to_str().ok_or_else(|| {
            DatasetError::Tokenizer(format!(
                "tokenizer path is not valid UTF-8: {}",
                path.display()
            ))
        })?;
        let tokenizer = llm_tokenizer::HuggingFaceTokenizer::from_file(path_text)
            .map_err(|error| DatasetError::Tokenizer(error.to_string()))?;
        let special_tokens = LlmTokenizer::get_special_tokens(&tokenizer);
        let bos_token_id = special_tokens
            .bos_token
            .as_deref()
            .and_then(|token| LlmTokenizer::token_to_id(&tokenizer, token));
        let eos_token_id = special_tokens
            .eos_token
            .as_deref()
            .and_then(|token| LlmTokenizer::token_to_id(&tokenizer, token));
        Ok(Self {
            tokenizer,
            name: path.display().to_string(),
            bos_token_id,
            eos_token_id,
        })
    }

    /// Load `tokenizer.json` plus BOS/EOS declarations from a model directory.
    pub fn from_directory(path: impl AsRef<Path>) -> Result<Self> {
        let directory = path.as_ref();
        let tokenizer_path = directory.join("tokenizer.json");
        let mut tokenizer = Self::from_file(&tokenizer_path)?;
        tokenizer.name = directory.display().to_string();
        let config_path = directory.join("tokenizer_config.json");
        if config_path.is_file() {
            let config: Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
            tokenizer.bos_token_id = special_token_id(&tokenizer.tokenizer, &config, "bos_token")
                .or(tokenizer.bos_token_id);
            tokenizer.eos_token_id = special_token_id(&tokenizer.tokenizer, &config, "eos_token")
                .or(tokenizer.eos_token_id);
        }
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

/// Download `repository`'s tokenizer files into the standard Hugging Face cache
/// and return the snapshot directory for [`HuggingFaceTokenizer::from_directory`].
///
/// Delegates to the `hf-hub` crate (through `llm_tokenizer::hub`) rather than
/// AIPerf's minimal [`crate::dataset::HttpDatasetFetcher`] so that the xet CDN
/// `302` redirect is followed, the shared on-disk `~/.cache/huggingface` cache is
/// reused across runs and processes, and `HF_HUB_OFFLINE=1` serves an already
/// cached tokenizer with no network access. `hf-hub` resolves the `main`
/// revision; pinned-commit acquisition stays on the HTTP fetcher seam. This is
/// intentionally a free function so a distribution can wrap or replace it while
/// the tokenizer type stays download-mechanism agnostic.
pub async fn download_hugging_face_tokenizer(repository: &str) -> Result<PathBuf> {
    llm_tokenizer::hub::download_tokenizer_from_hf(repository)
        .await
        .map_err(|error| {
            DatasetError::Tokenizer(format!(
                "downloading Hugging Face tokenizer {repository:?}: {error}"
            ))
        })
}

impl TextTokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        LlmEncoder::encode(&self.tokenizer, text, false)
            .map(|encoding| encoding.token_ids().to_vec())
            .map_err(|error| DatasetError::Tokenizer(error.to_string()))
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        LlmDecoder::decode(&self.tokenizer, token_ids, false)
            .map_err(|error| DatasetError::Tokenizer(error.to_string()))
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn vocab_size(&self) -> Option<u32> {
        u32::try_from(LlmTokenizer::vocab_size(&self.tokenizer)).ok()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn apply_chat_template(
        &self,
        messages: &[Value],
        add_generation_prompt: bool,
    ) -> Result<Option<Vec<u32>>> {
        let rendered = match LlmTokenizer::apply_chat_template(
            &self.tokenizer,
            messages,
            ChatTemplateParams {
                add_generation_prompt,
                ..ChatTemplateParams::default()
            },
        ) {
            Ok(rendered) => rendered,
            Err(_) => return Ok(None),
        };
        match self.encode(&rendered) {
            Ok(tokens) => Ok(Some(tokens)),
            Err(_) => Ok(None),
        }
    }
}

fn special_token_id(
    tokenizer: &llm_tokenizer::HuggingFaceTokenizer,
    config: &Value,
    field: &str,
) -> Option<u32> {
    let value = config.get(field)?;
    let token = match value {
        Value::String(token) => token.as_str(),
        Value::Object(object) => object.get("content")?.as_str()?,
        _ => return None,
    };
    LlmTokenizer::token_to_id(tokenizer, token)
}

/// Test tokenizer that encodes to a fixed token run and refuses to decode.
///
/// Shared by the raw-token composition/generation tests (in `dataset::prompt`
/// and `dataset::loader::synthetic`) that must never fall back through a decode
/// path — both wanted the identical fixture, so it lives here once.
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
}
