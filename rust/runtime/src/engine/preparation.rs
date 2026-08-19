// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local-only preparation boundaries shared by native graph inspection tools.

use std::error::Error;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use crate::dataset::{
    HuggingFaceTokenizer, NativeTiktokenTokenizer, TextTokenizer, TiktokenEncoding,
    TiktokenTokenizer, find_tiktoken_model_file,
};

/// Failure to resolve a tokenizer without network access.
#[derive(Debug)]
pub enum LocalTokenizerError {
    /// The supplied name is neither a built-in encoding nor a local tokenizer path.
    Unsupported {
        /// Authored tokenizer name that cannot be resolved locally.
        spec: String,
    },
    /// A local tokenizer path existed but could not be loaded.
    Load {
        /// Authored local tokenizer path.
        spec: String,
        /// Underlying local parse, read, or decode error.
        source: anyhow::Error,
    },
}

impl fmt::Display for LocalTokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { spec } => write!(formatter, "unsupported local tokenizer {spec:?}"),
            Self::Load { spec, source } => {
                write!(
                    formatter,
                    "failed to load local tokenizer {spec:?}: {source}"
                )
            }
        }
    }
}

impl Error for LocalTokenizerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Unsupported { .. } => None,
            Self::Load { source, .. } => Some(source.as_ref()),
        }
    }
}

/// Load a tokenizer from a built-in encoding or already-local tokenizer files.
///
/// This resolver performs no network access. Directories resolve `tokenizer.json`
/// first, then native tiktoken model files, then the Hugging Face directory
/// loader. Existing paths that fail to parse remain distinct from unsupported
/// names so callers can surface an actionable local-file error.
pub fn load_local_tokenizer(
    spec: Option<&str>,
) -> Result<Arc<dyn TextTokenizer>, LocalTokenizerError> {
    let spec = spec.unwrap_or("builtin");
    let path = Path::new(spec);
    if path.is_dir() {
        if path.join("tokenizer.json").is_file() {
            return HuggingFaceTokenizer::from_directory(path)
                .map(|tokenizer| Arc::new(tokenizer) as Arc<dyn TextTokenizer>)
                .map_err(|source| local_load_error(spec, source));
        }
        if find_tiktoken_model_file(path).is_some() {
            return NativeTiktokenTokenizer::from_directory(path)
                .map(|tokenizer| Arc::new(tokenizer) as Arc<dyn TextTokenizer>)
                .map_err(|source| local_load_error(spec, source));
        }
        return HuggingFaceTokenizer::from_directory(path)
            .map(|tokenizer| Arc::new(tokenizer) as Arc<dyn TextTokenizer>)
            .map_err(|source| local_load_error(spec, source));
    }
    if path.is_file() {
        return HuggingFaceTokenizer::from_file(path)
            .map(|tokenizer| Arc::new(tokenizer) as Arc<dyn TextTokenizer>)
            .map_err(|source| local_load_error(spec, source));
    }
    if path.exists() {
        return Err(local_load_error(
            spec,
            anyhow::anyhow!("tokenizer path is neither a file nor a directory"),
        ));
    }
    spec.parse::<TiktokenEncoding>()
        .map(|encoding| Arc::new(TiktokenTokenizer::new(encoding)) as Arc<dyn TextTokenizer>)
        .map_err(|_| LocalTokenizerError::Unsupported {
            spec: spec.to_owned(),
        })
}

fn local_load_error(spec: &str, source: impl Into<anyhow::Error>) -> LocalTokenizerError {
    LocalTokenizerError::Load {
        spec: spec.to_owned(),
        source: source.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{LocalTokenizerError, load_local_tokenizer};

    #[test]
    fn local_tokenizer_none_and_builtin_select_the_builtin_encoding() {
        let none = load_local_tokenizer(None).unwrap();
        let builtin = load_local_tokenizer(Some("builtin")).unwrap();
        assert_eq!(
            none.count("hello").unwrap(),
            builtin.count("hello").unwrap()
        );
    }

    #[test]
    fn local_tokenizer_recognizes_builtin_encoding() {
        let tokenizer = load_local_tokenizer(Some("cl100k_base")).unwrap();
        assert_eq!(tokenizer.name(), "cl100k_base");
    }

    #[test]
    fn local_tokenizer_classifies_existing_malformed_file_as_load_error() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "not a tokenizer").unwrap();

        let error = match load_local_tokenizer(Some(file.path().to_str().unwrap())) {
            Err(error) => error,
            Ok(_) => panic!("malformed tokenizer file must fail to load"),
        };
        assert!(matches!(error, LocalTokenizerError::Load { .. }));
    }

    #[test]
    fn local_tokenizer_classifies_absent_network_name_as_unsupported() {
        let error = match load_local_tokenizer(Some("acme/missing-tokenizer")) {
            Err(error) => error,
            Ok(_) => panic!("absent tokenizer name must not trigger network loading"),
        };
        assert!(matches!(error, LocalTokenizerError::Unsupported { .. }));
    }

    #[test]
    fn local_tokenizer_loads_tiktoken_model_dir_natively() {
        use base64::Engine as _;

        let dir = tempfile::tempdir().unwrap();
        let engine = base64::engine::general_purpose::STANDARD;
        let mut model = String::new();
        for byte in 0u8..=255 {
            model.push_str(&format!("{} {byte}\n", engine.encode([byte])));
        }
        std::fs::write(dir.path().join("tiktoken.model"), model).unwrap();

        let tokenizer = load_local_tokenizer(Some(dir.path().to_str().unwrap())).unwrap();
        let text = "hello world";
        assert_eq!(
            tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap(),
            text
        );
        assert_eq!(tokenizer.count("hi").unwrap(), 2);
    }
}
