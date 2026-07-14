// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multimodal content resolution for loaders.
//!
//! URLs and already-encoded values pass through byte-identically. Local image,
//! audio, and video files are validated and encoded once at composition time,
//! before the resulting bytes enter the content-addressed store.

use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;

use crate::dataset::error::{DatasetError, Result};
use crate::dataset::model::MediaKind;

/// Pluggable conversion from an authored media reference to endpoint-ready bytes.
pub trait MediaResolver: Send + Sync {
    /// Resolve one URL, encoded value, or local path.
    fn resolve(&self, kind: MediaKind, authored: &str) -> Result<Bytes>;
}

/// Default resolver that inlines local files and preserves URLs/encoded values.
#[derive(Debug, Clone, Copy, Default)]
pub struct InlineMediaResolver;

impl MediaResolver for InlineMediaResolver {
    fn resolve(&self, kind: MediaKind, authored: &str) -> Result<Bytes> {
        if kind == MediaKind::Text || is_already_encoded(kind, authored) {
            return Ok(Bytes::copy_from_slice(authored.as_bytes()));
        }
        if authored.contains("://") {
            let url = url::Url::parse(authored).map_err(|error| {
                DatasetError::Validation(format!("invalid media URL {authored:?}: {error}"))
            })?;
            if url.scheme().is_empty() || url.host_str().is_none() {
                return Err(DatasetError::Validation(format!(
                    "media URL must have a scheme and host: {authored:?}"
                )));
            }
            return Ok(Bytes::copy_from_slice(authored.as_bytes()));
        }

        let path = Path::new(authored);
        let raw = std::fs::read(path).map_err(|error| {
            DatasetError::Io(std::io::Error::new(
                error.kind(),
                format!("failed to read media file {}: {error}", path.display()),
            ))
        })?;
        let encoded = match kind {
            MediaKind::Text => authored.to_string(),
            MediaKind::Image => encode_image(path, &raw)?,
            MediaKind::Audio => encode_audio(path, &raw)?,
            MediaKind::Video => encode_video(path, &raw)?,
        };
        Ok(Bytes::from(encoded))
    }
}

fn is_already_encoded(kind: MediaKind, content: &str) -> bool {
    match kind {
        MediaKind::Text => true,
        MediaKind::Image | MediaKind::Video => content.starts_with("data:"),
        MediaKind::Audio => content.split_once(',').is_some_and(|(format, _)| {
            !format.contains(':') && matches!(format.to_ascii_lowercase().as_str(), "wav" | "mp3")
        }),
    }
}

fn encode_image(path: &Path, raw: &[u8]) -> Result<String> {
    let format = image::guess_format(raw).map_err(|error| {
        DatasetError::Validation(format!(
            "failed to determine image format for {}: {error}",
            path.display()
        ))
    })?;
    let mime = match format {
        image::ImageFormat::Png => "png",
        image::ImageFormat::Jpeg => "jpeg",
        other => {
            return Err(DatasetError::Validation(format!(
                "unsupported image format {other:?} for {}; expected PNG or JPEG",
                path.display()
            )));
        }
    };
    image::load_from_memory_with_format(raw, format).map_err(|error| {
        DatasetError::Validation(format!("invalid image {}: {error}", path.display()))
    })?;
    Ok(format!("data:image/{mime};base64,{}", STANDARD.encode(raw)))
}

fn encode_audio(path: &Path, raw: &[u8]) -> Result<String> {
    let format = match extension(path) {
        "wav" => "wav",
        "mp3" => "mp3",
        other => {
            return Err(DatasetError::Validation(format!(
                "unsupported audio extension {other:?} for {}; expected wav or mp3",
                path.display()
            )));
        }
    };
    Ok(format!("{format},{}", STANDARD.encode(raw)))
}

fn encode_video(path: &Path, raw: &[u8]) -> Result<String> {
    let format = match extension(path) {
        "mp4" => "mp4",
        "webm" => "webm",
        other => {
            return Err(DatasetError::Validation(format!(
                "unsupported video extension {other:?} for {}; expected mp4 or webm",
                path.display()
            )));
        }
    };
    Ok(format!(
        "data:video/{format};base64,{}",
        STANDARD.encode(raw)
    ))
}

fn extension(path: &Path) -> &str {
    path.extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
}

#[cfg(test)]
mod tests {
    use image::{ImageBuffer, Rgb};

    use super::*;

    #[test]
    fn urls_and_encoded_content_pass_through() {
        let resolver = InlineMediaResolver;
        for (kind, value) in [
            (MediaKind::Image, "https://example.com/a.png"),
            (MediaKind::Image, "data:image/png;base64,AA=="),
            (MediaKind::Audio, "wav,AA=="),
            (MediaKind::Video, "data:video/mp4;base64,AA=="),
        ] {
            assert_eq!(resolver.resolve(kind, value).unwrap(), value.as_bytes());
        }
    }

    #[test]
    fn local_png_is_validated_and_inlined() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pixel.png");
        ImageBuffer::<Rgb<u8>, _>::from_pixel(1, 1, Rgb([1, 2, 3]))
            .save(&path)
            .unwrap();
        let encoded = InlineMediaResolver
            .resolve(MediaKind::Image, path.to_str().unwrap())
            .unwrap();
        assert!(encoded.starts_with(b"data:image/png;base64,"));
        let data = encoded.split(|byte| *byte == b',').nth(1).unwrap();
        let decoded = STANDARD.decode(data).unwrap();
        assert_eq!(
            image::guess_format(&decoded).unwrap(),
            image::ImageFormat::Png
        );
    }

    #[test]
    fn windows_style_paths_are_not_misclassified_as_urls() {
        let error = InlineMediaResolver
            .resolve(MediaKind::Image, r"C:\Users\missing.png")
            .unwrap_err();
        assert!(matches!(error, DatasetError::Io(_)));
    }
}
