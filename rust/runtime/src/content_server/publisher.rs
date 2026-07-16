// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic image/video persistence with endpoint-ready URL results.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::dataset::{
    InlineSyntheticMediaPublisher, MediaKind, SyntheticMediaFormat, SyntheticMediaPublisher,
};
use bytes::Bytes;
use url::Url;

use crate::content_server::{ContentServerError, Result};

/// Publisher that writes synthetic images/videos under a served directory.
#[derive(Debug)]
pub struct ContentServerMediaPublisher {
    content_dir: PathBuf,
    base_url: String,
    image_counter: AtomicU64,
    video_counter: AtomicU64,
    inline: InlineSyntheticMediaPublisher,
}

impl ContentServerMediaPublisher {
    /// Validate an existing content root and advertised HTTP(S) base URL.
    pub fn new(content_dir: impl AsRef<Path>, base_url: impl Into<String>) -> Result<Self> {
        let content_dir = content_dir.as_ref().canonicalize().map_err(|source| {
            ContentServerError::io(
                format!(
                    "canonicalizing content-server directory {}",
                    content_dir.as_ref().display()
                ),
                source,
            )
        })?;
        if !content_dir.is_dir() {
            return Err(ContentServerError::invalid(format!(
                "content-server path {} is not a directory",
                content_dir.display()
            )));
        }
        let base_url = base_url.into().trim_end_matches('/').to_owned();
        let parsed = Url::parse(&base_url)
            .map_err(|source| ContentServerError::url(base_url.clone(), source))?;
        if !matches!(parsed.scheme(), "http" | "https")
            || parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
        {
            return Err(ContentServerError::invalid(
                "content-server base URL must use http:// or https:// and include a host",
            ));
        }
        if parsed.path() != "/" || parsed.query().is_some() || parsed.fragment().is_some() {
            return Err(ContentServerError::invalid(
                "content-server base URL must not contain a path, query, or fragment",
            ));
        }
        Ok(Self {
            content_dir,
            base_url,
            image_counter: AtomicU64::new(0),
            video_counter: AtomicU64::new(0),
            inline: InlineSyntheticMediaPublisher,
        })
    }

    /// Canonical serving root shared with the HTTP server.
    pub fn content_dir(&self) -> &Path {
        &self.content_dir
    }

    fn next_file(&self, format: SyntheticMediaFormat) -> Result<(PathBuf, String)> {
        let (subdir, prefix, counter) = match format.kind() {
            MediaKind::Image => ("images", "img", &self.image_counter),
            MediaKind::Video => ("video", "vid", &self.video_counter),
            MediaKind::Audio => {
                return Err(ContentServerError::invalid(
                    "audio is always published inline",
                ));
            }
            MediaKind::Text => unreachable!("synthetic media formats never represent text"),
        };
        let index = counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map(|previous| previous + 1)
            .map_err(|_| {
                ContentServerError::invalid("content-server media file counter exhausted")
            })?;
        let filename = format!("{prefix}_{index:06}.{}", format.extension());
        let directory = self.content_dir.join(subdir);
        std::fs::create_dir_all(&directory).map_err(|source| {
            ContentServerError::io(
                format!(
                    "creating content-server media directory {}",
                    directory.display()
                ),
                source,
            )
        })?;
        let canonical_directory = directory.canonicalize().map_err(|source| {
            ContentServerError::io(
                format!(
                    "canonicalizing content-server media directory {}",
                    directory.display()
                ),
                source,
            )
        })?;
        if !canonical_directory.starts_with(&self.content_dir) {
            return Err(ContentServerError::invalid(format!(
                "content-server media directory {} escapes serving root {}",
                canonical_directory.display(),
                self.content_dir.display()
            )));
        }
        let path = canonical_directory.join(&filename);
        let url = format!("{}/content/{subdir}/{filename}", self.base_url);
        Ok((path, url))
    }
}

impl SyntheticMediaPublisher for ContentServerMediaPublisher {
    fn publish(
        &self,
        format: SyntheticMediaFormat,
        encoded: Bytes,
    ) -> crate::dataset::Result<Bytes> {
        if format.kind() == MediaKind::Audio {
            return self.inline.publish(format, encoded);
        }
        let (path, url) = self.next_file(format).map_err(dataset_error)?;
        persist_media(&path, &encoded)?;
        Ok(Bytes::from(url))
    }
}

fn persist_media(path: &Path, encoded: &[u8]) -> crate::dataset::Result<()> {
    let directory = path
        .parent()
        .expect("generated content-server media paths always have a parent");
    let mut temporary = tempfile::NamedTempFile::new_in(directory).map_err(|error| {
        crate::dataset::DatasetError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "failed to create temporary media file in {}: {error}",
                directory.display()
            ),
        ))
    })?;
    temporary.write_all(encoded).map_err(|error| {
        crate::dataset::DatasetError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "failed to write temporary content-server media for {}: {error}",
                path.display()
            ),
        ))
    })?;
    temporary.flush().map_err(|error| {
        crate::dataset::DatasetError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "failed to flush temporary content-server media for {}: {error}",
                path.display()
            ),
        ))
    })?;
    temporary.persist(path).map_err(|error| {
        crate::dataset::DatasetError::Io(std::io::Error::new(
            error.error.kind(),
            format!(
                "failed to publish content-server media {}: {}",
                path.display(),
                error.error
            ),
        ))
    })?;
    Ok(())
}

fn dataset_error(error: ContentServerError) -> crate::dataset::DatasetError {
    match error {
        ContentServerError::Io { operation, source } => {
            let kind = source.kind();
            crate::dataset::DatasetError::Io(std::io::Error::new(
                kind,
                format!("{operation}: {source}"),
            ))
        }
        error => crate::dataset::DatasetError::Validation(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;

    use super::*;

    #[test]
    fn images_and_videos_are_numbered_per_modality_while_audio_stays_inline() {
        let directory = tempfile::tempdir().unwrap();
        let publisher =
            ContentServerMediaPublisher::new(directory.path(), "http://host:8090").unwrap();

        let first_image = publisher
            .publish(
                SyntheticMediaFormat::ImagePng,
                Bytes::from_static(b"png-one"),
            )
            .unwrap();
        let second_image = publisher
            .publish(
                SyntheticMediaFormat::ImageJpeg,
                Bytes::from_static(b"jpeg-two"),
            )
            .unwrap();
        let video = publisher
            .publish(
                SyntheticMediaFormat::VideoWebM,
                Bytes::from_static(b"video"),
            )
            .unwrap();
        let audio = publisher
            .publish(SyntheticMediaFormat::AudioWav, Bytes::from_static(b"wav"))
            .unwrap();

        assert_eq!(
            first_image,
            "http://host:8090/content/images/img_000001.png"
        );
        assert_eq!(
            second_image,
            "http://host:8090/content/images/img_000002.jpeg"
        );
        assert_eq!(video, "http://host:8090/content/video/vid_000001.webm");
        assert_eq!(audio, format!("wav,{}", STANDARD.encode(b"wav")));
        assert_eq!(
            std::fs::read(directory.path().join("images/img_000001.png")).unwrap(),
            b"png-one"
        );
        assert_eq!(
            std::fs::read(directory.path().join("video/vid_000001.webm")).unwrap(),
            b"video"
        );
    }

    #[test]
    fn publisher_rejects_non_origin_base_urls() {
        let directory = tempfile::tempdir().unwrap();

        for base_url in [
            "file:///tmp/content",
            "http://user@host:8090",
            "http://host:8090/nested",
            "http://host:8090?query=1",
        ] {
            assert!(ContentServerMediaPublisher::new(directory.path(), base_url).is_err());
        }
    }

    #[test]
    fn exhausted_counter_fails_without_wrapping_or_overwriting() {
        let directory = tempfile::tempdir().unwrap();
        let publisher =
            ContentServerMediaPublisher::new(directory.path(), "http://host:8090").unwrap();
        publisher.image_counter.store(u64::MAX, Ordering::Relaxed);

        let error = publisher
            .publish(
                SyntheticMediaFormat::ImagePng,
                Bytes::from_static(b"never-written"),
            )
            .unwrap_err();

        assert!(error.to_string().contains("counter exhausted"));
        assert_eq!(publisher.image_counter.load(Ordering::Relaxed), u64::MAX);
        assert!(!directory.path().join("images").exists());
    }

    #[cfg(unix)]
    #[test]
    fn publisher_rejects_media_subdirectory_symlink_escapes() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), directory.path().join("images")).unwrap();
        let publisher =
            ContentServerMediaPublisher::new(directory.path(), "http://host:8090").unwrap();

        let error = publisher
            .publish(
                SyntheticMediaFormat::ImagePng,
                Bytes::from_static(b"must-not-escape"),
            )
            .unwrap_err();

        assert!(error.to_string().contains("escapes serving root"));
        assert!(!outside.path().join("img_000001.png").exists());
    }

    #[cfg(unix)]
    #[test]
    fn atomic_publication_replaces_a_final_symlink_without_following_it() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::create_dir(directory.path().join("images")).unwrap();
        let outside_file = outside.path().join("outside.png");
        std::fs::write(&outside_file, b"outside-must-survive").unwrap();
        let published_path = directory.path().join("images/img_000001.png");
        symlink(&outside_file, &published_path).unwrap();
        let publisher =
            ContentServerMediaPublisher::new(directory.path(), "http://host:8090").unwrap();

        publisher
            .publish(
                SyntheticMediaFormat::ImagePng,
                Bytes::from_static(b"published-inside-root"),
            )
            .unwrap();

        assert_eq!(
            std::fs::read(&outside_file).unwrap(),
            b"outside-must-survive"
        );
        assert_eq!(
            std::fs::read(&published_path).unwrap(),
            b"published-inside-root"
        );
        assert!(
            !std::fs::symlink_metadata(published_path)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }
}
