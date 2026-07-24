// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native deterministic image generator.

use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::rng::{ConfiguredRandomGenerator, RandomGenerator, RngRoot, RuntimeRandomGenerator};
use bytes::Bytes;
use image::imageops::FilterType;
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};

use super::{
    GeneratedMedia, InlineSyntheticMediaPublisher, SourceImageSampling, SyntheticImageConfig,
    SyntheticImageFormat, SyntheticImageSource, SyntheticMediaFormat, SyntheticMediaGenerator,
    SyntheticMediaPublisher,
};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::model::MediaKind;

static BUNDLED_IMAGES: &[&[u8]] = &[
    include_bytes!(
        "../../../../../src/aiperf/dataset/generator/assets/source_images/0bfd8fdf-457f-43c8-9253-a2346d37d26a_1024.jpg"
    ),
    include_bytes!(
        "../../../../../src/aiperf/dataset/generator/assets/source_images/119544eb-9bbf-47d1-8d93-a51de6370295_861.jpg"
    ),
    include_bytes!(
        "../../../../../src/aiperf/dataset/generator/assets/source_images/1ba066e0-7291-4ef4-8a6e-e8724af7f046_865.jpg"
    ),
    include_bytes!(
        "../../../../../src/aiperf/dataset/generator/assets/source_images/c946f826-4a9a-4ca2-b891-9f791d38092b_1024.jpg"
    ),
];

#[derive(Debug, Clone)]
enum SourceImage {
    Bundled(usize),
    Path(PathBuf),
}

/// Rust-native synthetic image generator.
pub struct NativeImageGenerator {
    config: SyntheticImageConfig,
    dimensions_rng: ConfiguredRandomGenerator,
    format_rng: ConfiguredRandomGenerator,
    source_rng: ConfiguredRandomGenerator,
    noise_rng: ConfiguredRandomGenerator,
    sources: Vec<SourceImage>,
    available: Vec<usize>,
    shuffle_cycle: Vec<usize>,
    sequential_index: usize,
    publisher: Arc<dyn SyntheticMediaPublisher>,
}

impl NativeImageGenerator {
    /// Validate configuration and index finite image sources without decoding them.
    pub fn new(config: SyntheticImageConfig, root: RngRoot) -> Result<Self> {
        Self::new_with_publisher(config, root, Arc::new(InlineSyntheticMediaPublisher))
    }

    /// Validate configuration and bind an injected final publication policy.
    pub fn new_with_publisher(
        config: SyntheticImageConfig,
        root: RngRoot,
        publisher: Arc<dyn SyntheticMediaPublisher>,
    ) -> Result<Self> {
        if config.batch_size == 0 {
            return Err(DatasetError::Validation(
                "an image generator requires batch_size > 0".into(),
            ));
        }
        if matches!(config.source, SyntheticImageSource::Noise)
            && config.source_sampling != SourceImageSampling::RandomWithReplacement
        {
            return Err(DatasetError::Validation(
                "noise images require random-with-replacement source sampling".into(),
            ));
        }
        let sources = match &config.source {
            SyntheticImageSource::Noise => Vec::new(),
            SyntheticImageSource::BundledAssets => (0..BUNDLED_IMAGES.len())
                .map(SourceImage::Bundled)
                .collect(),
            SyntheticImageSource::Directory(path) => index_directory(path)?,
        };
        let available = (0..sources.len()).collect();
        Ok(Self {
            config,
            dimensions_rng: root.derive_generator("dataset.image.dimensions"),
            format_rng: root.derive_generator("dataset.image.format"),
            source_rng: root.derive_generator("dataset.image.source"),
            noise_rng: root.derive_generator("dataset.image.noise"),
            sources,
            available,
            shuffle_cycle: Vec::new(),
            sequential_index: 0,
            publisher,
        })
    }

    fn dimensions(&mut self) -> Result<(u32, u32)> {
        let width = self
            .config
            .width
            .sample_int(&mut self.dimensions_rng)
            .map_err(|error| DatasetError::Validation(error.to_string()))?;
        let height = self
            .config
            .height
            .sample_int(&mut self.dimensions_rng)
            .map_err(|error| DatasetError::Validation(error.to_string()))?;
        let width = u32::try_from(width).map_err(|_| {
            DatasetError::Validation(format!("sampled image width {width} exceeds u32"))
        })?;
        let height = u32::try_from(height).map_err(|_| {
            DatasetError::Validation(format!("sampled image height {height} exceeds u32"))
        })?;
        width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| DatasetError::Validation("sampled image dimensions overflow".into()))?;
        Ok((width, height))
    }

    fn source_image(&mut self, width: u32, height: u32) -> Result<DynamicImage> {
        if matches!(self.config.source, SyntheticImageSource::Noise) {
            let len = usize::try_from(width)
                .ok()
                .and_then(|width| {
                    usize::try_from(height)
                        .ok()
                        .and_then(|height| width.checked_mul(height))
                })
                .and_then(|pixels| pixels.checked_mul(3))
                .ok_or_else(|| DatasetError::Validation("image allocation overflow".into()))?;
            let mut pixels = vec![0; len];
            self.noise_rng.fill_bytes(&mut pixels);
            let image = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, pixels)
                .ok_or_else(|| DatasetError::Validation("invalid RGB image geometry".into()))?;
            return Ok(DynamicImage::ImageRgb8(image));
        }

        while !self.available.is_empty() {
            let index = self.next_source_index()?;
            let decoded = match &self.sources[index] {
                SourceImage::Bundled(index) => image::load_from_memory(BUNDLED_IMAGES[*index]),
                SourceImage::Path(path) => image::open(path),
            };
            match decoded {
                Ok(image) => {
                    return Ok(image.resize_exact(width, height, FilterType::Lanczos3));
                }
                Err(_) => self.retire(index),
            }
        }
        Err(DatasetError::Validation(
            "no readable source images remain".into(),
        ))
    }

    fn next_source_index(&mut self) -> Result<usize> {
        match self.config.source_sampling {
            SourceImageSampling::RandomWithReplacement => self
                .source_rng
                .choice(&self.available)
                .copied()
                .map_err(|error| DatasetError::Validation(error.to_string())),
            SourceImageSampling::ShuffleCycle => {
                if self.shuffle_cycle.is_empty() {
                    self.shuffle_cycle.clone_from(&self.available);
                    self.source_rng.shuffle(&mut self.shuffle_cycle);
                }
                self.shuffle_cycle.pop().ok_or_else(|| {
                    DatasetError::Validation("no readable source images remain".into())
                })
            }
            SourceImageSampling::SequentialCycle => {
                for _ in 0..self.sources.len() {
                    let index = self.sequential_index;
                    self.sequential_index = (self.sequential_index + 1) % self.sources.len();
                    if self.available.contains(&index) {
                        return Ok(index);
                    }
                }
                Err(DatasetError::Validation(
                    "no readable source images remain".into(),
                ))
            }
        }
    }

    fn retire(&mut self, index: usize) {
        self.available.retain(|candidate| *candidate != index);
        self.shuffle_cycle.retain(|candidate| *candidate != index);
    }

    fn output_format(&mut self) -> Result<ImageFormat> {
        match self.config.format {
            SyntheticImageFormat::Png => Ok(ImageFormat::Png),
            SyntheticImageFormat::Jpeg => Ok(ImageFormat::Jpeg),
            SyntheticImageFormat::Random => self
                .format_rng
                .choice(&[ImageFormat::Png, ImageFormat::Jpeg])
                .copied()
                .map_err(|error| DatasetError::Validation(error.to_string())),
        }
    }
}

impl SyntheticMediaGenerator for NativeImageGenerator {
    fn generate(&mut self) -> Result<GeneratedMedia> {
        let (width, height) = self.dimensions()?;
        let image = self.source_image(width, height)?;
        let format = self.output_format()?;
        let mut encoded = Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, format)
            .map_err(|error| DatasetError::Validation(format!("image encoding failed: {error}")))?;
        let media_format = match format {
            ImageFormat::Png => SyntheticMediaFormat::ImagePng,
            ImageFormat::Jpeg => SyntheticMediaFormat::ImageJpeg,
            _ => unreachable!("output_format returns PNG or JPEG"),
        };
        Ok(GeneratedMedia {
            kind: MediaKind::Image,
            wire: self
                .publisher
                .publish(media_format, Bytes::from(encoded.into_inner()))?,
            duration_seconds: None,
        })
    }
}

fn index_directory(path: &Path) -> Result<Vec<SourceImage>> {
    if !path.exists() {
        return Err(DatasetError::Validation(format!(
            "image source directory {} does not exist",
            path.display()
        )));
    }
    if !path.is_dir() {
        return Err(DatasetError::Validation(format!(
            "image source {} is not a directory",
            path.display()
        )));
    }
    let mut paths = fs::read_dir(path)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| ImageFormat::from_path(path).is_ok())
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(DatasetError::Validation(format!(
            "no supported source images found in {}",
            path.display()
        )));
    }
    Ok(paths.into_iter().map(SourceImage::Path).collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;

    use super::*;

    #[derive(Debug, Default)]
    struct RecordingPublisher {
        published: Mutex<Vec<(SyntheticMediaFormat, Bytes)>>,
    }

    impl SyntheticMediaPublisher for RecordingPublisher {
        fn publish(&self, format: SyntheticMediaFormat, encoded: Bytes) -> Result<Bytes> {
            self.published.lock().unwrap().push((format, encoded));
            Ok(Bytes::from_static(b"http://content/image.png"))
        }
    }

    #[test]
    fn noise_image_has_sampled_geometry_and_valid_data_uri() {
        let config = SyntheticImageConfig {
            batch_size: 1,
            width: crate::rng::SamplingDistribution::fixed(7.0).unwrap(),
            height: crate::rng::SamplingDistribution::fixed(5.0).unwrap(),
            format: SyntheticImageFormat::Png,
            ..SyntheticImageConfig::default()
        };
        let mut generator = NativeImageGenerator::new(config, RngRoot::new(Some(9))).unwrap();
        let generated = generator.generate().unwrap();
        let encoded = generated.wire.split(|byte| *byte == b',').nth(1).unwrap();
        let decoded = STANDARD.decode(encoded).unwrap();
        let image = image::load_from_memory_with_format(&decoded, ImageFormat::Png).unwrap();
        assert_eq!((image.width(), image.height()), (7, 5));
    }

    #[test]
    fn injected_publisher_receives_encoded_image_before_wire_representation() {
        let publisher = Arc::new(RecordingPublisher::default());
        let config = SyntheticImageConfig {
            batch_size: 1,
            width: crate::rng::SamplingDistribution::fixed(3.0).unwrap(),
            height: crate::rng::SamplingDistribution::fixed(2.0).unwrap(),
            format: SyntheticImageFormat::Png,
            ..SyntheticImageConfig::default()
        };
        let mut generator = NativeImageGenerator::new_with_publisher(
            config,
            RngRoot::new(Some(9)),
            publisher.clone(),
        )
        .unwrap();

        let generated = generator.generate().unwrap();

        assert_eq!(generated.wire, "http://content/image.png");
        let published = publisher.published.lock().unwrap();
        assert_eq!(published[0].0, SyntheticMediaFormat::ImagePng);
        let decoded =
            image::load_from_memory_with_format(&published[0].1, ImageFormat::Png).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (3, 2));
    }

    #[test]
    fn bundled_assets_are_lazy_and_cycle() {
        let config = SyntheticImageConfig {
            batch_size: 1,
            width: crate::rng::SamplingDistribution::fixed(8.0).unwrap(),
            height: crate::rng::SamplingDistribution::fixed(6.0).unwrap(),
            format: SyntheticImageFormat::Jpeg,
            source: SyntheticImageSource::BundledAssets,
            source_sampling: SourceImageSampling::SequentialCycle,
        };
        let mut generator = NativeImageGenerator::new(config, RngRoot::new(Some(1))).unwrap();
        for _ in 0..BUNDLED_IMAGES.len() + 1 {
            assert!(
                generator
                    .generate()
                    .unwrap()
                    .wire
                    .starts_with(b"data:image/jpeg;base64,")
            );
        }
    }
}
