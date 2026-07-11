// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic multimodal generation seams and native implementations.
//!
//! The behavior is grounded in the complete Python generators at
//! `src/aiperf/dataset/generator/base.py:12-55`,
//! `audio.py:21-193`, `image.py:18-180`, and `video.py:20-507`.
//! Generators return endpoint-ready wire values and never retain payload bytes;
//! the composer interns each result immediately into the shared segment pool.

mod audio;
mod image;
mod video;

use std::path::PathBuf;

use aiperf_rng::{RngRoot, SamplingDistribution};
use bytes::Bytes;

use crate::error::Result;
use crate::model::MediaKind;

pub use audio::{NativeAudioGenerator, audio_duration_seconds, transcode_audio_to_wav};
pub use image::NativeImageGenerator;
pub use video::NativeVideoGenerator;

/// One generated endpoint-ready media value and optional duration metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedMedia {
    /// Content category.
    pub kind: MediaKind,
    /// Endpoint-ready data URI or `format,base64` audio value.
    pub wire: Bytes,
    /// Duration in seconds for audio-bearing values.
    pub duration_seconds: Option<f64>,
}

/// Stateful generator for one configured media category.
pub trait SyntheticMediaGenerator {
    /// Generate one endpoint-ready value.
    fn generate(&mut self) -> Result<GeneratedMedia>;
}

/// Injectable factory for all native synthetic media categories.
pub trait SyntheticMediaGeneratorFactory: Send + Sync {
    /// Construct an image generator.
    fn image(
        &self,
        config: &SyntheticImageConfig,
        root: RngRoot,
    ) -> Result<Box<dyn SyntheticMediaGenerator>>;

    /// Construct an audio generator.
    fn audio(
        &self,
        config: &SyntheticAudioConfig,
        root: RngRoot,
    ) -> Result<Box<dyn SyntheticMediaGenerator>>;

    /// Construct a video generator.
    fn video(
        &self,
        config: &SyntheticVideoConfig,
        root: RngRoot,
    ) -> Result<Box<dyn SyntheticMediaGenerator>>;
}

/// Rust-native image/audio/video generator factory.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeSyntheticMediaGeneratorFactory;

impl SyntheticMediaGeneratorFactory for NativeSyntheticMediaGeneratorFactory {
    fn image(
        &self,
        config: &SyntheticImageConfig,
        root: RngRoot,
    ) -> Result<Box<dyn SyntheticMediaGenerator>> {
        Ok(Box::new(NativeImageGenerator::new(config.clone(), root)?))
    }

    fn audio(
        &self,
        config: &SyntheticAudioConfig,
        root: RngRoot,
    ) -> Result<Box<dyn SyntheticMediaGenerator>> {
        Ok(Box::new(NativeAudioGenerator::new(config.clone(), root)?))
    }

    fn video(
        &self,
        config: &SyntheticVideoConfig,
        root: RngRoot,
    ) -> Result<Box<dyn SyntheticMediaGenerator>> {
        Ok(Box::new(NativeVideoGenerator::new(config.clone(), root)?))
    }
}

/// Image output encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntheticImageFormat {
    /// PNG.
    Png,
    /// JPEG.
    Jpeg,
    /// Uniformly choose PNG or JPEG for each generated image.
    Random,
}

/// Image pixel source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyntheticImageSource {
    /// Fresh random RGB noise for every image.
    Noise,
    /// One of AIPerf's bundled natural source images.
    BundledAssets,
    /// A supported image selected lazily from this directory.
    Directory(PathBuf),
}

/// Selection policy for finite source-image pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceImageSampling {
    /// Independent random selection with replacement.
    RandomWithReplacement,
    /// Visit every readable image in a shuffled cycle.
    ShuffleCycle,
    /// Visit readable images in sorted path order and wrap.
    SequentialCycle,
}

/// Synthetic image generation configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticImageConfig {
    /// Images per turn.
    pub batch_size: usize,
    /// Width distribution in pixels.
    pub width: SamplingDistribution,
    /// Height distribution in pixels.
    pub height: SamplingDistribution,
    /// Output encoding.
    pub format: SyntheticImageFormat,
    /// Pixel source.
    pub source: SyntheticImageSource,
    /// Finite-source selection policy.
    pub source_sampling: SourceImageSampling,
}

impl Default for SyntheticImageConfig {
    fn default() -> Self {
        Self {
            batch_size: 0,
            width: fixed(512.0),
            height: fixed(512.0),
            format: SyntheticImageFormat::Jpeg,
            source: SyntheticImageSource::Noise,
            source_sampling: SourceImageSampling::RandomWithReplacement,
        }
    }
}

/// Audio container/codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntheticAudioFormat {
    /// PCM WAV.
    Wav,
    /// MP3 encoded through FFmpeg.
    Mp3,
}

/// Synthetic audio generation configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticAudioConfig {
    /// Audio clips per turn.
    pub batch_size: usize,
    /// Duration distribution in seconds.
    pub duration_seconds: SamplingDistribution,
    /// Output format.
    pub format: SyntheticAudioFormat,
    /// Candidate sample rates in hertz.
    pub sample_rates_hz: Vec<u32>,
    /// Candidate PCM depths: 8, 16, 24, or 32.
    pub bit_depths: Vec<u16>,
    /// One or two channels.
    pub channels: u16,
}

impl Default for SyntheticAudioConfig {
    fn default() -> Self {
        Self {
            batch_size: 0,
            duration_seconds: fixed(10.0),
            format: SyntheticAudioFormat::Wav,
            sample_rates_hz: vec![16_000],
            bit_depths: vec![16],
            channels: 1,
        }
    }
}

/// Video container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntheticVideoFormat {
    /// MP4 container.
    Mp4,
    /// WebM container.
    WebM,
}

impl SyntheticVideoFormat {
    pub(crate) const fn extension(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::WebM => "webm",
        }
    }
}

/// Deterministic frame synthesis algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntheticVideoPattern {
    /// Three moving geometric shapes.
    MovingShapes,
    /// Grid with frame-driven clock hands.
    GridClock,
    /// Fresh random RGB noise for each frame.
    Noise,
}

/// Optional synthetic audio track embedded in a video.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticVideoAudioConfig {
    /// Zero disables audio; otherwise one or two channels.
    pub channels: u16,
    /// Sample rate in hertz.
    pub sample_rate_hz: u32,
    /// PCM source depth: 8, 16, 24, or 32.
    pub bit_depth: u16,
    /// Optional FFmpeg audio codec; absent selects AAC for MP4 and Vorbis for WebM.
    pub codec: Option<String>,
}

impl Default for SyntheticVideoAudioConfig {
    fn default() -> Self {
        Self {
            channels: 0,
            sample_rate_hz: 44_100,
            bit_depth: 16,
            codec: None,
        }
    }
}

/// Synthetic video generation configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticVideoConfig {
    /// Videos per turn.
    pub batch_size: usize,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Clip duration in seconds.
    pub duration_seconds: f64,
    /// Frames per second.
    pub frames_per_second: u32,
    /// Container format.
    pub format: SyntheticVideoFormat,
    /// FFmpeg video codec.
    pub codec: String,
    /// Frame synthesis algorithm.
    pub pattern: SyntheticVideoPattern,
    /// Optional generated audio track.
    pub audio: SyntheticVideoAudioConfig,
}

impl Default for SyntheticVideoConfig {
    fn default() -> Self {
        Self {
            batch_size: 0,
            width: 640,
            height: 480,
            duration_seconds: 1.0,
            frames_per_second: 4,
            format: SyntheticVideoFormat::WebM,
            codec: "libvpx-vp9".into(),
            pattern: SyntheticVideoPattern::MovingShapes,
            audio: SyntheticVideoAudioConfig::default(),
        }
    }
}

/// Synthetic text prompt configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticPromptConfig {
    /// Input length sampled independently when no paired sequence distribution is configured.
    pub input_tokens: SamplingDistribution,
    /// Number of independently generated text values in each turn.
    pub batch_size: usize,
}

impl Default for SyntheticPromptConfig {
    fn default() -> Self {
        Self {
            input_tokens: fixed(128.0),
            batch_size: 1,
        }
    }
}

/// Synthetic KV-prefix and conversation-context prompt configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SyntheticPrefixConfig {
    /// Number of reusable prefixes sampled per turn.
    pub pool_size: Option<usize>,
    /// Token length of each reusable prefix.
    pub prefix_tokens: Option<usize>,
    /// Token length of the one shared system prompt.
    pub shared_system_tokens: Option<usize>,
    /// Token length of the unique per-conversation context prompt.
    pub user_context_tokens: Option<usize>,
}

/// Complete synthetic conversation dataset shape.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticDatasetConfig {
    /// Number of reusable conversations to create.
    pub entries: usize,
    /// Number of turns per conversation.
    pub turns: SamplingDistribution,
    /// Inter-turn delay in milliseconds.
    pub turn_delay_ms: SamplingDistribution,
    /// Multiplicative delay scale.
    pub turn_delay_ratio: f64,
    /// Optional text generation.
    pub prompts: Option<SyntheticPromptConfig>,
    /// Optional prefix/context generation.
    pub prefixes: SyntheticPrefixConfig,
    /// Optional image generation.
    pub images: Option<SyntheticImageConfig>,
    /// Optional audio generation.
    pub audio: Option<SyntheticAudioConfig>,
    /// Optional video generation.
    pub video: Option<SyntheticVideoConfig>,
    /// Optional rankings-specific shape used by `synthetic_rankings`.
    pub rankings: Option<SyntheticRankingsConfig>,
}

impl Default for SyntheticDatasetConfig {
    fn default() -> Self {
        Self {
            entries: 100,
            turns: fixed(1.0),
            turn_delay_ms: fixed(0.0),
            turn_delay_ratio: 1.0,
            prompts: Some(SyntheticPromptConfig::default()),
            prefixes: SyntheticPrefixConfig::default(),
            images: None,
            audio: None,
            video: None,
            rankings: None,
        }
    }
}

/// Synthetic query/passage rankings shape.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticRankingsConfig {
    /// Passage count per request.
    pub passages: SamplingDistribution,
    /// Tokens per passage.
    pub passage_tokens: SamplingDistribution,
    /// Query tokens.
    pub query_tokens: SamplingDistribution,
}

impl Default for SyntheticRankingsConfig {
    fn default() -> Self {
        Self {
            passages: fixed(10.0),
            passage_tokens: fixed(128.0),
            query_tokens: fixed(32.0),
        }
    }
}

fn fixed(value: f64) -> SamplingDistribution {
    SamplingDistribution::fixed(value).expect("positive fixed defaults are valid")
}
