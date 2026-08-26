// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic multimodal generation seams and native implementations.
//!
//! Generators return endpoint-ready wire values and never retain payload bytes;
//! the composer interns each result immediately into the shared segment pool.

mod audio;
mod image;
mod video;

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use crate::rng::{RngRoot, SamplingDistribution};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;

use crate::dataset::error::Result;
use crate::dataset::model::MediaKind;

pub use audio::{NativeAudioGenerator, audio_duration_seconds, transcode_audio_to_wav};
pub use image::NativeImageGenerator;
pub use video::NativeVideoGenerator;

/// One generated endpoint-ready media value and optional duration metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedMedia {
    /// Content category.
    pub kind: MediaKind,
    /// Endpoint-ready URL, data URI, or `format,base64` audio value.
    pub wire: Bytes,
    /// Duration in seconds for audio-bearing values.
    pub duration_seconds: Option<f64>,
}

/// Encoded media container handed to a synthetic-media publisher.
///
/// Native generators own pixels, samples, frames, and codec invocation; this
/// value leaves the final delivery representation open. The default publisher
/// produces inline values, while an online content-server
/// publisher can persist image/video bytes and return URLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntheticMediaFormat {
    /// PNG image.
    ImagePng,
    /// JPEG image.
    ImageJpeg,
    /// PCM WAV audio.
    AudioWav,
    /// MP3 audio.
    AudioMp3,
    /// MP4 video.
    VideoMp4,
    /// WebM video.
    VideoWebM,
}

impl SyntheticMediaFormat {
    /// Media category represented by this encoded container.
    pub const fn kind(self) -> MediaKind {
        match self {
            Self::ImagePng | Self::ImageJpeg => MediaKind::Image,
            Self::AudioWav | Self::AudioMp3 => MediaKind::Audio,
            Self::VideoMp4 | Self::VideoWebM => MediaKind::Video,
        }
    }

    /// Stable filename extension without a leading dot.
    pub const fn extension(self) -> &'static str {
        match self {
            Self::ImagePng => "png",
            Self::ImageJpeg => "jpeg",
            Self::AudioWav => "wav",
            Self::AudioMp3 => "mp3",
            Self::VideoMp4 => "mp4",
            Self::VideoWebM => "webm",
        }
    }

    /// MIME type used by inline data URIs and HTTP responses.
    pub const fn mime_type(self) -> &'static str {
        match self {
            Self::ImagePng => "image/png",
            Self::ImageJpeg => "image/jpeg",
            Self::AudioWav => "audio/wav",
            Self::AudioMp3 => "audio/mpeg",
            Self::VideoMp4 => "video/mp4",
            Self::VideoWebM => "video/webm",
        }
    }
}

/// Final publication policy for one encoded synthetic media value.
///
/// This is intentionally separate from [`SyntheticMediaGeneratorFactory`]: a
/// distribution may replace image generation, media publication, or both.
/// Implementations return the exact endpoint-ready bytes interned by the
/// composer.
pub trait SyntheticMediaPublisher: Send + Sync {
    /// Publish one complete encoded media object.
    fn publish(&self, format: SyntheticMediaFormat, encoded: Bytes) -> Result<Bytes>;
}

/// Default publisher retaining image/video data URIs and `format,base64` audio.
#[derive(Debug, Clone, Copy, Default)]
pub struct InlineSyntheticMediaPublisher;

impl SyntheticMediaPublisher for InlineSyntheticMediaPublisher {
    fn publish(&self, format: SyntheticMediaFormat, encoded: Bytes) -> Result<Bytes> {
        let base64 = STANDARD.encode(encoded);
        let wire = match format.kind() {
            MediaKind::Audio => format!("{},{base64}", format.extension()),
            MediaKind::Image | MediaKind::Video => {
                format!("data:{};base64,{base64}", format.mime_type())
            }
            MediaKind::Text => unreachable!("synthetic media formats never represent text"),
        };
        Ok(Bytes::from(wire))
    }
}

/// Stateful generator for one configured media category.
pub trait SyntheticMediaGenerator {
    /// Generate one endpoint-ready value.
    fn generate(&mut self) -> Result<GeneratedMedia>;
}

/// Injectable factory for all native synthetic media categories.
pub trait SyntheticMediaGeneratorFactory: Send + Sync {
    fn image(
        &self,
        config: &SyntheticImageConfig,
        root: RngRoot,
    ) -> Result<Box<dyn SyntheticMediaGenerator>>;

    fn audio(
        &self,
        config: &SyntheticAudioConfig,
        root: RngRoot,
    ) -> Result<Box<dyn SyntheticMediaGenerator>>;

    fn video(
        &self,
        config: &SyntheticVideoConfig,
        root: RngRoot,
    ) -> Result<Box<dyn SyntheticMediaGenerator>>;
}

/// Rust-native image/audio/video generator factory.
#[derive(Clone)]
pub struct NativeSyntheticMediaGeneratorFactory {
    publisher: Arc<dyn SyntheticMediaPublisher>,
}

impl fmt::Debug for NativeSyntheticMediaGeneratorFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeSyntheticMediaGeneratorFactory")
            .finish_non_exhaustive()
    }
}

impl Default for NativeSyntheticMediaGeneratorFactory {
    fn default() -> Self {
        Self::new(Arc::new(InlineSyntheticMediaPublisher))
    }
}

impl NativeSyntheticMediaGeneratorFactory {
    /// Bind all native generators to one final publication policy.
    pub fn new(publisher: Arc<dyn SyntheticMediaPublisher>) -> Self {
        Self { publisher }
    }
}

impl SyntheticMediaGeneratorFactory for NativeSyntheticMediaGeneratorFactory {
    fn image(
        &self,
        config: &SyntheticImageConfig,
        root: RngRoot,
    ) -> Result<Box<dyn SyntheticMediaGenerator>> {
        Ok(Box::new(NativeImageGenerator::new_with_publisher(
            config.clone(),
            root,
            self.publisher.clone(),
        )?))
    }

    fn audio(
        &self,
        config: &SyntheticAudioConfig,
        root: RngRoot,
    ) -> Result<Box<dyn SyntheticMediaGenerator>> {
        Ok(Box::new(NativeAudioGenerator::new_with_publisher(
            config.clone(),
            root,
            self.publisher.clone(),
        )?))
    }

    fn video(
        &self,
        config: &SyntheticVideoConfig,
        root: RngRoot,
    ) -> Result<Box<dyn SyntheticMediaGenerator>> {
        Ok(Box::new(NativeVideoGenerator::new_with_publisher(
            config.clone(),
            root,
            self.publisher.clone(),
        )?))
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
    /// Fraction of generated prompts, in `[0, 1]`, that draw the shared reusable
    /// prefix so a server KV cache observes real prefix hits. The default `0.0`
    /// leaves every prompt fully unique.
    pub prefix_reuse_fraction: f64,
    /// Fraction of each reusing prompt's input length, in `[0, 1]`, occupied by
    /// the shared prefix; the remaining tokens stay unique to that prompt.
    pub prefix_reuse_ratio: f64,
}

impl Default for SyntheticPromptConfig {
    fn default() -> Self {
        Self {
            input_tokens: fixed(128.0),
            batch_size: 1,
            prefix_reuse_fraction: 0.0,
            prefix_reuse_ratio: 0.5,
        }
    }
}

/// Synthetic KV-prefix and conversation-context prompt configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SyntheticPrefixConfig {
    /// Number of reusable prefixes in the pool one prefix is drawn from.
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
