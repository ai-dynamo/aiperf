// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native synthetic audio and source-audio normalization.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use crate::rng::{RandomGenerator, RngRoot};
use bytes::Bytes;

use super::{
    GeneratedMedia, InlineSyntheticMediaPublisher, SyntheticAudioConfig, SyntheticAudioFormat,
    SyntheticMediaFormat, SyntheticMediaGenerator, SyntheticMediaPublisher,
};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::model::MediaKind;

const MP3_SAMPLE_RATES: &[u32] = &[
    8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
];
const SUPPORTED_DEPTHS: &[u16] = &[8, 16, 24, 32];

/// Rust-native Gaussian-noise audio generator.
pub struct NativeAudioGenerator {
    config: SyntheticAudioConfig,
    duration_rng: RandomGenerator,
    format_rng: RandomGenerator,
    data_rng: RandomGenerator,
    publisher: Arc<dyn SyntheticMediaPublisher>,
}

impl NativeAudioGenerator {
    /// Validate audio geometry and initialize independent deterministic RNG streams.
    pub fn new(config: SyntheticAudioConfig, root: RngRoot) -> Result<Self> {
        Self::new_with_publisher(config, root, Arc::new(InlineSyntheticMediaPublisher))
    }

    /// Validate audio geometry and bind an injected final publication policy.
    pub fn new_with_publisher(
        config: SyntheticAudioConfig,
        root: RngRoot,
        publisher: Arc<dyn SyntheticMediaPublisher>,
    ) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self {
            config,
            duration_rng: RandomGenerator::from_seed(root.derive_seed("dataset.audio.duration")),
            format_rng: RandomGenerator::from_seed(root.derive_seed("dataset.audio.format")),
            data_rng: RandomGenerator::from_seed(root.derive_seed("dataset.audio.data")),
            publisher,
        })
    }

    pub(crate) fn generate_wav(
        &mut self,
        duration_seconds: f64,
        sample_rate_hz: u32,
        bit_depth: u16,
        channels: u16,
    ) -> Result<Bytes> {
        generate_noise_wav(
            &mut self.data_rng,
            duration_seconds,
            sample_rate_hz,
            bit_depth,
            channels,
        )
    }
}

impl SyntheticMediaGenerator for NativeAudioGenerator {
    fn generate(&mut self) -> Result<GeneratedMedia> {
        let sampled = self
            .config
            .duration_seconds
            .sample(&mut self.duration_rng)
            .map_err(|error| DatasetError::Validation(error.to_string()))?;
        let duration = sampled.max(0.01);
        let sample_rate_hz = *self
            .format_rng
            .choice(&self.config.sample_rates_hz)
            .map_err(|error| DatasetError::Validation(error.to_string()))?;
        let bit_depth = *self
            .format_rng
            .choice(&self.config.bit_depths)
            .map_err(|error| DatasetError::Validation(error.to_string()))?;
        let wav = self.generate_wav(duration, sample_rate_hz, bit_depth, self.config.channels)?;
        let (format, encoded) = match self.config.format {
            SyntheticAudioFormat::Wav => (SyntheticMediaFormat::AudioWav, wav),
            SyntheticAudioFormat::Mp3 => (SyntheticMediaFormat::AudioMp3, encode_mp3(&wav)?),
        };
        Ok(GeneratedMedia {
            kind: MediaKind::Audio,
            wire: self.publisher.publish(format, encoded)?,
            duration_seconds: Some(duration),
        })
    }
}

/// Return the duration of WAV/FFmpeg-readable audio bytes.
pub fn audio_duration_seconds(raw: &[u8]) -> Result<f64> {
    if let Some(duration) = wav_duration(raw)? {
        return Ok(duration);
    }
    let output = run_filter(
        "ffprobe",
        &[
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            "pipe:0",
        ],
        raw,
    )?;
    let duration = std::str::from_utf8(&output)
        .map_err(|error| DatasetError::Validation(format!("ffprobe returned non-UTF-8: {error}")))?
        .trim()
        .parse::<f64>()
        .map_err(|error| DatasetError::Validation(format!("invalid ffprobe duration: {error}")))?;
    if !duration.is_finite() || duration < 0.0 {
        return Err(DatasetError::Validation(format!(
            "invalid audio duration {duration}"
        )));
    }
    Ok(duration)
}

/// Decode any FFmpeg-readable audio into PCM WAV and return its exact duration.
pub fn transcode_audio_to_wav(raw: &[u8]) -> Result<(Bytes, f64)> {
    if let Some(duration) = wav_duration(raw)? {
        return Ok((Bytes::copy_from_slice(raw), duration));
    }
    let wav = run_filter(
        "ffmpeg",
        &[
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ],
        raw,
    )?;
    let duration = wav_duration(&wav)?
        .ok_or_else(|| DatasetError::Validation("FFmpeg returned an invalid WAV stream".into()))?;
    Ok((Bytes::from(wav), duration))
}

pub(crate) fn generate_noise_wav(
    rng: &mut RandomGenerator,
    duration_seconds: f64,
    sample_rate_hz: u32,
    bit_depth: u16,
    channels: u16,
) -> Result<Bytes> {
    if !duration_seconds.is_finite() || duration_seconds < 0.01 {
        return Err(DatasetError::Validation(format!(
            "audio duration must be finite and at least 0.01 seconds, got {duration_seconds}"
        )));
    }
    validate_pcm(sample_rate_hz, bit_depth, channels)?;
    let frames = (duration_seconds * f64::from(sample_rate_hz)).floor();
    if frames >= usize::MAX as f64 {
        return Err(DatasetError::Validation(
            "audio sample count exceeds addressable memory".into(),
        ));
    }
    let frames = frames as usize;
    let samples = frames
        .checked_mul(usize::from(channels))
        .ok_or_else(|| DatasetError::Validation("audio sample count overflow".into()))?;
    let bytes_per_sample = usize::from(bit_depth / 8);
    let data_len = samples
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| DatasetError::Validation("audio byte count overflow".into()))?;
    let data_len_u32 = u32::try_from(data_len)
        .map_err(|_| DatasetError::Validation("WAV data exceeds the 4 GiB RIFF limit".into()))?;
    let block_align = channels
        .checked_mul(bit_depth / 8)
        .ok_or_else(|| DatasetError::Validation("WAV block alignment overflow".into()))?;
    let byte_rate = sample_rate_hz
        .checked_mul(u32::from(block_align))
        .ok_or_else(|| DatasetError::Validation("WAV byte rate overflow".into()))?;

    let mut wav = Vec::with_capacity(44 + data_len);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36_u32 + data_len_u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate_hz.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bit_depth.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len_u32.to_le_bytes());

    for _ in 0..samples {
        let sample = rng
            .normal(0.0, 0.3)
            .map_err(|error| DatasetError::Validation(error.to_string()))?
            .clamp(-1.0, 1.0);
        match bit_depth {
            8 => wav.push(((sample + 1.0) * 127.5).round() as u8),
            16 => wav.extend_from_slice(&((sample * f64::from(i16::MAX)) as i16).to_le_bytes()),
            24 => {
                let value = (sample * 8_388_607.0) as i32;
                wav.extend_from_slice(&value.to_le_bytes()[..3]);
            }
            32 => wav.extend_from_slice(&((sample * f64::from(i32::MAX)) as i32).to_le_bytes()),
            _ => unreachable!("validate_pcm rejects unsupported depths"),
        }
    }
    Ok(Bytes::from(wav))
}

fn validate_config(config: &SyntheticAudioConfig) -> Result<()> {
    if config.batch_size == 0 {
        return Err(DatasetError::Validation(
            "an audio generator requires batch_size > 0".into(),
        ));
    }
    if config.duration_seconds.expected_value() < 0.01 {
        return Err(DatasetError::Validation(
            "audio duration expected value must be at least 0.01 seconds".into(),
        ));
    }
    if config.sample_rates_hz.is_empty() || config.bit_depths.is_empty() {
        return Err(DatasetError::Validation(
            "audio sample-rate and bit-depth lists must be non-empty".into(),
        ));
    }
    for sample_rate in &config.sample_rates_hz {
        validate_pcm(*sample_rate, config.bit_depths[0], config.channels)?;
        if config.format == SyntheticAudioFormat::Mp3 && !MP3_SAMPLE_RATES.contains(sample_rate) {
            return Err(DatasetError::Validation(format!(
                "MP3 does not support sample rate {sample_rate}; supported rates are {MP3_SAMPLE_RATES:?}"
            )));
        }
    }
    for depth in &config.bit_depths {
        validate_pcm(config.sample_rates_hz[0], *depth, config.channels)?;
    }
    Ok(())
}

fn validate_pcm(sample_rate_hz: u32, bit_depth: u16, channels: u16) -> Result<()> {
    if sample_rate_hz == 0 {
        return Err(DatasetError::Validation(
            "audio sample rate must be positive".into(),
        ));
    }
    if !SUPPORTED_DEPTHS.contains(&bit_depth) {
        return Err(DatasetError::Validation(format!(
            "unsupported audio bit depth {bit_depth}; expected one of {SUPPORTED_DEPTHS:?}"
        )));
    }
    if !(1..=2).contains(&channels) {
        return Err(DatasetError::Validation(format!(
            "audio channels must be 1 or 2, got {channels}"
        )));
    }
    Ok(())
}

fn wav_duration(raw: &[u8]) -> Result<Option<f64>> {
    if raw.len() < 12 || &raw[..4] != b"RIFF" || &raw[8..12] != b"WAVE" {
        return Ok(None);
    }
    let mut cursor = 12;
    let mut byte_rate = None;
    let mut data_len = None;
    while cursor + 8 <= raw.len() {
        let id = &raw[cursor..cursor + 4];
        let len = u32::from_le_bytes(raw[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        cursor += 8;
        let declared_end = cursor
            .checked_add(len)
            .ok_or_else(|| DatasetError::Validation("WAV chunk length overflow".into()))?;
        // FFmpeg uses 0xffff_ffff for a non-seekable pipe's RIFF/data size.
        // In that legal streaming form, the data chunk consumes the remainder.
        let streaming_data = id == b"data" && declared_end > raw.len();
        let end = if streaming_data {
            raw.len()
        } else {
            declared_end
        };
        if end > raw.len() {
            return Err(DatasetError::Validation(
                "WAV chunk extends past end of input".into(),
            ));
        }
        if id == b"fmt " {
            if len < 16 {
                return Err(DatasetError::Validation(
                    "WAV fmt chunk is too short".into(),
                ));
            }
            byte_rate = Some(u32::from_le_bytes(
                raw[cursor + 8..cursor + 12].try_into().unwrap(),
            ));
        } else if id == b"data" {
            data_len = Some(end - cursor);
        }
        if streaming_data {
            break;
        }
        cursor = end + (len & 1);
    }
    let byte_rate =
        byte_rate.ok_or_else(|| DatasetError::Validation("WAV has no fmt chunk".into()))?;
    let data_len =
        data_len.ok_or_else(|| DatasetError::Validation("WAV has no data chunk".into()))?;
    if byte_rate == 0 {
        return Err(DatasetError::Validation("WAV byte rate is zero".into()));
    }
    Ok(Some(data_len as f64 / f64::from(byte_rate)))
}

fn encode_mp3(wav: &[u8]) -> Result<Bytes> {
    run_filter(
        "ffmpeg",
        &[
            "-v", "error", "-f", "wav", "-i", "pipe:0", "-f", "mp3", "pipe:1",
        ],
        wav,
    )
    .map(Bytes::from)
}

fn run_filter(program: &str, args: &[&str], input: &[u8]) -> Result<Vec<u8>> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            DatasetError::Io(std::io::Error::new(
                error.kind(),
                format!("failed to launch {program}: {error}"),
            ))
        })?;
    // Write stdin on a dedicated thread so the parent can drain stdout/stderr
    // concurrently: FFmpeg output larger than a pipe buffer (~64 KiB, i.e. any
    // audio longer than a couple of seconds) would otherwise deadlock, with the
    // child blocked writing stdout while we block writing stdin.
    let mut stdin = child.stdin.take().expect("piped stdin");
    let owned_input = input.to_vec();
    let writer = std::thread::spawn(move || -> std::io::Result<()> {
        stdin.write_all(&owned_input)?;
        // Drop `stdin` to send EOF so the child can finish and close stdout.
        drop(stdin);
        Ok(())
    });
    let output = child.wait_with_output()?;
    // A BrokenPipe here means the child closed stdin early (typically because it
    // failed); surface its stderr from the status check below instead of masking
    // it with the write error.
    if let Err(error) = writer.join().expect("stdin writer thread panicked")
        && error.kind() != std::io::ErrorKind::BrokenPipe
    {
        return Err(DatasetError::Io(error));
    }
    if !output.status.success() {
        return Err(DatasetError::Validation(format!(
            "{program} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    if output.stdout.is_empty() {
        return Err(DatasetError::Validation(format!(
            "{program} produced no output"
        )));
    }
    Ok(output.stdout)
}

#[cfg(test)]
mod tests {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;

    use super::*;

    #[test]
    fn wav_generation_preserves_geometry_and_duration() {
        let config = SyntheticAudioConfig {
            batch_size: 1,
            duration_seconds: crate::rng::SamplingDistribution::fixed(0.025).unwrap(),
            sample_rates_hz: vec![8_000],
            bit_depths: vec![24],
            channels: 2,
            ..SyntheticAudioConfig::default()
        };
        let mut generator = NativeAudioGenerator::new(config, RngRoot::new(Some(4))).unwrap();
        let generated = generator.generate().unwrap();
        let encoded = generated.wire.split(|byte| *byte == b',').nth(1).unwrap();
        let wav = STANDARD.decode(encoded).unwrap();
        assert_eq!(audio_duration_seconds(&wav).unwrap(), 0.025);
        let (normalized, duration) = transcode_audio_to_wav(&wav).unwrap();
        assert_eq!(normalized.as_ref(), wav);
        assert_eq!(duration, 0.025);
    }

    #[test]
    fn ffmpeg_mp3_round_trip_is_decodable() {
        let config = SyntheticAudioConfig {
            batch_size: 1,
            duration_seconds: crate::rng::SamplingDistribution::fixed(0.1).unwrap(),
            format: SyntheticAudioFormat::Mp3,
            sample_rates_hz: vec![16_000],
            bit_depths: vec![16],
            channels: 1,
        };
        let mut generator = NativeAudioGenerator::new(config, RngRoot::new(Some(5))).unwrap();
        let generated = generator.generate().unwrap();
        let mp3 = STANDARD
            .decode(generated.wire.split(|byte| *byte == b',').nth(1).unwrap())
            .unwrap();
        let (wav, duration) = transcode_audio_to_wav(&mp3).unwrap();
        assert!(wav.starts_with(b"RIFF"));
        assert!(duration >= 0.09);
    }
}
