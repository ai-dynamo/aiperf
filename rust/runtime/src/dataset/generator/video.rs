// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! FFmpeg-backed synthetic video generator.

use std::f64::consts::PI;
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use crate::rng::{ConfiguredRandomGenerator, RngRoot, RuntimeRandomGenerator};
use bytes::Bytes;

use super::audio::generate_noise_wav;
use super::{
    GeneratedMedia, InlineSyntheticMediaPublisher, SyntheticMediaFormat, SyntheticMediaGenerator,
    SyntheticMediaPublisher, SyntheticVideoConfig, SyntheticVideoFormat, SyntheticVideoPattern,
};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::model::MediaKind;

/// Rust-native frame synthesizer and FFmpeg encoder.
pub struct NativeVideoGenerator {
    config: SyntheticVideoConfig,
    noise_rng: ConfiguredRandomGenerator,
    audio_rng: ConfiguredRandomGenerator,
    publisher: Arc<dyn SyntheticMediaPublisher>,
}

impl NativeVideoGenerator {
    /// Validate video/audio geometry and initialize deterministic RNG streams.
    pub fn new(config: SyntheticVideoConfig, root: RngRoot) -> Result<Self> {
        Self::new_with_publisher(config, root, Arc::new(InlineSyntheticMediaPublisher))
    }

    /// Validate video/audio geometry and bind an injected final publication policy.
    pub fn new_with_publisher(
        config: SyntheticVideoConfig,
        root: RngRoot,
        publisher: Arc<dyn SyntheticMediaPublisher>,
    ) -> Result<Self> {
        validate(&config)?;
        Ok(Self {
            config,
            noise_rng: root.derive_generator("dataset.video.noise"),
            audio_rng: root.derive_generator("dataset.video.audio"),
            publisher,
        })
    }

    fn frame_count(&self) -> Result<usize> {
        let frames = self.config.duration_seconds * f64::from(self.config.frames_per_second);
        if !frames.is_finite() || frames < 1.0 || frames >= usize::MAX as f64 {
            return Err(DatasetError::Validation(format!(
                "video duration/fps produces invalid frame count {frames}"
            )));
        }
        Ok(frames.floor() as usize)
    }

    fn raw_frames(&mut self) -> Result<Vec<u8>> {
        let frame_count = self.frame_count()?;
        let frame_bytes = usize::try_from(self.config.width)
            .ok()
            .and_then(|width| {
                usize::try_from(self.config.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| DatasetError::Validation("video frame size overflow".into()))?;
        let total = frame_bytes
            .checked_mul(frame_count)
            .ok_or_else(|| DatasetError::Validation("video frame buffer overflow".into()))?;
        let mut frames = Vec::with_capacity(total);
        for frame_index in 0..frame_count {
            let mut frame = vec![0; frame_bytes];
            match self.config.pattern {
                SyntheticVideoPattern::MovingShapes => draw_moving_shapes(
                    &mut frame,
                    self.config.width,
                    self.config.height,
                    frame_index,
                    frame_count,
                ),
                SyntheticVideoPattern::GridClock => draw_grid_clock(
                    &mut frame,
                    self.config.width,
                    self.config.height,
                    frame_index,
                    frame_count,
                ),
                SyntheticVideoPattern::Noise => self.noise_rng.fill_bytes(&mut frame),
            }
            frames.extend(frame);
        }
        Ok(frames)
    }

    fn encode(&mut self, frames: &[u8]) -> Result<Bytes> {
        let directory = tempfile::tempdir()?;
        let output_path = directory
            .path()
            .join(format!("output.{}", self.config.format.extension()));
        let mut command = Command::new("ffmpeg");
        command.args([
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            &format!("{}x{}", self.config.width, self.config.height),
            "-framerate",
            &self.config.frames_per_second.to_string(),
            "-i",
            "pipe:0",
        ]);

        if self.config.audio.channels > 0 {
            let audio_path = directory.path().join("audio.wav");
            let wav = generate_noise_wav(
                &mut self.audio_rng,
                self.config.duration_seconds,
                self.config.audio.sample_rate_hz,
                self.config.audio.bit_depth,
                self.config.audio.channels,
            )?;
            fs::write(&audio_path, wav)?;
            command.arg("-i").arg(&audio_path).arg("-shortest");
            let codec = self
                .config
                .audio
                .codec
                .as_deref()
                .unwrap_or(match self.config.format {
                    SyntheticVideoFormat::Mp4 => "libopus",
                    SyntheticVideoFormat::WebM => "libvorbis",
                });
            command.arg("-c:a").arg(codec);
        }

        command
            .arg("-c:v")
            .arg(&self.config.codec)
            .args(["-pix_fmt", "yuv420p"])
            .arg(&output_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().map_err(|error| {
            DatasetError::Io(std::io::Error::new(
                error.kind(),
                format!("failed to launch FFmpeg: {error}"),
            ))
        })?;
        child.stdin.take().expect("piped stdin").write_all(frames)?;
        let output = child.wait_with_output()?;
        if !output.status.success() {
            return Err(DatasetError::Validation(format!(
                "FFmpeg video encoding failed for codec {:?}: {}",
                self.config.codec,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        let bytes = fs::read(&output_path)?;
        if bytes.is_empty() {
            return Err(DatasetError::Validation(
                "FFmpeg produced an empty video".into(),
            ));
        }
        Ok(Bytes::from(bytes))
    }
}

impl SyntheticMediaGenerator for NativeVideoGenerator {
    fn generate(&mut self) -> Result<GeneratedMedia> {
        let frames = self.raw_frames()?;
        let encoded = self.encode(&frames)?;
        let format = match self.config.format {
            SyntheticVideoFormat::Mp4 => SyntheticMediaFormat::VideoMp4,
            SyntheticVideoFormat::WebM => SyntheticMediaFormat::VideoWebM,
        };
        Ok(GeneratedMedia {
            kind: MediaKind::Video,
            wire: self.publisher.publish(format, encoded)?,
            duration_seconds: Some(self.config.duration_seconds),
        })
    }
}

fn validate(config: &SyntheticVideoConfig) -> Result<()> {
    if config.batch_size == 0 {
        return Err(DatasetError::Validation(
            "a video generator requires batch_size > 0".into(),
        ));
    }
    if config.width == 0 || config.height == 0 {
        return Err(DatasetError::Validation(
            "video width and height must be positive".into(),
        ));
    }
    if config.frames_per_second == 0
        || !config.duration_seconds.is_finite()
        || config.duration_seconds <= 0.0
    {
        return Err(DatasetError::Validation(
            "video duration and frame rate must be finite and positive".into(),
        ));
    }
    if config.codec.trim().is_empty() {
        return Err(DatasetError::Validation(
            "video codec cannot be empty".into(),
        ));
    }
    if config.audio.channels > 2 {
        return Err(DatasetError::Validation(
            "video audio channels must be 0, 1, or 2".into(),
        ));
    }
    if config.audio.channels == 0 && config.audio.codec.is_some() {
        return Err(DatasetError::Validation(
            "video audio codec is set while the audio track is disabled".into(),
        ));
    }
    if config.audio.channels > 0
        && (config.audio.sample_rate_hz == 0 || !matches!(config.audio.bit_depth, 8 | 16 | 24 | 32))
    {
        return Err(DatasetError::Validation(
            "enabled video audio requires a positive sample rate and 8/16/24/32-bit depth".into(),
        ));
    }
    Ok(())
}

fn draw_moving_shapes(
    frame: &mut [u8],
    width: u32,
    height: u32,
    frame_index: usize,
    frame_count: usize,
) {
    let progress = frame_index as f64 / frame_count as f64;
    let red_x = (progress * f64::from(width) * 2.0) as i32;
    let red_y = (height / 2) as i32;
    draw_circle(frame, width, height, red_x, red_y, 15, [255, 0, 0]);
    let green_x = (width / 2) as i32;
    let green_y = (progress * f64::from(height) * 2.0) as i32;
    draw_rectangle(frame, width, height, green_x, green_y, 12, [0, 255, 0]);
    let blue_x = (f64::from(width) * (1.0 - progress * 1.5)) as i32;
    let blue_y = (f64::from(height) * (1.0 - progress * 1.5)) as i32;
    draw_circle(frame, width, height, blue_x, blue_y, 10, [0, 0, 255]);
}

fn draw_grid_clock(
    frame: &mut [u8],
    width: u32,
    height: u32,
    frame_index: usize,
    frame_count: usize,
) {
    for pixel in frame.chunks_exact_mut(3) {
        pixel.copy_from_slice(&[32, 32, 32]);
    }
    for x in (0..width).step_by(32) {
        draw_line(
            frame,
            width,
            height,
            x as i32,
            0,
            x as i32,
            height as i32 - 1,
            [64, 64, 64],
        );
    }
    for y in (0..height).step_by(32) {
        draw_line(
            frame,
            width,
            height,
            0,
            y as i32,
            width as i32 - 1,
            y as i32,
            [64, 64, 64],
        );
    }
    let center_x = width as i32 / 2;
    let center_y = height as i32 / 2;
    let radius = f64::from(width.min(height)) / 4.0;
    let angle = frame_index as f64 / frame_count as f64 * 2.0 * PI;
    let minute_x = center_x + (radius * 0.9 * (angle - PI / 2.0).cos()) as i32;
    let minute_y = center_y + (radius * 0.9 * (angle - PI / 2.0).sin()) as i32;
    draw_line(
        frame,
        width,
        height,
        center_x,
        center_y,
        minute_x,
        minute_y,
        [255, 255, 255],
    );
    let hour = angle / 12.0;
    let hour_x = center_x + (radius * 0.6 * (hour - PI / 2.0).cos()) as i32;
    let hour_y = center_y + (radius * 0.6 * (hour - PI / 2.0).sin()) as i32;
    draw_line(
        frame,
        width,
        height,
        center_x,
        center_y,
        hour_x,
        hour_y,
        [255, 255, 0],
    );
    draw_circle(frame, width, height, center_x, center_y, 3, [255, 0, 0]);
}

fn draw_circle(
    frame: &mut [u8],
    width: u32,
    height: u32,
    center_x: i32,
    center_y: i32,
    radius: i32,
    color: [u8; 3],
) {
    for y in -radius..=radius {
        for x in -radius..=radius {
            if x * x + y * y <= radius * radius {
                put_wrapped(frame, width, height, center_x + x, center_y + y, color);
            }
        }
    }
}

fn draw_rectangle(
    frame: &mut [u8],
    width: u32,
    height: u32,
    center_x: i32,
    center_y: i32,
    half_size: i32,
    color: [u8; 3],
) {
    for y in -half_size..=half_size {
        for x in -half_size..=half_size {
            put_wrapped(frame, width, height, center_x + x, center_y + y, color);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_line(
    frame: &mut [u8],
    width: u32,
    height: u32,
    mut x0: i32,
    mut y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 3],
) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        put(frame, width, height, x0, y0, color);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let twice = 2 * error;
        if twice >= dy {
            error += dy;
            x0 += sx;
        }
        if twice <= dx {
            error += dx;
            y0 += sy;
        }
    }
}

fn put_wrapped(frame: &mut [u8], width: u32, height: u32, x: i32, y: i32, color: [u8; 3]) {
    put(
        frame,
        width,
        height,
        x.rem_euclid(width as i32),
        y.rem_euclid(height as i32),
        color,
    );
}

fn put(frame: &mut [u8], width: u32, height: u32, x: i32, y: i32, color: [u8; 3]) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }
    let offset = (y as usize * width as usize + x as usize) * 3;
    frame[offset..offset + 3].copy_from_slice(&color);
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;

    use super::*;

    #[test]
    #[ignore = "requires ffmpeg for generation and ffprobe for stream inspection"]
    fn ffmpeg_encodes_a_real_video_with_audio() {
        let config = SyntheticVideoConfig {
            batch_size: 1,
            width: 16,
            height: 16,
            duration_seconds: 0.25,
            frames_per_second: 8,
            format: SyntheticVideoFormat::WebM,
            codec: "libvpx-vp9".into(),
            pattern: SyntheticVideoPattern::GridClock,
            audio: super::super::SyntheticVideoAudioConfig {
                channels: 1,
                sample_rate_hz: 8_000,
                bit_depth: 16,
                codec: Some("libvorbis".into()),
            },
        };
        let mut generator = NativeVideoGenerator::new(config, RngRoot::new(Some(7))).unwrap();
        let generated = generator.generate().unwrap();
        let video = STANDARD
            .decode(generated.wire.split(|byte| *byte == b',').nth(1).unwrap())
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("generated.webm");
        fs::write(&path, video).unwrap();
        let output = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(output.status.success());
        let streams = String::from_utf8(output.stdout).unwrap();
        assert!(streams.lines().any(|line| line == "video"));
        assert!(streams.lines().any(|line| line == "audio"));
    }
}
