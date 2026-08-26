// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AWS `application/vnd.amazon.eventstream` binary frame codec.
//!
//! Used by the AWS SageMaker Runtime `InvokeEndpointWithResponseStream` API.
//! Each message is a length-prefixed, CRC32-checked binary frame:
//!
//! ```text
//! [4B total length][4B headers length][4B prelude CRC32]
//! [headers][payload]
//! [4B message CRC32]
//! ```
//!
//! This module only encodes/decodes the frame envelope. It has no HTTP or SSE
//! dependency so both the mock server (encoder) and the client transport
//! (decoder) can share it without a `core -> http` dependency.

use std::collections::VecDeque;

use bytes::{BufMut, Bytes, BytesMut};

const PRELUDE_LEN: usize = 8;
const CRC_LEN: usize = 4;
/// Minimum frame size: prelude + prelude CRC + message CRC, zero headers/payload.
const MIN_FRAME_LEN: usize = PRELUDE_LEN + CRC_LEN + CRC_LEN;
/// AWS eventstream's maximum permitted total frame length.
pub const MAX_EVENTSTREAM_FRAME_LEN: usize = 16 * 1024 * 1024;

fn crc32(bytes: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB88320;
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// A single AWS eventstream message: raw header bytes plus payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventStreamMessage {
    /// Encoded header block bytes (opaque; not decomposed into name/value pairs
    /// since AIPerf only ever emits/consumes the fixed `:event-type`,
    /// `:content-type`, `:message-type` triad).
    pub headers: Bytes,
    pub payload: Bytes,
}

impl EventStreamMessage {
    /// A `PayloadPart` event message with the standard SageMaker header triad.
    ///
    /// `payload` is the raw inner chat-completion-chunk JSON bytes. The AWS
    /// eventstream `PayloadPart` shape's `Bytes` member carries the
    /// `eventpayload` trait, so the frame payload IS the blob directly — no
    /// base64/JSON envelope wraps it (confirmed against `boto3`'s
    /// `sagemaker-runtime` client, which reads `event["PayloadPart"]["Bytes"]`
    /// as the raw frame payload bytes with no further JSON decoding).
    pub fn payload_part(payload: impl Into<Bytes>) -> Self {
        Self {
            headers: encode_headers(&[
                (":event-type", "PayloadPart"),
                (":content-type", "application/octet-stream"),
                (":message-type", "event"),
            ]),
            payload: payload.into(),
        }
    }

    /// Encode this message into its wire framing.
    pub fn encode(&self) -> Bytes {
        let headers_len = self.headers.len() as u32;
        let total_len =
            (PRELUDE_LEN + CRC_LEN + self.headers.len() + self.payload.len() + CRC_LEN) as u32;

        let mut prelude = BytesMut::with_capacity(PRELUDE_LEN);
        prelude.put_u32(total_len);
        prelude.put_u32(headers_len);
        let prelude_crc = crc32(&prelude);

        let mut out = BytesMut::with_capacity(total_len as usize);
        out.extend_from_slice(&prelude);
        out.put_u32(prelude_crc);
        out.extend_from_slice(&self.headers);
        out.extend_from_slice(&self.payload);
        let message_crc = crc32(&out);
        out.put_u32(message_crc);
        out.freeze()
    }
}

/// Encode a fixed set of AWS eventstream string headers.
///
/// Each header is `[1B name len][name][1B type=7 (string)][2B value
/// len][value]`, matching the wire format AWS SDKs emit for string header
/// values.
fn encode_headers(headers: &[(&str, &str)]) -> Bytes {
    let mut out = BytesMut::new();
    for (name, value) in headers {
        out.put_u8(name.len() as u8);
        out.extend_from_slice(name.as_bytes());
        out.put_u8(7); // header value type: string
        out.put_u16(value.len() as u16);
        out.extend_from_slice(value.as_bytes());
    }
    out.freeze()
}

/// Decode error for a malformed eventstream frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventStreamDecodeError(pub String);

impl std::fmt::Display for EventStreamDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "eventstream decode error: {}", self.0)
    }
}

impl std::error::Error for EventStreamDecodeError {}

/// Incremental decoder: buffers arbitrary byte chunks and yields complete,
/// CRC-verified messages as they become available.
#[derive(Debug, Default)]
pub struct EventStreamDecoder {
    buf: BytesMut,
    frame_len: Option<usize>,
    ready: VecDeque<EventStreamMessage>,
}

impl EventStreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of bytes from the wire.
    pub fn push(&mut self, mut chunk: &[u8]) -> Result<(), EventStreamDecodeError> {
        loop {
            if self.frame_len.is_none() {
                let prelude_len = PRELUDE_LEN + CRC_LEN;
                let copied = (prelude_len - self.buf.len()).min(chunk.len());
                self.buf.extend_from_slice(&chunk[..copied]);
                chunk = &chunk[copied..];
                if self.buf.len() < prelude_len {
                    return Ok(());
                }

                let mut prelude = [0; PRELUDE_LEN + CRC_LEN];
                prelude.copy_from_slice(&self.buf[..prelude_len]);
                self.frame_len = Some(validate_prelude(prelude)?.0);
            }

            let Some(frame_len) = self.frame_len else {
                continue;
            };
            let copied = (frame_len - self.buf.len()).min(chunk.len());
            self.buf.extend_from_slice(&chunk[..copied]);
            chunk = &chunk[copied..];
            if self.buf.len() < frame_len {
                return Ok(());
            }

            self.ready
                .push_back(decode_complete_frame(&mut self.buf, frame_len)?);
            self.frame_len = None;
            if chunk.is_empty() {
                return Ok(());
            }
        }
    }

    /// Decode as many complete messages as are currently buffered.
    pub fn drain_messages(&mut self) -> Result<Vec<EventStreamMessage>, EventStreamDecodeError> {
        Ok(self.ready.drain(..).collect())
    }

    /// True if unconsumed trailing bytes remain (a truncated final frame).
    pub fn has_trailing_bytes(&self) -> bool {
        !self.buf.is_empty()
    }

    #[cfg(test)]
    fn buffered_len(&self) -> usize {
        self.buf.len()
    }
}

fn validate_prelude(
    prelude: [u8; PRELUDE_LEN + CRC_LEN],
) -> Result<(usize, usize), EventStreamDecodeError> {
    let total_len = u32::from_be_bytes([prelude[0], prelude[1], prelude[2], prelude[3]]) as usize;
    let headers_len = u32::from_be_bytes([prelude[4], prelude[5], prelude[6], prelude[7]]) as usize;
    let prelude_crc = u32::from_be_bytes([prelude[8], prelude[9], prelude[10], prelude[11]]);
    if total_len < MIN_FRAME_LEN {
        return Err(EventStreamDecodeError(format!(
            "frame total length {total_len} is smaller than the minimum frame size"
        )));
    }
    if total_len > MAX_EVENTSTREAM_FRAME_LEN {
        return Err(EventStreamDecodeError(format!(
            "frame total length {total_len} exceeds the maximum eventstream frame size"
        )));
    }
    if prelude_crc != crc32(&prelude[..PRELUDE_LEN]) {
        return Err(EventStreamDecodeError("prelude CRC mismatch".to_string()));
    }
    let payload_end = total_len - CRC_LEN;
    let headers_end = (PRELUDE_LEN + CRC_LEN)
        .checked_add(headers_len)
        .ok_or_else(|| EventStreamDecodeError("headers length exceeds frame bounds".to_string()))?;
    if headers_end > payload_end {
        return Err(EventStreamDecodeError(
            "headers length exceeds frame bounds".to_string(),
        ));
    }
    Ok((total_len, headers_len))
}

fn decode_complete_frame(
    buf: &mut BytesMut,
    total_len: usize,
) -> Result<EventStreamMessage, EventStreamDecodeError> {
    let payload_end = total_len - CRC_LEN;
    let message_crc = u32::from_be_bytes([
        buf[payload_end],
        buf[payload_end + 1],
        buf[payload_end + 2],
        buf[payload_end + 3],
    ]);
    if message_crc != crc32(&buf[..payload_end]) {
        return Err(EventStreamDecodeError("message CRC mismatch".to_string()));
    }
    let headers_len = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let headers_start = PRELUDE_LEN + CRC_LEN;
    let payload_start = headers_start + headers_len;
    let mut frame = buf.split_to(total_len);
    let headers = frame
        .split_to(payload_start)
        .split_off(headers_start)
        .freeze();
    frame.truncate(payload_end - payload_start);
    Ok(EventStreamMessage {
        headers,
        payload: frame.freeze(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prelude(total_len: u32, headers_len: u32) -> BytesMut {
        let mut bytes = BytesMut::with_capacity(PRELUDE_LEN + CRC_LEN);
        bytes.put_u32(total_len);
        bytes.put_u32(headers_len);
        bytes.put_u32(crc32(&bytes));
        bytes
    }

    fn assert_rejects_prelude_without_retaining_payload(prelude: BytesMut) {
        let mut chunk = prelude;
        chunk.extend_from_slice(&[0xA5; 4_096]);
        let mut decoder = EventStreamDecoder::new();
        assert!(decoder.push(&chunk).is_err());
        assert!(decoder.buffered_len() <= PRELUDE_LEN + CRC_LEN);
    }

    #[test]
    fn rejects_total_length_smaller_than_minimum_before_retaining_payload() {
        assert_rejects_prelude_without_retaining_payload(prelude((MIN_FRAME_LEN - 1) as u32, 0));
    }

    #[test]
    fn rejects_headers_outside_frame_before_retaining_payload() {
        assert_rejects_prelude_without_retaining_payload(prelude(MIN_FRAME_LEN as u32, 1));
    }

    #[test]
    fn rejects_bad_prelude_crc_before_retaining_payload() {
        let mut invalid = prelude(MIN_FRAME_LEN as u32, 0);
        invalid[PRELUDE_LEN + CRC_LEN - 1] ^= 0xFF;
        assert_rejects_prelude_without_retaining_payload(invalid);
    }

    #[test]
    fn rejects_oversized_frame_before_retaining_payload() {
        assert_rejects_prelude_without_retaining_payload(prelude(
            MAX_EVENTSTREAM_FRAME_LEN as u32 + 1,
            0,
        ));
    }

    #[test]
    fn round_trips_minimum_legal_frame() {
        let message = EventStreamMessage {
            headers: Bytes::new(),
            payload: Bytes::new(),
        };
        let mut decoder = EventStreamDecoder::new();
        decoder
            .push(&message.encode())
            .expect("minimum frame is accepted");
        assert_eq!(decoder.drain_messages().expect("decodes"), vec![message]);
    }

    #[test]
    fn round_trips_single_message() {
        let message = EventStreamMessage::payload_part(Bytes::from_static(b"hello world"));
        let encoded = message.encode();

        let mut decoder = EventStreamDecoder::new();
        decoder.push(&encoded).expect("frame is accepted");
        let decoded = decoder.drain_messages().expect("decode succeeds");

        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].payload, Bytes::from_static(b"hello world"));
        assert_eq!(decoded[0].headers, message.headers);
        assert!(!decoder.has_trailing_bytes());
    }

    #[test]
    fn decodes_multiple_messages_split_across_chunks() {
        let m1 = EventStreamMessage::payload_part(Bytes::from_static(b"chunk-1"));
        let m2 = EventStreamMessage::payload_part(Bytes::from_static(b"chunk-2"));
        let mut wire = BytesMut::new();
        wire.extend_from_slice(&m1.encode());
        wire.extend_from_slice(&m2.encode());
        let wire = wire.freeze();

        // Feed byte-by-byte to exercise partial-frame buffering.
        let mut decoder = EventStreamDecoder::new();
        let mut all = Vec::new();
        for byte in wire.iter() {
            decoder.push(&[*byte]).expect("chunk is accepted");
            all.extend(decoder.drain_messages().unwrap());
        }

        assert_eq!(all.len(), 2);
        assert_eq!(all[0].payload, Bytes::from_static(b"chunk-1"));
        assert_eq!(all[1].payload, Bytes::from_static(b"chunk-2"));
        assert!(!decoder.has_trailing_bytes());
    }

    #[test]
    fn rejects_corrupted_prelude_crc() {
        let message = EventStreamMessage::payload_part(Bytes::from_static(b"x"));
        let mut encoded = BytesMut::from(&message.encode()[..]);
        encoded[11] ^= 0xFF; // flip a byte inside the prelude CRC field

        let mut decoder = EventStreamDecoder::new();
        let err = decoder.push(&encoded).unwrap_err();
        assert!(err.0.contains("prelude CRC"));
    }

    #[test]
    fn rejects_corrupted_message_crc() {
        let message = EventStreamMessage::payload_part(Bytes::from_static(b"x"));
        let mut encoded = BytesMut::from(&message.encode()[..]);
        let last = encoded.len() - 1;
        encoded[last] ^= 0xFF;

        let mut decoder = EventStreamDecoder::new();
        let err = decoder.push(&encoded).unwrap_err();
        assert!(err.0.contains("message CRC"));
    }
}
