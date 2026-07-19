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

use bytes::{BufMut, Bytes, BytesMut};

const PRELUDE_LEN: usize = 8;
const CRC_LEN: usize = 4;
/// Minimum frame size: prelude + prelude CRC + message CRC, zero headers/payload.
const MIN_FRAME_LEN: usize = PRELUDE_LEN + CRC_LEN + CRC_LEN;

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
    pub fn payload_part(payload: impl Into<Bytes>) -> Self {
        Self {
            headers: encode_headers(&[
                (":event-type", "PayloadPart"),
                (":content-type", "application/json"),
                (":message-type", "event"),
            ]),
            payload: payload.into(),
        }
    }

    /// Encode this message into its wire framing.
    pub fn encode(&self) -> Bytes {
        let headers_len = self.headers.len() as u32;
        let total_len = (PRELUDE_LEN + CRC_LEN + self.headers.len() + self.payload.len() + CRC_LEN)
            as u32;

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
}

impl EventStreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of bytes from the wire.
    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// Decode as many complete messages as are currently buffered.
    pub fn drain_messages(&mut self) -> Result<Vec<EventStreamMessage>, EventStreamDecodeError> {
        let mut out = Vec::new();
        loop {
            if self.buf.len() < PRELUDE_LEN + CRC_LEN {
                return Ok(out);
            }
            let total_len = u32::from_be_bytes(self.buf[0..4].try_into().unwrap()) as usize;
            if total_len < MIN_FRAME_LEN {
                return Err(EventStreamDecodeError(format!(
                    "frame total length {total_len} is smaller than the minimum frame size"
                )));
            }
            if self.buf.len() < total_len {
                return Ok(out);
            }

            let headers_len = u32::from_be_bytes(self.buf[4..8].try_into().unwrap()) as usize;
            let prelude_crc = u32::from_be_bytes(self.buf[8..12].try_into().unwrap());
            let computed_prelude_crc = crc32(&self.buf[0..PRELUDE_LEN]);
            if prelude_crc != computed_prelude_crc {
                return Err(EventStreamDecodeError(
                    "prelude CRC mismatch".to_string(),
                ));
            }

            let headers_start = PRELUDE_LEN + CRC_LEN;
            let payload_start = headers_start + headers_len;
            if payload_start > total_len - CRC_LEN {
                return Err(EventStreamDecodeError(
                    "headers length exceeds frame bounds".to_string(),
                ));
            }
            let payload_end = total_len - CRC_LEN;

            let message_crc =
                u32::from_be_bytes(self.buf[payload_end..total_len].try_into().unwrap());
            let computed_message_crc = crc32(&self.buf[0..payload_end]);
            if message_crc != computed_message_crc {
                return Err(EventStreamDecodeError(
                    "message CRC mismatch".to_string(),
                ));
            }

            let mut frame = self.buf.split_to(total_len);
            let headers = frame.split_to(payload_start).split_off(headers_start).freeze();
            frame.truncate(payload_end - payload_start);
            let payload = frame.freeze();

            out.push(EventStreamMessage { headers, payload });
        }
    }

    /// True if unconsumed trailing bytes remain (a truncated final frame).
    pub fn has_trailing_bytes(&self) -> bool {
        !self.buf.is_empty()
    }
}

/// Extract `PayloadPart.Bytes` (base64) from a decoded SageMaker eventstream
/// message payload and base64-decode it back to the inner chat-completion-chunk
/// JSON bytes. Returns `None` if the payload isn't the expected
/// `{"PayloadPart":{"Bytes": "..."}}` shape.
pub fn decode_payload_part(payload: &[u8]) -> Option<Bytes> {
    use base64::Engine as _;
    let value: serde_json::Value = serde_json::from_slice(payload).ok()?;
    let b64 = value.get("PayloadPart")?.get("Bytes")?.as_str()?;
    base64::engine::general_purpose::STANDARD
        .decode(b64)
        .ok()
        .map(Bytes::from)
}

/// Build the `{"PayloadPart":{"Bytes": "<base64>"}}` envelope for one chunk of
/// inner (chat-completion-chunk) JSON bytes.
pub fn encode_payload_part(inner_json: &[u8]) -> Bytes {
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD.encode(inner_json);
    let value = serde_json::json!({"PayloadPart": {"Bytes": b64}});
    Bytes::from(serde_json::to_vec(&value).expect("json serialization of PayloadPart"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_single_message() {
        let message = EventStreamMessage::payload_part(Bytes::from_static(b"hello world"));
        let encoded = message.encode();

        let mut decoder = EventStreamDecoder::new();
        decoder.push(&encoded);
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
            decoder.push(&[*byte]);
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
        decoder.push(&encoded);
        let err = decoder.drain_messages().unwrap_err();
        assert!(err.0.contains("prelude CRC"));
    }

    #[test]
    fn rejects_corrupted_message_crc() {
        let message = EventStreamMessage::payload_part(Bytes::from_static(b"x"));
        let mut encoded = BytesMut::from(&message.encode()[..]);
        let last = encoded.len() - 1;
        encoded[last] ^= 0xFF;

        let mut decoder = EventStreamDecoder::new();
        decoder.push(&encoded);
        let err = decoder.drain_messages().unwrap_err();
        assert!(err.0.contains("message CRC"));
    }

    #[test]
    fn payload_part_round_trips_through_base64_envelope() {
        let inner = br#"{"choices":[{"delta":{"content":"hi"}}]}"#;
        let enveloped = encode_payload_part(inner);
        let decoded = decode_payload_part(&enveloped).expect("envelope decodes");
        assert_eq!(&decoded[..], &inner[..]);
    }
}
