// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Incremental SSE reader: buffers bytes, splits on `\n\n`/`\r\n\r\n` with a
//! 3-byte cross-chunk back-scan, timestamps each message at chunk arrival via
//! the `Clock`, handles JSON continuations, and flushes a trailing message.

use std::borrow::Cow;
use std::rc::Rc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::Stream;
use futures::StreamExt;
use futures::future::poll_fn;

use crate::clock::Clock;

use crate::transport::core::SseMessage;
use crate::transport::core::{ErrorDetails, ErrorKind};

/// Backpressured consumer for one decoded SSE message.
///
/// The HTTP reader awaits this seam before decoding the next frame, so a slow
/// downstream placement cannot turn one large network chunk into an unbounded
/// queue or a false queue-full transport failure.
pub trait SseMessageHandler {
    /// Reserve capacity for the next decoded message.
    fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), ErrorDetails>>;

    /// Consume one message after [`Self::poll_ready`] returned ready.
    fn start_send(&mut self, message: SseMessage) -> Result<(), ErrorDetails>;
}

/// Read `stream` as SSE, invoking `on_message` per parsed message and draining
/// through transport EOF, including after a `[DONE]` sentinel. Returns `Err`
/// (kind `Sse`) if an `event: error` message is seen, or propagates a stream
/// error as `Err`.
///
/// Draining after `[DONE]` makes an HTTP/1 response safe to return to the pool.
pub async fn read_sse<S>(
    stream: S,
    clock: Rc<dyn Clock>,
    on_message: impl FnMut(SseMessage),
) -> Result<(), ErrorDetails>
where
    S: Stream<Item = Result<Bytes, ErrorDetails>>,
{
    let mut handler = SynchronousSseMessageHandler(on_message);
    read_sse_with_handler(stream, clock, &mut handler).await
}

/// Read `stream` while awaiting an injected message handler after every frame.
pub async fn read_sse_with_handler<S, H>(
    stream: S,
    clock: Rc<dyn Clock>,
    handler: &mut H,
) -> Result<(), ErrorDetails>
where
    S: Stream<Item = Result<Bytes, ErrorDetails>>,
    H: SseMessageHandler + ?Sized,
{
    futures::pin_mut!(stream);

    let mut buffer: Vec<u8> = Vec::new();
    let mut consumed: usize = 0;
    let mut search_offset: usize = 0;

    while let Some(item) = stream.next().await {
        let chunk = item?;
        let arrival_ns = clock.now_ns();
        buffer.extend_from_slice(&chunk);

        loop {
            let scan_from = consumed.max(search_offset.saturating_sub(3));
            let (idx, dlen) = match find_subslice(&buffer, b"\n\n", scan_from) {
                Some(i) => (i, 2usize),
                None => match find_subslice(&buffer, b"\r\n\r\n", scan_from) {
                    Some(i) => (i, 4usize),
                    None => {
                        search_offset = buffer.len();
                        break;
                    }
                },
            };

            // `from_utf8_lossy` validates through the `Utf8Chunks` iterator, which
            // is markedly slower than `str::from_utf8`'s validator; at one SSE
            // message per generated token this scan is on the hottest path in the
            // client. Well-formed UTF-8 (every real endpoint, every frame) takes
            // the fast validator and borrows; anything invalid falls back to the
            // identical lossy replacement so behavior is unchanged.
            let slice = &buffer[consumed..idx];
            let raw = match std::str::from_utf8(slice) {
                Ok(valid) => Cow::Borrowed(valid),
                Err(_) => String::from_utf8_lossy(slice),
            };
            let raw = raw.trim();
            consumed = idx + dlen;
            search_offset = consumed;

            if !raw.is_empty() {
                let msg = SseMessage::parse(raw, arrival_ns);
                if let Some(err) = msg.error_message() {
                    return Err(ErrorDetails::sse(format!(
                        "Error occurred in SSE response: {err}"
                    )));
                }
                poll_fn(|context| handler.poll_ready(context)).await?;
                handler.start_send(msg)?;
            }
        }

        if consumed > 0 {
            buffer.drain(..consumed);
            search_offset = search_offset.saturating_sub(consumed);
            consumed = 0;
        }
    }

    // Flush any trailing delimiter-less message.
    let remaining = String::from_utf8_lossy(&buffer[consumed..]);
    let remaining = remaining.trim();
    if !remaining.is_empty() {
        let final_ns = clock.now_ns();
        let msg = SseMessage::parse(remaining, final_ns);
        if let Some(err) = msg.error_message() {
            return Err(ErrorDetails::sse(format!(
                "Error occurred in SSE response: {err}"
            )));
        }
        poll_fn(|context| handler.poll_ready(context)).await?;
        handler.start_send(msg)?;
    }
    Ok(())
}

/// Read an SSE stream while enforcing a raw-frame cap before the reader
/// reserves or decodes a complete message.
///
/// This deliberately has a separate entry point from [`read_sse_with_handler`]
/// because ordinary benchmark responses retain their existing framing
/// behavior. Bounded model-decision responses use this path so a delimiterless
/// frame cannot make the SSE buffer, raw response, or JSON decoder grow beyond
/// its immutable decision-derived allowance.
pub async fn read_sse_with_bounded_frames<S, H>(
    stream: S,
    clock: Rc<dyn Clock>,
    max_frame_bytes: usize,
    handler: &mut H,
) -> Result<(), ErrorDetails>
where
    S: Stream<Item = Result<Bytes, ErrorDetails>>,
    H: SseMessageHandler + ?Sized,
{
    if max_frame_bytes == 0 {
        return Err(frame_limit_error(max_frame_bytes));
    }
    let max_buffer_bytes = max_frame_bytes
        .checked_add(4)
        .ok_or_else(|| frame_limit_error(max_frame_bytes))?;
    futures::pin_mut!(stream);

    let mut frame = Vec::new();
    while let Some(item) = stream.next().await {
        let chunk = item?;
        let arrival_ns = clock.now_ns();
        for byte in chunk {
            if frame.len() == max_buffer_bytes {
                return Err(frame_limit_error(max_frame_bytes));
            }
            frame
                .try_reserve(1)
                .map_err(|_| frame_allocation_error(max_frame_bytes))?;
            frame.push(byte);

            if frame.len() > max_frame_bytes && !is_sse_delimiter_prefix(&frame[max_frame_bytes..])
            {
                return Err(frame_limit_error(max_frame_bytes));
            }

            if let Some(delimiter_len) = frame_delimiter_len(&frame) {
                let raw_len = frame.len() - delimiter_len;
                emit_sse_frame(&frame[..raw_len], arrival_ns, handler).await?;
                frame.clear();
            }
        }
    }

    // EOF has the same trailing-message behavior as the ordinary reader. The
    // preflight above has already bounded this frame before this decode.
    if !frame.is_empty() {
        let final_ns = clock.now_ns();
        emit_sse_frame(&frame, final_ns, handler).await?;
    }
    Ok(())
}

async fn emit_sse_frame<H>(raw: &[u8], arrival_ns: i64, handler: &mut H) -> Result<(), ErrorDetails>
where
    H: SseMessageHandler + ?Sized,
{
    // `from_utf8_lossy` validates through the `Utf8Chunks` iterator, which is
    // markedly slower than `str::from_utf8`'s validator; at one SSE message per
    // generated token this scan is on the hottest path in the client. Well-
    // formed UTF-8 takes the fast borrowed path, while invalid input preserves
    // the ordinary reader's lossy replacement behavior.
    let raw = match std::str::from_utf8(raw) {
        Ok(valid) => Cow::Borrowed(valid),
        Err(_) => String::from_utf8_lossy(raw),
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(());
    }
    let message = SseMessage::parse(raw, arrival_ns);
    if let Some(error) = message.error_message() {
        return Err(ErrorDetails::sse(format!(
            "Error occurred in SSE response: {error}"
        )));
    }
    poll_fn(|context| handler.poll_ready(context)).await?;
    handler.start_send(message)
}

fn frame_delimiter_len(frame: &[u8]) -> Option<usize> {
    if frame.ends_with(b"\n\n") {
        Some(2)
    } else if frame.ends_with(b"\r\n\r\n") {
        Some(4)
    } else {
        None
    }
}

fn is_sse_delimiter_prefix(bytes: &[u8]) -> bool {
    matches!(
        bytes,
        b"\n" | b"\n\n" | b"\r" | b"\r\n" | b"\r\n\r" | b"\r\n\r\n"
    )
}

fn frame_limit_error(max_frame_bytes: usize) -> ErrorDetails {
    ErrorDetails {
        kind: ErrorKind::Protocol,
        code: None,
        message: format!("SSE frame exceeds bounded {max_frame_bytes}-byte SSE frame limit"),
    }
}

fn frame_allocation_error(max_frame_bytes: usize) -> ErrorDetails {
    ErrorDetails {
        kind: ErrorKind::Other,
        code: None,
        message: format!("unable to reserve the bounded {max_frame_bytes}-byte SSE frame buffer"),
    }
}

struct SynchronousSseMessageHandler<F>(F);

impl<F> SseMessageHandler for SynchronousSseMessageHandler<F>
where
    F: FnMut(SseMessage),
{
    fn poll_ready(&mut self, _context: &mut Context<'_>) -> Poll<Result<(), ErrorDetails>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(&mut self, message: SseMessage) -> Result<(), ErrorDetails> {
        (self.0)(message);
        Ok(())
    }
}

/// First index >= `from` where `needle` occurs in `haystack`.
fn find_subslice(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from >= haystack.len() || needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::SimClock;
    use bytes::Bytes;
    use futures::stream;
    use std::cell::Cell;
    use std::rc::Rc;

    fn collect(chunks: Vec<&'static str>) -> Vec<crate::transport::core::SseMessage> {
        let clock: Rc<dyn crate::clock::Clock> = Rc::new(SimClock::new());
        let s = stream::iter(chunks.into_iter().map(|c| Ok(Bytes::from(c))));
        let mut msgs = Vec::new();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async {
            read_sse(s, clock, |m| msgs.push(m)).await.unwrap();
        });
        msgs
    }

    #[test]
    fn parses_two_messages_split_by_blank_line() {
        let msgs = collect(vec!["data: a\n\ndata: b\n\n"]);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].data(), Some("a"));
        assert_eq!(msgs[1].data(), Some("b"));
    }

    #[test]
    fn delimiter_split_across_chunks() {
        let msgs = collect(vec!["data: a\n", "\ndata: b\n\n"]);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].data(), Some("a"));
    }

    #[test]
    fn crlf_delimiter() {
        let msgs = collect(vec!["data: a\r\n\r\n"]);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].data(), Some("a"));
    }

    #[test]
    fn trailing_message_without_final_delimiter_is_flushed() {
        let msgs = collect(vec!["data: a\n\ndata: b"]);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1].data(), Some("b"));
    }

    #[test]
    fn done_is_delivered_without_abandoning_the_response_body() {
        let msgs = collect(vec!["data: a\n\ndata: [DONE]\n\ndata: c\n\n"]);
        assert_eq!(msgs.len(), 3);
        assert!(msgs[1].is_done());
        assert_eq!(msgs[2].data(), Some("c"));
    }

    #[test]
    fn bounded_reader_rejects_a_delimiterless_frame_before_decoding_or_another_read() {
        let reads = Rc::new(Cell::new(0_usize));
        let source_reads = reads.clone();
        let stream = stream::unfold(0_usize, move |index| {
            let reads = source_reads.clone();
            async move {
                let chunk = match index {
                    0 => Bytes::from_static(b"data: 12"),
                    1 => Bytes::from_static(b"3"),
                    2 => Bytes::from_static(b"unread\n\n"),
                    _ => return None,
                };
                reads.set(reads.get() + 1);
                Some((Ok(chunk), index + 1))
            }
        });
        let clock: Rc<dyn crate::clock::Clock> = Rc::new(SimClock::new());
        let mut messages = Vec::new();
        let mut handler = SynchronousSseMessageHandler(|message| messages.push(message));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let local = tokio::task::LocalSet::new();
        let error = local.block_on(&runtime, async {
            read_sse_with_bounded_frames(stream, clock, 8, &mut handler)
                .await
                .unwrap_err()
        });

        assert_eq!(error.kind, crate::transport::core::ErrorKind::Protocol);
        assert!(error.message.contains("bounded 8-byte SSE frame limit"));
        assert_eq!(reads.get(), 2, "the reader must abort before another read");
        assert!(messages.is_empty(), "no decoded frame reaches the handler");
    }

    #[test]
    fn bounded_reader_accepts_an_escaped_json_frame_at_the_exact_limit() {
        let raw = Bytes::from_static(b"data: {\"content\":\"\\u20ac\"}\n\n");
        let exact_limit = raw.len() - 2;
        let stream = stream::iter(std::iter::once(Ok(raw)));
        let clock: Rc<dyn crate::clock::Clock> = Rc::new(SimClock::new());
        let mut messages = Vec::new();
        let mut handler = SynchronousSseMessageHandler(|message| messages.push(message));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let local = tokio::task::LocalSet::new();
        local.block_on(&runtime, async {
            read_sse_with_bounded_frames(stream, clock, exact_limit, &mut handler)
                .await
                .unwrap();
        });

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].data(), Some("{\"content\":\"\\u20ac\"}"));
    }

    #[test]
    fn bounded_reader_accepts_utf8_split_across_chunks_at_the_exact_limit() {
        let chunks = [
            Bytes::from_static(b"data: \xe2"),
            Bytes::from_static(b"\x82\xac\n\n"),
        ];
        let exact_limit = chunks.iter().map(Bytes::len).sum::<usize>() - 2;
        let stream = stream::iter(chunks.into_iter().map(Ok));
        let clock: Rc<dyn crate::clock::Clock> = Rc::new(SimClock::new());
        let mut messages = Vec::new();
        let mut handler = SynchronousSseMessageHandler(|message| messages.push(message));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let local = tokio::task::LocalSet::new();
        local.block_on(&runtime, async {
            read_sse_with_bounded_frames(stream, clock, exact_limit, &mut handler)
                .await
                .unwrap();
        });

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].data(), Some("€"));
    }
}
