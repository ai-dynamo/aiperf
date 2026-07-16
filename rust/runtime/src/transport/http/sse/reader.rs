// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Incremental SSE reader: buffers bytes, splits on `\n\n`/`\r\n\r\n` with a
//! 3-byte cross-chunk back-scan, timestamps each message at chunk arrival via
//! the `Clock`, handles JSON continuations, and flushes a trailing message.
//! Behavioral port of Python `AsyncSSEStreamReader`.

use std::rc::Rc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::Stream;
use futures::StreamExt;
use futures::future::poll_fn;

use crate::clock::Clock;

use crate::transport::core::ErrorDetails;
use crate::transport::core::SseMessage;

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
/// Python's `AsyncSSEStreamReader` yields every framed message until its input
/// iterator ends; neither it nor `AioHttpClient` exits on `[DONE]`. Draining is
/// also what makes an HTTP/1 response safe to return to the shared pool.
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

            let raw = String::from_utf8_lossy(&buffer[consumed..idx]);
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
        // "\n\n" arrives split: "...a\n" then "\ndata: b\n\n"
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
}
