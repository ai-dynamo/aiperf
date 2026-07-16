// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native `aiperf chat` — pure-Rust port of `aiperf.cli_commands.chat` +
//! `_chat_stats`. No Python: an interactive (or `--quick` single-shot)
//! OpenAI-compatible chat client that streams the reply live and prints the same
//! per-turn stats block (TTFT / TPS / ITL / cache) after each turn.
//!
//! Reuses runner infrastructure rather than re-implementing it: the tiktoken/HF
//! tokenizer (`aiperf_runtime::dataset::tokenizer`) for client-side token counts, the
//! canonical SSE reader (`aiperf_runtime::transport::http::sse::read_sse`, the behavioral
//! port of the same `AsyncSSEStreamReader` Python's `chat` uses — it owns the
//! multibyte-across-TCP-chunk + JSON-continuation edge cases), and a single
//! **injected `Clock`** for all timing (the runner's `RealClock`, substitutable
//! by a `SimClock`) — never `Instant::now`/`SystemTime::now`. Only the
//! connection itself is a lightweight direct hyper client (the full
//! `transport::http` client is a Clock-injected, connection-pooled, cancellation-
//! aware *measured-dispatch* engine — overkill for a sequential REPL); no
//! reqwest, so loopback is never proxied.
//!
//! Per-turn metric formulas mirror the aiperf metric classes exactly (`TTFT =
//! first content ts − start`, `RequestLatency = last content ts − start`, `ITL =
//! (latency − ttft)/(osl−1)`); the wall-clock ms are non-deterministic but
//! OSL/ISL/cache and the line format are byte-exact vs Python. Supports
//! `http://` and `https://` (TLS via tokio-rustls + webpki roots, the same
//! crypto stack the runner's transport / exporters use).

use std::io::Write as _;
use std::sync::Arc;

use anyhow::Context as _;
use futures::StreamExt as _;
use http_body_util::BodyExt as _;
use hyper::Request;
use serde_json::{Value, json};

const NANOS_PER_MILLIS: f64 = 1_000_000.0;
const NANOS_PER_SECOND: f64 = 1_000_000_000.0;

/// One arrived content/reasoning chunk with its perf timestamp (`ParsedResponse`).
struct Chunk {
    perf_ns: u128,
    content: String,
    reasoning: String,
}

/// Resolve a base URL to a chat-completions endpoint (`_chat_completions_url` +
/// `normalize_http_url`): prepend `http://` when scheme-less, strip a trailing
/// slash, then append `/v1/chat/completions` unless already a chat/`/v1` path.
fn chat_completions_url(url: &str) -> String {
    let with_scheme = if url.contains("://") {
        url.to_string()
    } else {
        format!("http://{url}")
    };
    let base = with_scheme.trim_end_matches('/');
    if base.ends_with("/chat/completions") {
        base.to_string()
    } else if base.ends_with("/v1") {
        format!("{base}/chat/completions")
    } else {
        format!("{base}/v1/chat/completions")
    }
}

/// A loaded tokenizer for client-side counts.
enum ChatTokenizer {
    Builtin(aiperf_runtime::dataset::tokenizer::TiktokenTokenizer),
    Hf(aiperf_runtime::dataset::tokenizer::HuggingFaceTokenizer),
}

impl ChatTokenizer {
    fn count(&self, text: &str) -> Option<usize> {
        use aiperf_runtime::dataset::tokenizer::TextTokenizer;
        if text.is_empty() {
            return None;
        }
        let n = match self {
            ChatTokenizer::Builtin(t) => t.encode(text).map(|v| v.len()),
            ChatTokenizer::Hf(t) => t.encode(text).map(|v| v.len()),
        };
        n.ok()
    }
}

/// Load the tokenizer (`builtin` → tiktoken o200k; else an HF repo/dir, matching
/// `Tokenizer.from_pretrained(tokenizer or model)`).
async fn load_tokenizer(name: &str) -> anyhow::Result<ChatTokenizer> {
    use aiperf_runtime::dataset::tokenizer::{HuggingFaceTokenizer, TiktokenTokenizer};
    if name == "builtin" || name == "o200k_base" {
        return Ok(ChatTokenizer::Builtin(TiktokenTokenizer::builtin()));
    }
    // A local path is loaded directly; a bare repo id is downloaded then loaded.
    let path = std::path::Path::new(name);
    if path.is_dir() {
        return Ok(ChatTokenizer::Hf(HuggingFaceTokenizer::from_directory(
            path,
        )?));
    }
    if path.is_file() {
        return Ok(ChatTokenizer::Hf(HuggingFaceTokenizer::from_file(path)?));
    }
    let dir = aiperf_runtime::dataset::tokenizer::download_hugging_face_tokenizer(name)
        .await
        .with_context(|| format!("resolving tokenizer {name:?}"))?;
    Ok(ChatTokenizer::Hf(HuggingFaceTokenizer::from_directory(
        dir,
    )?))
}

/// Extract a usage integer from the first matching alias key (top-level or under
/// a `*_details` sub-object), mirroring the `Usage` model's alias set.
fn usage_int(usage: &Value, top: &[&str], nested: &[(&str, &str)]) -> Option<i64> {
    for k in top {
        if let Some(v) = usage.get(k).and_then(Value::as_i64) {
            return Some(v);
        }
    }
    for (obj, k) in nested {
        if let Some(v) = usage
            .get(obj)
            .and_then(|o| o.get(k))
            .and_then(Value::as_i64)
        {
            return Some(v);
        }
    }
    None
}

fn prompt_tokens(usage: &Value) -> Option<i64> {
    usage_int(
        usage,
        &["prompt_tokens", "input_tokens", "inputTokens"],
        &[],
    )
}

fn cache_read_tokens(usage: &Value) -> Option<i64> {
    usage_int(
        usage,
        &[
            "prompt_cache_read_tokens",
            "cache_read_input_tokens",
            "prompt_cache_hit_tokens",
            "cachedContentTokenCount",
        ],
        &[
            ("prompt_tokens_details", "cached_tokens"),
            ("input_tokens_details", "cached_tokens"),
        ],
    )
}

/// Render the per-turn stats block (`format_stats`).
fn format_stats(
    ttft_ns: Option<u128>,
    latency_ns: Option<u128>,
    osl: Option<usize>,
    reasoning_tokens: Option<usize>,
    isl: Option<i64>,
    cache_read: Option<i64>,
) -> String {
    let (Some(ttft_ns), Some(latency_ns), Some(osl)) = (ttft_ns, latency_ns, osl) else {
        return "(no tokens received)".to_string();
    };
    if osl == 0 {
        return "(no tokens received)".to_string();
    }
    let latency_s = latency_ns as f64 / NANOS_PER_SECOND;
    let tps = if latency_s > 0.0 {
        osl as f64 / latency_s
    } else {
        0.0
    };
    let mut tokens_desc = format!("{osl} tokens");
    if let Some(r) = reasoning_tokens.filter(|&r| r > 0) {
        tokens_desc += &format!(", {r} reasoning");
    }
    let mut lines = vec![
        format!("TTFT: {:.2} ms", ttft_ns as f64 / NANOS_PER_MILLIS),
        format!("TPS:  {tps:.2} tokens/s ({tokens_desc} in {latency_s:.2}s)"),
    ];
    // ITL = (latency − ttft)/(osl−1), defined for osl ≥ 2.
    if osl >= 2 {
        let itl_ns = (latency_ns - ttft_ns) as f64 / (osl - 1) as f64;
        if itl_ns > 0.0 {
            let decode_tps = NANOS_PER_SECOND / itl_ns;
            lines.push(format!(
                "ITL:  {:.2} ms/token (decode {decode_tps:.2} tokens/s)",
                itl_ns / NANOS_PER_MILLIS
            ));
        }
    }
    if let (Some(isl), Some(cache_read)) = (isl, cache_read) {
        if isl > 0 {
            let rate = 100.0 * cache_read as f64 / isl as f64;
            lines.push(format!(
                "Cache: {cache_read}/{isl} prompt tokens cached ({rate:.1}%)"
            ));
        }
    }
    lines.join("\n")
}

/// Build a tokio-rustls TLS connector with webpki roots (mirrors
/// `aiperf_runtime::export::otel` / `transport::http`).
fn tls_connector() -> anyhow::Result<tokio_rustls::TlsConnector> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(tokio_rustls::TlsConnector::from(Arc::new(config)))
}

/// Split a streaming `delta` into `(content, reasoning)` (`split_delta`).
fn split_delta(delta: &Value) -> (String, String) {
    let content = delta
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let reasoning = delta
        .get("reasoning_content")
        .or_else(|| delta.get("reasoning"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    (content, reasoning)
}

/// Stream one chat completion: POST, read SSE, print live, return
/// `(chunks, last_usage)`. Supports `http://` and `https://` (TLS via
/// tokio-rustls + webpki roots, mirroring the runner's transport).
async fn stream_turn(
    url: &str,
    headers: &[(String, String)],
    body: Vec<u8>,
    clock: &std::rc::Rc<dyn aiperf_runtime::clock::Clock>,
) -> anyhow::Result<(Vec<Chunk>, Option<Value>)> {
    let uri: hyper::Uri = url.parse().with_context(|| format!("bad url {url}"))?;
    let https = match uri.scheme_str() {
        Some("http") => false,
        Some("https") => true,
        other => anyhow::bail!("chat url must be http/https (got {other:?})"),
    };
    let host = uri.host().context("url has no host")?.to_string();
    let port = uri.port_u16().unwrap_or(if https { 443 } else { 80 });
    let authority = format!("{host}:{port}");

    // All timing flows through the INJECTED clock (never `Instant::now` /
    // `SystemTime::now`): `start_ns` is captured before the connect (matching
    // Python, whose `start_ns` precedes `session.post` opening the connection),
    // and SSE arrival timestamps come from the same clock via `read_sse`.
    let start_ns = clock.now_ns();

    let tcp = tokio::net::TcpStream::connect((host.as_str(), port))
        .await
        .with_context(|| format!("connect {authority}"))?;

    // http1 SendRequest is the same type regardless of the underlying IO, so the
    // request-send + SSE-read path is shared across the http / TLS branches.
    let mut sender = if https {
        let connector = tls_connector()?;
        let dnsname = rustls::pki_types::ServerName::try_from(host.clone())
            .with_context(|| format!("invalid TLS server name {host:?}"))?;
        let tls = connector
            .connect(dnsname, tcp)
            .await
            .context("TLS handshake")?;
        let (sender, conn) =
            hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(tls))
                .await
                .context("https handshake")?;
        tokio::spawn(async move {
            let _ = conn.await;
        });
        sender
    } else {
        let (sender, conn) =
            hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(tcp))
                .await
                .context("http handshake")?;
        tokio::spawn(async move {
            let _ = conn.await;
        });
        sender
    };

    let mut req = Request::builder()
        .method("POST")
        .uri(uri.path())
        .header("Host", &authority)
        .header("Content-Type", "application/json");
    for (k, v) in headers {
        req = req.header(k, v);
    }
    let req = req.body(http_body_util::Full::new(bytes::Bytes::from(body)))?;
    let resp = sender.send_request(req).await.context("send request")?;
    let status = resp.status();
    if !status.is_success() {
        // OpenAI-compatible servers put the real diagnostic in the body.
        let detail = resp
            .into_body()
            .collect()
            .await
            .map(|c| c.to_bytes())
            .unwrap_or_default();
        let text = String::from_utf8_lossy(&detail);
        anyhow::bail!(
            "HTTP {status}: {}",
            text.trim().chars().take(500).collect::<String>()
        );
    }

    // Reuse the runner's canonical SSE reader (`read_sse`, the behavioral port of
    // Python's `AsyncSSEStreamReader` that `chat` uses) rather than re-parsing SSE
    // by hand — it owns the multibyte-across-TCP-chunk + JSON-continuation edge
    // cases and timestamps each message off the injected clock.
    let body_stream = http_body_util::BodyStream::new(resp.into_body())
        .filter_map(|frame| async move {
            match frame {
                Ok(f) => f.into_data().ok().map(Ok),
                Err(e) => Some(Err(aiperf_runtime::transport::core::ErrorDetails::other(
                    format!("read body: {e}"),
                ))),
            }
        })
        .boxed_local();

    let mut chunks = Vec::new();
    let mut last_usage = None;
    let stdout = std::io::stdout();
    aiperf_runtime::transport::http::sse::read_sse(body_stream, clock.clone(), |msg| {
        if msg.is_done() {
            return;
        }
        let Some(data) = msg.data() else { return };
        let Ok(chunk): Result<Value, _> = serde_json::from_str(data) else {
            return;
        };
        if let Some(u) = chunk.get("usage").filter(|u| !u.is_null()) {
            last_usage = Some(u.clone());
        }
        let Some(delta) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
            .and_then(|c| c.get("delta"))
        else {
            return;
        };
        let (content, reasoning) = split_delta(delta);
        if content.is_empty() && reasoning.is_empty() {
            return;
        }
        let perf_ns = (msg.perf_ns - start_ns).max(0) as u128;
        {
            let mut h = stdout.lock();
            let _ = h.write_all(reasoning.as_bytes());
            let _ = h.write_all(content.as_bytes());
            let _ = h.flush();
        }
        chunks.push(Chunk {
            perf_ns,
            content,
            reasoning,
        });
    })
    .await
    .map_err(|e| anyhow::anyhow!("sse stream error: {}", e.message))?;

    Ok((chunks, last_usage))
}

/// Run one turn: stream, print live + stats, return the assistant text.
async fn run_turn(
    url: &str,
    headers: &[(String, String)],
    model: &str,
    conversation: &Value,
    tok: &ChatTokenizer,
    clock: &std::rc::Rc<dyn aiperf_runtime::clock::Clock>,
) -> anyhow::Result<String> {
    let payload = json!({
        "messages": conversation,
        "model": model,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let body = serde_json::to_vec(&payload)?;
    let (chunks, last_usage) = stream_turn(url, headers, body, clock).await?;
    println!();

    let output: String = chunks.iter().map(|c| c.content.as_str()).collect();
    let reasoning: String = chunks.iter().map(|c| c.reasoning.as_str()).collect();
    let osl = tok.count(&output);
    let reasoning_tokens = tok.count(&reasoning);

    let content_ts: Vec<u128> = chunks.iter().map(|c| c.perf_ns).collect();
    let ttft_ns = content_ts.first().copied();
    let latency_ns = content_ts.last().copied();
    let isl = last_usage.as_ref().and_then(prompt_tokens);
    let cache_read = last_usage.as_ref().and_then(cache_read_tokens);

    println!(
        "{}",
        format_stats(ttft_ns, latency_ns, osl, reasoning_tokens, isl, cache_read)
    );
    Ok(if !output.is_empty() {
        output
    } else {
        reasoning
    })
}

/// Run `aiperf chat [--model M] [--url U] [--quick MSG] [--no-history]
/// [--system-prompt S] [--api-key K] [--tokenizer T]`.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let mut model: Option<String> = None;
    let mut url = "http://localhost:8000".to_string();
    let mut system_prompt: Option<String> = None;
    let mut quick: Option<String> = None;
    let mut history = true;
    let mut api_key: Option<String> = None;
    let mut tokenizer: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut next = || {
            it.next()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("{a} needs a value"))
        };
        match a.as_str() {
            "--model" | "-m" => model = Some(next()?),
            "--url" | "-u" => url = next()?,
            "--system-prompt" => system_prompt = Some(next()?),
            "--quick" | "-q" => quick = Some(next()?),
            "--history" => history = true,
            "--no-history" => history = false,
            "--api-key" => api_key = Some(next()?),
            "--tokenizer" => tokenizer = Some(next()?),
            other => anyhow::bail!("unknown chat flag {other:?}"),
        }
    }
    let model = model.ok_or_else(|| anyhow::anyhow!("chat requires --model"))?;
    let url = chat_completions_url(&url);
    let mut headers: Vec<(String, String)> = Vec::new();
    if let Some(key) = api_key.or_else(|| std::env::var("OPENAI_API_KEY").ok()) {
        headers.push(("Authorization".into(), format!("Bearer {key}")));
    }

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?;

    rt.block_on(async move {
        // One injected clock for the whole session (the runner's RealClock; a
        // SimClock can be substituted). All chat timing flows through it.
        let clock: std::rc::Rc<dyn aiperf_runtime::clock::Clock> =
            aiperf_runtime::clock::RealClock::new();
        let tok = load_tokenizer(tokenizer.as_deref().unwrap_or(&model)).await?;
        let mut system_messages: Vec<Value> = Vec::new();
        if let Some(sp) = &system_prompt {
            system_messages.push(json!({"role": "system", "content": sp}));
        }

        if let Some(q) = quick {
            let conv = build_messages(&system_messages, &[], &q);
            run_turn(&url, &headers, &model, &conv, &tok, &clock).await?;
            return Ok(0);
        }

        let mut hist: Vec<Value> = Vec::new();
        println!("Please enter a message for the chat model (Ctrl-D to exit):");
        loop {
            print!("> ");
            std::io::stdout().flush().ok();
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line)? == 0 {
                println!();
                break;
            }
            let msg = line.trim_end_matches(['\n', '\r']).to_string();
            let conv = build_messages(&system_messages, &hist, &msg);
            match run_turn(&url, &headers, &model, &conv, &tok, &clock).await {
                Ok(assistant) => {
                    if history {
                        hist.push(json!({"role": "user", "content": msg}));
                        hist.push(json!({"role": "assistant", "content": assistant}));
                    }
                }
                Err(e) => eprintln!("request failed: {e}"),
            }
        }
        Ok::<i32, anyhow::Error>(0)
    })
}

/// Assemble one turn's messages: system, history, then the new user message.
fn build_messages(system: &[Value], history: &[Value], user: &str) -> Value {
    let mut msgs: Vec<Value> = system.to_vec();
    msgs.extend(history.iter().cloned());
    msgs.push(json!({"role": "user", "content": user}));
    Value::Array(msgs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_resolution_matches_python() {
        assert_eq!(
            chat_completions_url("localhost:8000"),
            "http://localhost:8000/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("http://h:8000/"),
            "http://h:8000/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("http://h:8000/v1"),
            "http://h:8000/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("http://h/openai/v1/chat/completions"),
            "http://h/openai/v1/chat/completions"
        );
    }

    #[test]
    fn stats_block_format_matches_python() {
        // TTFT 21.00 ms, latency 31.00 ms, osl 2 → ITL = (31−21)/1 = 10.00 ms.
        let s = format_stats(
            Some(21_000_000),
            Some(31_000_000),
            Some(2),
            None,
            Some(3),
            Some(0),
        );
        assert_eq!(
            s,
            "TTFT: 21.00 ms\n\
             TPS:  64.52 tokens/s (2 tokens in 0.03s)\n\
             ITL:  10.00 ms/token (decode 100.00 tokens/s)\n\
             Cache: 0/3 prompt tokens cached (0.0%)"
        );
    }

    #[test]
    fn stats_block_omits_itl_and_cache_when_absent() {
        // Single token → no ITL; no usage → no Cache line.
        let s = format_stats(
            Some(20_000_000),
            Some(20_000_000),
            Some(1),
            None,
            None,
            None,
        );
        assert_eq!(
            s,
            "TTFT: 20.00 ms\nTPS:  50.00 tokens/s (1 tokens in 0.02s)"
        );
    }

    #[test]
    fn stats_no_tokens_placeholder() {
        assert_eq!(
            format_stats(None, None, None, None, None, None),
            "(no tokens received)"
        );
    }

    #[test]
    fn usage_extraction_matches_aliases() {
        let u = json!({"prompt_tokens": 42, "prompt_tokens_details": {"cached_tokens": 7}});
        assert_eq!(prompt_tokens(&u), Some(42));
        assert_eq!(cache_read_tokens(&u), Some(7));
        // Anthropic-style aliases.
        let a = json!({"input_tokens": 10, "cache_read_input_tokens": 4});
        assert_eq!(prompt_tokens(&a), Some(10));
        assert_eq!(cache_read_tokens(&a), Some(4));
    }
}
