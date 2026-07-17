// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NVIDIA Riva ASR / TTS / NLP gRPC target.
//!
//! Serves the Riva speech and language services AIPerf's native gRPC client
//! dials (`aiperf_runtime::endpoints::riva` and
//! `aiperf_runtime::transport::grpc::riva_binding`). It routes by gRPC method
//! path over the shared hyper h2c stack and reuses `crate::grpc::ProstCodec`.
//! The wire
//! contract is guaranteed by construction: every request/response message is the
//! exact prost struct the runner's Riva codec
//! (`aiperf_runtime::transport::grpc::riva_proto`) encodes/decodes, so there is no second
//! schema to drift, and the field numbers match by sharing the same types.
//!
//! The service returns deterministic
//! content (a canned transcript, deterministic PCM audio bytes, canned
//! classification/answer results) so an e2e run's raw records are exactly
//! predictable. The public [`RIVA_ASR_TRANSCRIPT`] / [`RIVA_NATURAL_QUERY_ANSWER`]
//! constants define the expected response content.
//!
//! Supported methods:
//!   * ASR `Recognize` (unary) + `StreamingRecognize` (bidirectional streaming)
//!   * TTS `Synthesize` (unary) + `SynthesizeOnline` (server streaming)
//!   * NLP `ClassifyText` / `ClassifyTokens` / `TransformText` / `PunctuateText`
//!     / `NaturalQuery` / `AnalyzeIntent` / `AnalyzeEntities` (all unary)

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures::Stream;
use tonic::body::Body;
use tonic::codec::Streaming;
use tonic::server::Grpc;
use tonic::{Request, Response, Status};

use aiperf_runtime::transport::grpc::riva_proto::{
    AnalyzeEntitiesRequest, AnalyzeIntentRequest, AnalyzeIntentResponse, Classification,
    ClassificationResult, NaturalQueryRequest, NaturalQueryResponse, NaturalQueryResult,
    RecognizeRequest, RecognizeResponse, SpeechRecognitionAlternative, SpeechRecognitionResult,
    StreamingRecognitionResult, StreamingRecognizeRequest, StreamingRecognizeResponse,
    SynthesizeSpeechRequest, SynthesizeSpeechResponse, TextClassRequest, TextClassResponse,
    TextTransformRequest, TextTransformResponse, TokenClassRequest, TokenClassResponse,
    TokenClassSequence, TokenClassValue,
};

use crate::grpc::ProstCodec;
use crate::state::AppState;

/// Riva ASR service method paths (must byte-match
/// `aiperf_runtime::transport::grpc::riva_binding`).
const ASR_RECOGNIZE: &str = "/nvidia.riva.asr.RivaSpeechRecognition/Recognize";
const ASR_STREAMING_RECOGNIZE: &str = "/nvidia.riva.asr.RivaSpeechRecognition/StreamingRecognize";
/// Riva TTS service method paths.
const TTS_SYNTHESIZE: &str = "/nvidia.riva.tts.RivaSpeechSynthesis/Synthesize";
const TTS_SYNTHESIZE_ONLINE: &str = "/nvidia.riva.tts.RivaSpeechSynthesis/SynthesizeOnline";
/// Riva NLP service method paths.
const NLP_CLASSIFY_TEXT: &str = "/nvidia.riva.nlp.RivaLanguageUnderstanding/ClassifyText";
const NLP_CLASSIFY_TOKENS: &str = "/nvidia.riva.nlp.RivaLanguageUnderstanding/ClassifyTokens";
const NLP_TRANSFORM_TEXT: &str = "/nvidia.riva.nlp.RivaLanguageUnderstanding/TransformText";
const NLP_PUNCTUATE_TEXT: &str = "/nvidia.riva.nlp.RivaLanguageUnderstanding/PunctuateText";
const NLP_NATURAL_QUERY: &str = "/nvidia.riva.nlp.RivaLanguageUnderstanding/NaturalQuery";
const NLP_ANALYZE_INTENT: &str = "/nvidia.riva.nlp.RivaLanguageUnderstanding/AnalyzeIntent";
const NLP_ANALYZE_ENTITIES: &str = "/nvidia.riva.nlp.RivaLanguageUnderstanding/AnalyzeEntities";

const RIVA_ENDPOINT: &str = "riva";
/// Riva selects models per request configuration, but mock behavior is fixed.
const RIVA_MODEL: &str = "mock-riva";

/// Transcript returned by every ASR RPC.
pub const RIVA_ASR_TRANSCRIPT: &str = "the quick brown fox jumps over the lazy dog";
/// Answer returned by every NaturalQuery RPC.
pub const RIVA_NATURAL_QUERY_ANSWER: &str = "the mock riva answer is forty two";
/// The deterministic intent class every AnalyzeIntent RPC returns.
pub const RIVA_INTENT_CLASS: &str = "mock_greeting";
/// The deterministic sentiment label every ClassifyText RPC returns.
pub const RIVA_SENTIMENT_CLASS: &str = "positive";

type RivaStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

/// True when `path` names a Riva service method this module serves.
///
/// Used by [`crate::grpc::route`] to hand a request to [`route_riva`] before the
/// KServe match; the two dialects' method paths are disjoint
/// (`/nvidia.riva.*` vs `/inference.*`).
pub fn is_riva_path(path: &str) -> bool {
    matches!(
        path,
        ASR_RECOGNIZE
            | ASR_STREAMING_RECOGNIZE
            | TTS_SYNTHESIZE
            | TTS_SYNTHESIZE_ONLINE
            | NLP_CLASSIFY_TEXT
            | NLP_CLASSIFY_TOKENS
            | NLP_TRANSFORM_TEXT
            | NLP_PUNCTUATE_TEXT
            | NLP_NATURAL_QUERY
            | NLP_ANALYZE_INTENT
            | NLP_ANALYZE_ENTITIES
    )
}

/// Route one Riva gRPC request to its handler by method path. The caller has
/// already confirmed `path` is a Riva method via [`is_riva_path`]; an unknown
/// path here still yields a gRPC `Unimplemented` response for defensiveness.
pub async fn route_riva(
    path: &str,
    state: Arc<AppState>,
    req: http::Request<hyper::body::Incoming>,
) -> http::Response<Body> {
    match path {
        ASR_RECOGNIZE => {
            let service = tower::service_fn(move |r: Request<RecognizeRequest>| {
                let state = state.clone();
                async move { asr_recognize(state, r).await }
            });
            Grpc::new(ProstCodec::<RecognizeRequest, RecognizeResponse>::default())
                .unary(service, req)
                .await
        }
        ASR_STREAMING_RECOGNIZE => {
            let service =
                tower::service_fn(move |r: Request<Streaming<StreamingRecognizeRequest>>| {
                    let state = state.clone();
                    async move { asr_streaming(state, r).await }
                });
            Grpc::new(ProstCodec::<
                StreamingRecognizeRequest,
                StreamingRecognizeResponse,
            >::default())
            .streaming(service, req)
            .await
        }
        TTS_SYNTHESIZE => {
            let service = tower::service_fn(move |r: Request<SynthesizeSpeechRequest>| {
                let state = state.clone();
                async move { tts_synthesize(state, r).await }
            });
            Grpc::new(ProstCodec::<
                SynthesizeSpeechRequest,
                SynthesizeSpeechResponse,
            >::default())
            .unary(service, req)
            .await
        }
        TTS_SYNTHESIZE_ONLINE => {
            let service = tower::service_fn(move |r: Request<SynthesizeSpeechRequest>| {
                let state = state.clone();
                async move { tts_synthesize_online(state, r).await }
            });
            Grpc::new(ProstCodec::<
                SynthesizeSpeechRequest,
                SynthesizeSpeechResponse,
            >::default())
            .server_streaming(service, req)
            .await
        }
        NLP_CLASSIFY_TEXT => {
            let service = tower::service_fn(move |r: Request<TextClassRequest>| {
                let state = state.clone();
                async move { classify_text(state, r).await }
            });
            Grpc::new(ProstCodec::<TextClassRequest, TextClassResponse>::default())
                .unary(service, req)
                .await
        }
        NLP_CLASSIFY_TOKENS => {
            let service = tower::service_fn(move |r: Request<TokenClassRequest>| {
                let state = state.clone();
                async move { classify_tokens(state, r).await }
            });
            Grpc::new(ProstCodec::<TokenClassRequest, TokenClassResponse>::default())
                .unary(service, req)
                .await
        }
        NLP_TRANSFORM_TEXT | NLP_PUNCTUATE_TEXT => {
            let service = tower::service_fn(move |r: Request<TextTransformRequest>| {
                let state = state.clone();
                async move { transform_text(state, r).await }
            });
            Grpc::new(ProstCodec::<TextTransformRequest, TextTransformResponse>::default())
                .unary(service, req)
                .await
        }
        NLP_NATURAL_QUERY => {
            let service = tower::service_fn(move |r: Request<NaturalQueryRequest>| {
                let state = state.clone();
                async move { natural_query(state, r).await }
            });
            Grpc::new(ProstCodec::<NaturalQueryRequest, NaturalQueryResponse>::default())
                .unary(service, req)
                .await
        }
        NLP_ANALYZE_INTENT => {
            let service = tower::service_fn(move |r: Request<AnalyzeIntentRequest>| {
                let state = state.clone();
                async move { analyze_intent(state, r).await }
            });
            Grpc::new(ProstCodec::<AnalyzeIntentRequest, AnalyzeIntentResponse>::default())
                .unary(service, req)
                .await
        }
        NLP_ANALYZE_ENTITIES => {
            let service = tower::service_fn(move |r: Request<AnalyzeEntitiesRequest>| {
                let state = state.clone();
                async move { analyze_entities(state, r).await }
            });
            Grpc::new(ProstCodec::<AnalyzeEntitiesRequest, TokenClassResponse>::default())
                .unary(service, req)
                .await
        }
        other => Status::unimplemented(format!("unknown Riva gRPC method: {other}")).into_http(),
    }
}

fn record_unary<T>(state: &AppState, start: Instant, body: T) -> Result<Response<T>, Status> {
    state
        .recorder
        .record_request_start(RIVA_ENDPOINT, RIVA_MODEL);
    state
        .recorder
        .record_basic_success(RIVA_ENDPOINT, start.elapsed().as_secs_f64());
    state.recorder.record_request_end(RIVA_ENDPOINT);
    Ok(Response::new(body))
}

fn asr_result() -> SpeechRecognitionResult {
    SpeechRecognitionResult {
        alternatives: vec![SpeechRecognitionAlternative {
            transcript: RIVA_ASR_TRANSCRIPT.to_string(),
            confidence: 0.99,
        }],
    }
}

async fn asr_recognize(
    state: Arc<AppState>,
    _request: Request<RecognizeRequest>,
) -> Result<Response<RecognizeResponse>, Status> {
    if state.inject_error() {
        return Err(Status::internal("Simulated error"));
    }
    let start = Instant::now();
    record_unary(
        &state,
        start,
        RecognizeResponse {
            results: vec![asr_result()],
            id: None,
        },
    )
}

/// `StreamingRecognize` (bidirectional): drain the inbound config + audio-chunk
/// stream, then emit one interim and one final transcript. The interim/final
/// pair exercises the runner's streaming `is_final` reconciliation; both carry
/// the same deterministic transcript.
async fn asr_streaming(
    state: Arc<AppState>,
    request: Request<Streaming<StreamingRecognizeRequest>>,
) -> Result<Response<RivaStream<StreamingRecognizeResponse>>, Status> {
    if state.inject_error() {
        return Err(Status::internal("Simulated error"));
    }
    let start = Instant::now();
    let mut inbound = request.into_inner();
    let stream = async_stream::stream! {
        // Consume every client message (initial config + audio chunks) so the
        // client's half-close resolves before responses flow. A decode error on
        // the inbound side terminates the RPC with that status.
        loop {
            match inbound.message().await {
                Ok(Some(_)) => continue,
                Ok(None) => break,
                Err(status) => {
                    yield Err(status);
                    return;
                }
            }
        }
        yield Ok(StreamingRecognizeResponse {
            results: vec![StreamingRecognitionResult {
                alternatives: vec![SpeechRecognitionAlternative {
                    transcript: RIVA_ASR_TRANSCRIPT.to_string(),
                    confidence: 0.5,
                }],
                is_final: false,
                stability: 0.1,
            }],
            id: None,
        });
        yield Ok(StreamingRecognizeResponse {
            results: vec![StreamingRecognitionResult {
                alternatives: vec![SpeechRecognitionAlternative {
                    transcript: RIVA_ASR_TRANSCRIPT.to_string(),
                    confidence: 0.99,
                }],
                is_final: true,
                stability: 1.0,
            }],
            id: None,
        });
        state.recorder.record_request_start(RIVA_ENDPOINT, RIVA_MODEL);
        state
            .recorder
            .record_basic_success(RIVA_ENDPOINT, start.elapsed().as_secs_f64());
        state.recorder.record_request_end(RIVA_ENDPOINT);
    };
    Ok(Response::new(Box::pin(stream)))
}

/// Deterministic 16-bit PCM audio for a TTS reply: `n` samples of a fixed ramp,
/// sized off the requested sample rate so the runner's PCM duration derivation
/// is sensible. Never empty (the runner drops empty audio as no response).
fn synth_audio(sample_rate_hz: i32) -> Vec<u8> {
    let rate = if sample_rate_hz > 0 {
        sample_rate_hz as usize
    } else {
        22_050
    };
    let samples = (rate / 10).max(1);
    let mut audio = Vec::with_capacity(samples * 2);
    for index in 0..samples {
        let value = (index % 256) as u16;
        audio.extend_from_slice(&value.to_le_bytes());
    }
    audio
}

/// `Synthesize` (unary): return deterministic PCM audio bytes for the text.
async fn tts_synthesize(
    state: Arc<AppState>,
    request: Request<SynthesizeSpeechRequest>,
) -> Result<Response<SynthesizeSpeechResponse>, Status> {
    if state.inject_error() {
        return Err(Status::internal("Simulated error"));
    }
    let start = Instant::now();
    let msg = request.into_inner();
    record_unary(
        &state,
        start,
        SynthesizeSpeechResponse {
            audio: synth_audio(msg.sample_rate_hz),
            meta: None,
            id: None,
        },
    )
}

/// `SynthesizeOnline` (server streaming): emit the same deterministic audio in a
/// few chunks so the runner measures streaming TTS timing. The concatenation of
/// the chunks equals the unary audio for the same request.
async fn tts_synthesize_online(
    state: Arc<AppState>,
    request: Request<SynthesizeSpeechRequest>,
) -> Result<Response<RivaStream<SynthesizeSpeechResponse>>, Status> {
    if state.inject_error() {
        return Err(Status::internal("Simulated error"));
    }
    let start = Instant::now();
    let msg = request.into_inner();
    let audio = synth_audio(msg.sample_rate_hz);
    let chunk_len = audio.len().div_ceil(4).max(1);
    let stream = async_stream::stream! {
        for chunk in audio.chunks(chunk_len) {
            yield Ok(SynthesizeSpeechResponse {
                audio: chunk.to_vec(),
                meta: None,
                id: None,
            });
        }
        state.recorder.record_request_start(RIVA_ENDPOINT, RIVA_MODEL);
        state
            .recorder
            .record_basic_success(RIVA_ENDPOINT, start.elapsed().as_secs_f64());
        state.recorder.record_request_end(RIVA_ENDPOINT);
    };
    Ok(Response::new(Box::pin(stream)))
}

/// `ClassifyText` (unary): one sentiment result per input text.
async fn classify_text(
    state: Arc<AppState>,
    request: Request<TextClassRequest>,
) -> Result<Response<TextClassResponse>, Status> {
    if state.inject_error() {
        return Err(Status::internal("Simulated error"));
    }
    let start = Instant::now();
    let msg = request.into_inner();
    let results = msg
        .text
        .iter()
        .map(|_| ClassificationResult {
            labels: vec![
                Classification {
                    class_name: RIVA_SENTIMENT_CLASS.to_string(),
                    score: 0.98,
                },
                Classification {
                    class_name: "negative".to_string(),
                    score: 0.02,
                },
            ],
        })
        .collect();
    record_unary(&state, start, TextClassResponse { results, id: None })
}

/// A one-token classification sequence per input text (shared by `ClassifyTokens`
/// and `AnalyzeEntities`, which the runner decodes identically).
fn token_class_results(count: usize) -> Vec<TokenClassSequence> {
    (0..count)
        .map(|_| TokenClassSequence {
            results: vec![TokenClassValue {
                token: "mock".to_string(),
                label: vec![Classification {
                    class_name: "O".to_string(),
                    score: 0.99,
                }],
                span: Vec::new(),
            }],
        })
        .collect()
}

/// `ClassifyTokens` (unary): one labeled-token sequence per input text.
async fn classify_tokens(
    state: Arc<AppState>,
    request: Request<TokenClassRequest>,
) -> Result<Response<TokenClassResponse>, Status> {
    if state.inject_error() {
        return Err(Status::internal("Simulated error"));
    }
    let start = Instant::now();
    let msg = request.into_inner();
    record_unary(
        &state,
        start,
        TokenClassResponse {
            results: token_class_results(msg.text.len().max(1)),
            id: None,
        },
    )
}

/// `AnalyzeEntities` (unary): the runner decodes this exactly like
/// `ClassifyTokens`, so it returns the same token-classification shape for the
/// single `query`.
async fn analyze_entities(
    state: Arc<AppState>,
    request: Request<AnalyzeEntitiesRequest>,
) -> Result<Response<TokenClassResponse>, Status> {
    if state.inject_error() {
        return Err(Status::internal("Simulated error"));
    }
    let start = Instant::now();
    let _msg = request.into_inner();
    record_unary(
        &state,
        start,
        TokenClassResponse {
            results: token_class_results(1),
            id: None,
        },
    )
}

/// `TransformText` / `PunctuateText` (unary): return each input text uppercased.
/// Deterministic and input-derived, so the runner's `texts` reconciliation sees
/// one transformed string per input.
async fn transform_text(
    state: Arc<AppState>,
    request: Request<TextTransformRequest>,
) -> Result<Response<TextTransformResponse>, Status> {
    if state.inject_error() {
        return Err(Status::internal("Simulated error"));
    }
    let start = Instant::now();
    let msg = request.into_inner();
    let text = if msg.text.is_empty() {
        vec![RIVA_ASR_TRANSCRIPT.to_uppercase()]
    } else {
        msg.text.iter().map(|value| value.to_uppercase()).collect()
    };
    record_unary(&state, start, TextTransformResponse { text, id: None })
}

/// `NaturalQuery` (unary): return the deterministic canned answer.
async fn natural_query(
    state: Arc<AppState>,
    request: Request<NaturalQueryRequest>,
) -> Result<Response<NaturalQueryResponse>, Status> {
    if state.inject_error() {
        return Err(Status::internal("Simulated error"));
    }
    let start = Instant::now();
    let _msg = request.into_inner();
    record_unary(
        &state,
        start,
        NaturalQueryResponse {
            results: vec![NaturalQueryResult {
                answer: RIVA_NATURAL_QUERY_ANSWER.to_string(),
                score: 0.87,
            }],
            id: None,
        },
    )
}

/// `AnalyzeIntent` (unary): return a deterministic intent classification with no
/// slots.
async fn analyze_intent(
    state: Arc<AppState>,
    request: Request<AnalyzeIntentRequest>,
) -> Result<Response<AnalyzeIntentResponse>, Status> {
    if state.inject_error() {
        return Err(Status::internal("Simulated error"));
    }
    let start = Instant::now();
    let _msg = request.into_inner();
    record_unary(
        &state,
        start,
        AnalyzeIntentResponse {
            intent: Some(Classification {
                class_name: RIVA_INTENT_CLASS.to_string(),
                score: 0.91,
            }),
            slots: Vec::new(),
            domain_str: String::new(),
            domain: None,
            id: None,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn fast_state() -> Arc<AppState> {
        let config = crate::config::MockServerConfig {
            fast: true,
            no_tokenizer: true,
            ..crate::config::MockServerConfig::default()
        }
        .apply_flags();
        AppState::build(config)
    }

    #[test]
    fn riva_paths_are_recognized_and_others_are_not() {
        assert!(is_riva_path(ASR_RECOGNIZE));
        assert!(is_riva_path(NLP_NATURAL_QUERY));
        assert!(is_riva_path(TTS_SYNTHESIZE_ONLINE));
        assert!(!is_riva_path("/inference.GRPCInferenceService/ModelInfer"));
        assert!(!is_riva_path(
            "/nvidia.riva.asr.RivaSpeechRecognition/Unknown"
        ));
    }

    #[tokio::test]
    async fn asr_recognize_returns_canned_transcript() {
        let response = asr_recognize(fast_state(), Request::new(RecognizeRequest::default()))
            .await
            .expect("asr ok")
            .into_inner();
        let transcript = &response.results[0].alternatives[0].transcript;
        assert_eq!(transcript, RIVA_ASR_TRANSCRIPT);
        let encoded = response.encode_to_vec();
        let decoded = RecognizeResponse::decode(encoded.as_slice()).expect("decode");
        assert_eq!(
            decoded.results[0].alternatives[0].transcript,
            RIVA_ASR_TRANSCRIPT
        );
    }

    #[tokio::test]
    async fn tts_synthesize_returns_nonempty_audio() {
        let request = SynthesizeSpeechRequest {
            text: "hello".to_string(),
            sample_rate_hz: 16_000,
            ..SynthesizeSpeechRequest::default()
        };
        let response = tts_synthesize(fast_state(), Request::new(request))
            .await
            .expect("tts ok")
            .into_inner();
        assert_eq!(response.audio.len(), 3200);
    }

    #[tokio::test]
    async fn natural_query_returns_canned_answer() {
        let response = natural_query(fast_state(), Request::new(NaturalQueryRequest::default()))
            .await
            .expect("nq ok")
            .into_inner();
        assert_eq!(response.results[0].answer, RIVA_NATURAL_QUERY_ANSWER);
    }

    #[tokio::test]
    async fn classify_text_returns_one_result_per_input() {
        let request = TextClassRequest {
            text: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            ..TextClassRequest::default()
        };
        let response = classify_text(fast_state(), Request::new(request))
            .await
            .expect("classify ok")
            .into_inner();
        assert_eq!(response.results.len(), 3);
        assert_eq!(
            response.results[0].labels[0].class_name,
            RIVA_SENTIMENT_CLASS
        );
    }

    #[tokio::test]
    async fn transform_text_uppercases_each_input() {
        let request = TextTransformRequest {
            text: vec!["hello world".to_string()],
            ..TextTransformRequest::default()
        };
        let response = transform_text(fast_state(), Request::new(request))
            .await
            .expect("transform ok")
            .into_inner();
        assert_eq!(response.text, vec!["HELLO WORLD".to_string()]);
    }

    #[tokio::test]
    async fn analyze_intent_returns_intent() {
        let response = analyze_intent(fast_state(), Request::new(AnalyzeIntentRequest::default()))
            .await
            .expect("intent ok")
            .into_inner();
        assert_eq!(
            response.intent.expect("intent present").class_name,
            RIVA_INTENT_CLASS
        );
    }
}
