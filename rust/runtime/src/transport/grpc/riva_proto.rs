// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked-in Prost representation of the Riva speech and NLP wire messages.
//!
//! Field numbers are grounded in the complete schemas vendored by reference
//! commit `a391cfe27` under
//! `src/aiperf/transports/grpc/proto/riva/riva_{common,audio,asr,tts,nlp}.proto`.
//! Keeping generated-equivalent Rust checked in avoids a build-time `protoc`
//! dependency. Fields not consumed by the reference serializers remain valid
//! unknown protobuf fields and are intentionally skipped during decoding.

use prost::Message;

/// Riva request identifier wrapper.
#[derive(Clone, PartialEq, Message)]
pub struct RequestId {
    /// Correlated request ID.
    #[prost(string, tag = "1")]
    pub value: String,
}

/// Riva audio encoding values shared by ASR and TTS.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, prost::Enumeration)]
#[repr(i32)]
pub enum AudioEncoding {
    /// Encoding was not specified.
    Unspecified = 0,
    /// Uncompressed signed little-endian PCM.
    LinearPcm = 1,
    /// Free Lossless Audio Codec.
    Flac = 2,
    /// G.711 mu-law.
    Mulaw = 3,
    /// Ogg Opus.
    Oggopus = 4,
    /// G.711 A-law.
    Alaw = 20,
}

/// Riva ASR recognition configuration fields used by AIPerf.
#[derive(Clone, PartialEq, Message)]
pub struct RecognitionConfig {
    /// Input audio encoding.
    #[prost(enumeration = "AudioEncoding", tag = "1")]
    pub encoding: i32,
    /// Input sample rate in hertz.
    #[prost(int32, tag = "2")]
    pub sample_rate_hertz: i32,
    /// BCP-47 language code.
    #[prost(string, tag = "3")]
    pub language_code: String,
    /// Maximum returned hypotheses.
    #[prost(int32, tag = "4")]
    pub max_alternatives: i32,
    /// Whether Riva should add punctuation.
    #[prost(bool, tag = "11")]
    pub enable_automatic_punctuation: bool,
    /// Explicit Riva model name.
    #[prost(string, tag = "13")]
    pub model: String,
}

/// Unary ASR recognition request.
#[derive(Clone, PartialEq, Message)]
pub struct RecognizeRequest {
    /// Recognition policy.
    #[prost(message, optional, tag = "1")]
    pub config: Option<RecognitionConfig>,
    /// Complete input audio.
    #[prost(bytes = "vec", tag = "2")]
    pub audio: Vec<u8>,
    /// Optional request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Streaming ASR recognition policy.
#[derive(Clone, PartialEq, Message)]
pub struct StreamingRecognitionConfig {
    /// Recognition policy shared with unary ASR.
    #[prost(message, optional, tag = "1")]
    pub config: Option<RecognitionConfig>,
    /// Whether interim hypotheses are returned.
    #[prost(bool, tag = "2")]
    pub interim_results: bool,
}

/// One request message on the ASR bidirectional stream.
#[derive(Clone, PartialEq, Message)]
pub struct StreamingRecognizeRequest {
    /// Config-first or audio-chunk request body.
    #[prost(oneof = "streaming_recognize_request::StreamingRequest", tags = "1, 2")]
    pub streaming_request: Option<streaming_recognize_request::StreamingRequest>,
    /// Optional request identity on the initial config message.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Nested ASR bidirectional request variants.
pub mod streaming_recognize_request {
    use prost::Oneof;

    use super::StreamingRecognitionConfig;

    /// Exactly one config or audio body.
    #[derive(Clone, PartialEq, Oneof)]
    pub enum StreamingRequest {
        /// Initial stream configuration.
        #[prost(message, tag = "1")]
        StreamingConfig(StreamingRecognitionConfig),
        /// Subsequent raw audio bytes.
        #[prost(bytes, tag = "2")]
        AudioContent(Vec<u8>),
    }
}

/// One ASR recognition alternative.
#[derive(Clone, PartialEq, Message)]
pub struct SpeechRecognitionAlternative {
    /// Transcript text.
    #[prost(string, tag = "1")]
    pub transcript: String,
    /// Confidence score, when supplied.
    #[prost(float, tag = "2")]
    pub confidence: f32,
}

/// One unary ASR result segment.
#[derive(Clone, PartialEq, Message)]
pub struct SpeechRecognitionResult {
    /// Ordered hypotheses.
    #[prost(message, repeated, tag = "1")]
    pub alternatives: Vec<SpeechRecognitionAlternative>,
}

/// Unary ASR response.
#[derive(Clone, PartialEq, Message)]
pub struct RecognizeResponse {
    /// Sequential recognition segments.
    #[prost(message, repeated, tag = "1")]
    pub results: Vec<SpeechRecognitionResult>,
    /// Correlated request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// One streaming ASR result segment.
#[derive(Clone, PartialEq, Message)]
pub struct StreamingRecognitionResult {
    /// Ordered hypotheses.
    #[prost(message, repeated, tag = "1")]
    pub alternatives: Vec<SpeechRecognitionAlternative>,
    /// Whether this hypothesis is final.
    #[prost(bool, tag = "2")]
    pub is_final: bool,
    /// Interim-hypothesis stability.
    #[prost(float, tag = "3")]
    pub stability: f32,
}

/// One response message on the ASR bidirectional stream.
#[derive(Clone, PartialEq, Message)]
pub struct StreamingRecognizeResponse {
    /// Latest recognition segments.
    #[prost(message, repeated, tag = "1")]
    pub results: Vec<StreamingRecognitionResult>,
    /// Correlated request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Riva TTS synthesis request.
#[derive(Clone, PartialEq, Message)]
pub struct SynthesizeSpeechRequest {
    /// Text to synthesize.
    #[prost(string, tag = "1")]
    pub text: String,
    /// BCP-47 language code.
    #[prost(string, tag = "2")]
    pub language_code: String,
    /// Requested audio encoding.
    #[prost(enumeration = "AudioEncoding", tag = "3")]
    pub encoding: i32,
    /// Requested output sample rate.
    #[prost(int32, tag = "4")]
    pub sample_rate_hz: i32,
    /// Requested voice.
    #[prost(string, tag = "5")]
    pub voice_name: String,
    /// Optional request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Experimental Riva TTS response metadata.
#[derive(Clone, PartialEq, Message)]
pub struct SynthesizeSpeechResponseMetadata {
    /// Original input text.
    #[prost(string, tag = "1")]
    pub text: String,
    /// Normalized text used for synthesis.
    #[prost(string, tag = "2")]
    pub processed_text: String,
    /// Predicted token durations.
    #[prost(float, repeated, tag = "8")]
    pub predicted_durations: Vec<f32>,
}

/// Riva TTS unary or streaming response message.
#[derive(Clone, PartialEq, Message)]
pub struct SynthesizeSpeechResponse {
    /// Synthesized audio bytes.
    #[prost(bytes = "vec", tag = "1")]
    pub audio: Vec<u8>,
    /// Optional text preprocessing metadata.
    #[prost(message, optional, tag = "2")]
    pub meta: Option<SynthesizeSpeechResponseMetadata>,
    /// Correlated request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Riva NLP model selection parameters.
#[derive(Clone, PartialEq, Message)]
pub struct NlpModelParams {
    /// Explicit model name.
    #[prost(string, tag = "1")]
    pub model_name: String,
    /// BCP-47 language code.
    #[prost(string, tag = "3")]
    pub language_code: String,
}

/// Text-classification request.
#[derive(Clone, PartialEq, Message)]
pub struct TextClassRequest {
    /// Independent text inputs.
    #[prost(string, repeated, tag = "1")]
    pub text: Vec<String>,
    /// Requested result count.
    #[prost(uint32, tag = "2")]
    pub top_n: u32,
    /// Model selection.
    #[prost(message, optional, tag = "3")]
    pub model: Option<NlpModelParams>,
    /// Optional request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// One classification label.
#[derive(Clone, PartialEq, Message)]
pub struct Classification {
    /// Class label.
    #[prost(string, tag = "1")]
    pub class_name: String,
    /// Confidence score.
    #[prost(float, tag = "2")]
    pub score: f32,
}

/// Labels for one classified text.
#[derive(Clone, PartialEq, Message)]
pub struct ClassificationResult {
    /// Ordered labels.
    #[prost(message, repeated, tag = "1")]
    pub labels: Vec<Classification>,
}

/// Text-classification response.
#[derive(Clone, PartialEq, Message)]
pub struct TextClassResponse {
    /// One result per input text.
    #[prost(message, repeated, tag = "1")]
    pub results: Vec<ClassificationResult>,
    /// Correlated request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Token-classification request.
#[derive(Clone, PartialEq, Message)]
pub struct TokenClassRequest {
    /// Independent text inputs.
    #[prost(string, repeated, tag = "1")]
    pub text: Vec<String>,
    /// Requested result count.
    #[prost(uint32, tag = "3")]
    pub top_n: u32,
    /// Model selection.
    #[prost(message, optional, tag = "4")]
    pub model: Option<NlpModelParams>,
    /// Optional request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Source span for a token classification.
#[derive(Clone, Copy, PartialEq, Message)]
pub struct Span {
    /// Inclusive start offset.
    #[prost(uint32, tag = "1")]
    pub start: u32,
    /// Exclusive end offset.
    #[prost(uint32, tag = "2")]
    pub end: u32,
}

/// One token and its labels.
#[derive(Clone, PartialEq, Message)]
pub struct TokenClassValue {
    /// Token text.
    #[prost(string, tag = "1")]
    pub token: String,
    /// Ordered labels.
    #[prost(message, repeated, tag = "2")]
    pub label: Vec<Classification>,
    /// Source spans.
    #[prost(message, repeated, tag = "3")]
    pub span: Vec<Span>,
}

/// Token classifications for one input sequence.
#[derive(Clone, PartialEq, Message)]
pub struct TokenClassSequence {
    /// Per-token classifications.
    #[prost(message, repeated, tag = "1")]
    pub results: Vec<TokenClassValue>,
}

/// Token-classification response.
#[derive(Clone, PartialEq, Message)]
pub struct TokenClassResponse {
    /// One sequence per input text.
    #[prost(message, repeated, tag = "1")]
    pub results: Vec<TokenClassSequence>,
    /// Correlated request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Text transformation or punctuation request.
#[derive(Clone, PartialEq, Message)]
pub struct TextTransformRequest {
    /// Independent text inputs.
    #[prost(string, repeated, tag = "1")]
    pub text: Vec<String>,
    /// Requested result count.
    #[prost(uint32, tag = "2")]
    pub top_n: u32,
    /// Model selection.
    #[prost(message, optional, tag = "3")]
    pub model: Option<NlpModelParams>,
    /// Optional request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Text transformation or punctuation response.
#[derive(Clone, PartialEq, Message)]
pub struct TextTransformResponse {
    /// Transformed texts in request order.
    #[prost(string, repeated, tag = "1")]
    pub text: Vec<String>,
    /// Correlated request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Riva natural-query request.
#[derive(Clone, PartialEq, Message)]
pub struct NaturalQueryRequest {
    /// Natural-language query.
    #[prost(string, tag = "1")]
    pub query: String,
    /// Maximum result count.
    #[prost(uint32, tag = "2")]
    pub top_n: u32,
    /// Context document.
    #[prost(string, tag = "3")]
    pub context: String,
    /// Optional request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// One natural-query answer.
#[derive(Clone, PartialEq, Message)]
pub struct NaturalQueryResult {
    /// Answer text.
    #[prost(string, tag = "1")]
    pub answer: String,
    /// Confidence score.
    #[prost(float, tag = "2")]
    pub score: f32,
}

/// Natural-query response.
#[derive(Clone, PartialEq, Message)]
pub struct NaturalQueryResponse {
    /// Ranked answers.
    #[prost(message, repeated, tag = "1")]
    pub results: Vec<NaturalQueryResult>,
    /// Correlated request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Riva intent-analysis options used by AIPerf.
#[derive(Clone, PartialEq, Message)]
pub struct AnalyzeIntentOptions {
    /// Optional fixed intent domain.
    #[prost(string, tag = "3")]
    pub domain: String,
    /// Optional language code.
    #[prost(string, tag = "4")]
    pub lang: String,
}

/// Intent-analysis request.
#[derive(Clone, PartialEq, Message)]
pub struct AnalyzeIntentRequest {
    /// Text to analyze.
    #[prost(string, tag = "1")]
    pub query: String,
    /// Optional analysis policy.
    #[prost(message, optional, tag = "2")]
    pub options: Option<AnalyzeIntentOptions>,
    /// Optional request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Intent-analysis response.
#[derive(Clone, PartialEq, Message)]
pub struct AnalyzeIntentResponse {
    /// Selected intent.
    #[prost(message, optional, tag = "1")]
    pub intent: Option<Classification>,
    /// Extracted slots.
    #[prost(message, repeated, tag = "2")]
    pub slots: Vec<TokenClassValue>,
    /// Legacy inferred domain text.
    #[prost(string, tag = "3")]
    pub domain_str: String,
    /// Inferred domain classification.
    #[prost(message, optional, tag = "4")]
    pub domain: Option<Classification>,
    /// Correlated request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Riva entity-analysis request.
#[derive(Clone, PartialEq, Message)]
pub struct AnalyzeEntitiesRequest {
    /// Text to analyze.
    #[prost(string, tag = "1")]
    pub query: String,
    /// Optional language policy; unused by the reference endpoint.
    #[prost(message, optional, tag = "2")]
    pub options: Option<AnalyzeEntitiesOptions>,
    /// Optional request identity.
    #[prost(message, optional, tag = "100")]
    pub id: Option<RequestId>,
}

/// Riva entity-analysis options.
#[derive(Clone, PartialEq, Message)]
pub struct AnalyzeEntitiesOptions {
    /// Optional language code.
    #[prost(string, tag = "4")]
    pub lang: String,
}
