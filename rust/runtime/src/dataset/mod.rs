// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified dataset substrate for every AIPerf execution mode.
//!
//! The crate owns the linear `load -> compose -> store -> sample -> materialize`
//! pipeline. Text, preformatted messages, raw request bodies, tools, headers, and
//! multimodal content are interned once in a prefix-dependent content-addressed
//! [`SegmentPool`]. [`Conversation`] and [`Turn`] contain only dense [`Handle`]s,
//! so sharing a [`Dataset`] across worker threads shares the payload bytes too.

pub mod analysis;
pub(crate) mod coding;
pub mod compose;
pub mod corpus;
pub mod error;
pub mod fetch;
pub mod generator;
pub mod hf_hub;
pub mod materialize;
pub mod model;
pub mod prompt;
pub mod request;
pub mod runtime_dataset;
pub mod sampler;
pub mod segment;
pub mod synthesis;
pub mod tokenizer;

pub use crate::body_plan::{
    BodyPlan, FieldName, FieldProgram, FieldValue, JsonBodyMaterializer, LiteralValue,
};
pub use corpus::{
    MAX_CHARS_PER_CHUNK, SHAKESPEARE_CORPUS, tokenize_corpus_chunked, tokenize_sonnet_corpus,
};
pub use error::{DatasetError, Result};
pub use fetch::{DatasetFetcher, HttpDatasetFetcher};
pub use generator::{
    GeneratedMedia, InlineSyntheticMediaPublisher, NativeAudioGenerator, NativeImageGenerator,
    NativeSyntheticMediaGeneratorFactory, NativeVideoGenerator, SourceImageSampling,
    SyntheticAudioConfig, SyntheticAudioFormat, SyntheticDatasetConfig, SyntheticImageConfig,
    SyntheticImageFormat, SyntheticImageSource, SyntheticMediaFormat, SyntheticMediaGenerator,
    SyntheticMediaGeneratorFactory, SyntheticMediaPublisher, SyntheticPrefixConfig,
    SyntheticPromptConfig, SyntheticRankingsConfig, SyntheticVideoAudioConfig,
    SyntheticVideoConfig, SyntheticVideoFormat, SyntheticVideoPattern, audio_duration_seconds,
    transcode_audio_to_wav,
};
pub use hf_hub::download_hugging_face_tokenizer;
pub use loader::{
    AccuracyComposer, AccuracyDatasetLoader, DatasetFormatRegistration, DatasetLoader,
    DatasetProbe, DatasetSource, LoadConfig, LoaderRegistry, RawRow, RowOrigin, load_raw_rows,
};
pub use materialize::{
    AssemblyItem, MessageSpliceResolver, Overrides, SegmentItemsMaterializer,
    build_message_body_from_wire_parts, build_message_body_from_wires,
};
pub use media::{InlineMediaResolver, MediaResolver, PrefetchMediaResolver};
pub use model::{
    AccuracyAssociation, BranchId, ContentGroup, Conversation, ConversationBranch,
    ConversationBranchMode, ConversationContextMode, ConversationMetadata, CorrelationId,
    DagMetadata, DispatchTiming, MediaKind, ModelId, NodeId, PrerequisiteKind, SessionId, Turn,
    TurnMetadata, TurnPrerequisite,
};
pub use prompt::{
    CorpusPromptGeneratorFactory, GeneratedPrompt, PreparedCorpusPromptGeneratorFactory,
    PromptGenerator, PromptGeneratorFactory,
};
pub use request::{
    BuiltinEndpointResolver, ConversationSession, EndpointRequestMaterializer, EndpointResolver,
    MaterializedRequest, RequestMaterializer, TraceHashAwareRequestMaterializer,
    WsRequestMaterializer,
};
pub use runtime_dataset as dataset;
pub use runtime_dataset::{Dataset, DatasetMetadata, TurnEndpointLookup};
pub use sampler::{
    RandomSampler, RandomSamplerFactory, Sampler, SamplerFactory, SamplerRegistry,
    SequentialSampler, SequentialSamplerFactory, ShuffleSampler, ShuffleSamplerFactory,
};
pub use segment::{
    Handle, InMemorySegmentStore, Payload, Role, Segment, SegmentId, SegmentPool, SegmentStore,
};
pub use synthesis::{
    PrefixTraceSynthesizer, TraceSynthesisConfig, TraceSynthesisRecord, TraceSynthesizer,
};
pub use tokenizer::{
    HuggingFaceTokenizer, NativeTiktokenTokenizer, ServerTokenizer, TextTokenizer,
    TiktokenEncoding, TiktokenTokenizer, find_tiktoken_model_file,
};
pub mod loader;
pub mod media;
pub use compose::{
    ComposeConfig, Composer, HashIdentityTracePromptStorage, MaterializedTracePromptStorage,
    ModelSelector, ModelSelectorFactory, RandomModelSelectorFactory,
    RoundRobinModelSelectorFactory, SessionIdGenerator, TracePromptStoragePolicy,
};
