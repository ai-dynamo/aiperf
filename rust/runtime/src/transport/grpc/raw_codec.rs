// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identity Tonic codec keeping the transport independent of protobuf types.

use bytes::{Buf, BufMut, Bytes};
use tonic::Status;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RawBytesCodec;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RawBytesEncoder;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RawBytesDecoder;

impl Codec for RawBytesCodec {
    type Encode = Bytes;
    type Decode = Bytes;
    type Encoder = RawBytesEncoder;
    type Decoder = RawBytesDecoder;

    fn encoder(&mut self) -> Self::Encoder {
        RawBytesEncoder
    }

    fn decoder(&mut self) -> Self::Decoder {
        RawBytesDecoder
    }
}

impl Encoder for RawBytesEncoder {
    type Item = Bytes;
    type Error = Status;

    fn encode(&mut self, item: Self::Item, destination: &mut EncodeBuf<'_>) -> Result<(), Status> {
        destination.put(item);
        Ok(())
    }
}

impl Decoder for RawBytesDecoder {
    type Item = Bytes;
    type Error = Status;

    fn decode(&mut self, source: &mut DecodeBuf<'_>) -> Result<Option<Self::Item>, Status> {
        let remaining = source.remaining();
        Ok(Some(source.copy_to_bytes(remaining)))
    }
}
