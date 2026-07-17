// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LevelDB-style block framing for the W&B `.wandb` transaction log.
//!
//! Byte-exact framing contract from `wandb/sdk/internal/datastore.py`
//! (W&B 0.28.0):
//!
//! - Header: `":W&B"` (4 bytes) + magic `0xBEE1` (u16 LE) + version `0` (u8).
//! - Records live in fixed 32768-byte blocks. Each record is a 7-byte header
//!   `{checksum: u32 LE, length: u16 LE, type: u8}` followed by `length` bytes.
//! - `type` is `FULL` when a record fits the current block; otherwise it is
//!   split `FIRST`/`MIDDLE*`/`LAST`. When a block has `< 7` trailing bytes they
//!   are zero-padded before the next record.
//! - `checksum` is the standard CRC-32 (IEEE, the same polynomial `zlib.crc32`
//!   uses — despite the SDK's "crc32c" docstring the code calls `zlib.crc32`)
//!   of the single type byte followed by the record data.

const HEADER_LEN: usize = 7;
const BLOCK_LEN: usize = 32768;
const DATA_LEN: usize = BLOCK_LEN - HEADER_LEN;

const TYPE_FULL: u8 = 1;
const TYPE_FIRST: u8 = 2;
const TYPE_MIDDLE: u8 = 3;
const TYPE_LAST: u8 = 4;

const HEADER_IDENT: &[u8; 4] = b":W&B";
const HEADER_MAGIC: u16 = 0xBEE1;
const HEADER_VERSION: u8 = 0;

/// Streaming writer that frames serialized protobuf records into the datastore
/// block layout. Construct with [`DataStore::new`], append with
/// [`DataStore::write`], then flush the buffer with [`DataStore::into_bytes`].
pub struct DataStore {
    buf: Vec<u8>,
    crc: Crc32,
}

impl DataStore {
    /// Start a new datastore stream, emitting the fixed file header.
    pub fn new() -> Self {
        let mut buf = Vec::with_capacity(BLOCK_LEN);
        buf.extend_from_slice(HEADER_IDENT);
        buf.extend_from_slice(&HEADER_MAGIC.to_le_bytes());
        buf.push(HEADER_VERSION);
        Self {
            buf,
            crc: Crc32::new(),
        }
    }

    /// Append one already-serialized record body, splitting across blocks.
    pub fn write(&mut self, data: &[u8]) {
        self.write_data(data);
    }

    /// Consume the writer and return the complete `.wandb` byte stream.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Port of `DataStore._write_record`: emit one framed record that fits the
    /// remaining block space.
    fn write_record(&mut self, data: &[u8], dtype: u8) {
        let checksum = self.crc.checksum_with_type(dtype, data);
        self.buf.extend_from_slice(&checksum.to_le_bytes());
        self.buf
            .extend_from_slice(&(data.len() as u16).to_le_bytes());
        self.buf.push(dtype);
        self.buf.extend_from_slice(data);
    }

    /// Port of `DataStore._write_data`: pad, then write FULL or split records.
    fn write_data(&mut self, data: &[u8]) {
        let mut space_left = BLOCK_LEN - (self.buf.len() % BLOCK_LEN);

        if space_left < HEADER_LEN {
            self.buf.extend(std::iter::repeat_n(0u8, space_left));
            space_left = BLOCK_LEN;
        }

        if data.len() + HEADER_LEN <= space_left {
            self.write_record(data, TYPE_FULL);
            return;
        }

        let data_room = space_left - HEADER_LEN;
        self.write_record(&data[..data_room], TYPE_FIRST);
        let mut used = data_room;
        let mut left = data.len() - data_room;

        while left > DATA_LEN {
            self.write_record(&data[used..used + DATA_LEN], TYPE_MIDDLE);
            used += DATA_LEN;
            left -= DATA_LEN;
        }
        self.write_record(&data[used..], TYPE_LAST);
    }
}

impl Default for DataStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Precomputed reflected CRC-32 (IEEE polynomial `0xEDB88320`), matching
/// `zlib.crc32`. The datastore seeds the CRC with the record's type byte, then
/// folds in the record data.
struct Crc32 {
    table: [u32; 256],
}

impl Crc32 {
    fn new() -> Self {
        let mut table = [0u32; 256];
        let mut n = 0;
        while n < 256 {
            let mut c = n as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
                k += 1;
            }
            table[n] = c;
            n += 1;
        }
        Self { table }
    }

    /// `zlib.crc32(data, zlib.crc32(chr(dtype)))` == CRC-32 of `[dtype] ++ data`.
    fn checksum_with_type(&self, dtype: u8, data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        crc = self.update_byte(crc, dtype);
        for &b in data {
            crc = self.update_byte(crc, b);
        }
        crc ^ 0xFFFF_FFFF
    }

    #[inline]
    fn update_byte(&self, crc: u32, b: u8) -> u32 {
        self.table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8)
    }
}
