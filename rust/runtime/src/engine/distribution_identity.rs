// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact executable-content identity for the selected runner distribution.
//!
//! The identity contract is the BLAKE3 digest of
//! `b"aiperf-runner-distribution-v1\0" || executable_bytes`. On Linux the
//! executing inode is opened through `/proc/self/exe`, so replacing the path
//! after process launch cannot make the capability process hash a different
//! image. Other platforms use their current-executable path because the Rust
//! standard library does not expose a portable executable handle.

use std::fs::File;
use std::io::{self, Read};

/// Versioned domain separating executable identities from every other digest.
pub const DISTRIBUTION_ID_DOMAIN: &[u8] = b"aiperf-runner-distribution-v1\0";

/// Source of the complete executable image used for identity computation.
///
/// The trait keeps image acquisition replaceable for platforms that can
/// provide a stronger process-image handle than a filesystem path.
pub trait ExecutableImageSource {
    /// Open the complete executable image associated with the current process.
    fn open_image(&self) -> io::Result<File>;
}

/// Platform-aware source for the image executing the current process.
#[derive(Clone, Copy, Debug, Default)]
pub struct CurrentExecutableImage;

impl ExecutableImageSource for CurrentExecutableImage {
    fn open_image(&self) -> io::Result<File> {
        #[cfg(target_os = "linux")]
        {
            File::open("/proc/self/exe")
        }

        #[cfg(not(target_os = "linux"))]
        {
            File::open(std::env::current_exe()?)
        }
    }
}

/// Compute the identity of the image executing the current process.
pub fn current_distribution_id() -> io::Result<String> {
    distribution_id_from_source(&CurrentExecutableImage)
}

/// Compute an executable identity through an injected image source.
pub fn distribution_id_from_source(source: &dyn ExecutableImageSource) -> io::Result<String> {
    let mut image = source.open_image()?;
    distribution_id_from_reader(&mut image)
}

fn distribution_id_from_reader(image: &mut dyn Read) -> io::Result<String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DISTRIBUTION_ID_DOMAIN);

    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = image.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn algorithm_is_domain_separated_and_byte_exact() {
        let mut image = Cursor::new(b"runner-image\0with-binary-bytes".as_slice());

        assert_eq!(
            distribution_id_from_reader(&mut image).unwrap(),
            "blake3:3cef527b3dd7185b4ab8590b425b730bd84d695f5c9e1a97302780b7056bf2e9"
        );
    }

    #[test]
    fn behavior_distinct_images_have_distinct_identities() {
        let mut first = Cursor::new(b"runner-image-a".as_slice());
        let mut second = Cursor::new(b"runner-image-b".as_slice());

        assert_ne!(
            distribution_id_from_reader(&mut first).unwrap(),
            distribution_id_from_reader(&mut second).unwrap()
        );
    }

    #[test]
    fn current_image_is_readable() {
        let identity = current_distribution_id().unwrap();

        assert_eq!(identity.len(), "blake3:".len() + 64);
        assert!(
            identity["blake3:".len()..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        );
    }
}
