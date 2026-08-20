// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Controller-owned admission credentials for cellular registrations.

use anyhow::{Result, ensure};
use base64::Engine;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::TryRngCore;

use crate::cellular::transport::{CellRegister, CellRegistrationProof};

const REGISTRATION_PROTOCOL_VERSION: u8 = 1;
const TRANSCRIPT_DOMAIN: &[u8] = b"aiperf-cellular-registration-v1\0";
const PEER_ADMISSION_TRANSCRIPT_DOMAIN: &[u8] = b"aiperf-cellular-peer-admission-v1\0";

/// The controller operation for which a fresh Velo peer may be admitted.
///
/// A proof for one purpose is deliberately unusable for another; a partition
/// shipper must not be able to turn its ticket into a dataset subscription.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum CellPeerAdmissionPurpose {
    /// Delivers a records shard to the controller.
    Partition = 1,
    /// Delivers a folded column store to the controller.
    StorePartition = 2,
    /// Subscribes to controller-owned phase transitions.
    PhaserSubscribe = 3,
    /// Subscribes to controller-owned dataset chunks.
    DatasetSubscribe = 4,
    /// Opens a Velo artifact stream.
    ArtifactOpen = 5,
    /// Delivers an aggregator-owned upstream partition.
    AggregatorStorePartition = 6,
}

/// Wire-visible proof for admitting one fresh, purpose-limited Velo peer.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CellPeerAdmissionProof {
    version: u8,
    run_nonce: [u8; 32],
    signature: Vec<u8>,
}

/// Controller-owned public keys and nonce for one cellular run.
pub(crate) struct CellRegistrationAuthority {
    run_nonce: [u8; 32],
    verifying_keys: Vec<VerifyingKey>,
}

/// The private, cell-specific signing key delivered only by a trusted launcher.
#[derive(Clone)]
pub(crate) struct CellRegistrationCredential {
    cell_id: u32,
    run_nonce: [u8; 32],
    signing_key: SigningKey,
}

/// A controller-verified registration identity.
#[derive(Clone, Copy)]
pub(crate) struct VerifiedCellRegistration {
    cell_id: u32,
}

impl VerifiedCellRegistration {
    pub(crate) fn cell_id(self) -> u32 {
        self.cell_id
    }
}

impl CellRegistrationAuthority {
    pub(crate) fn mint(cell_count: u32) -> Result<(Self, Vec<CellRegistrationCredential>)> {
        ensure!(
            cell_count > 0,
            "cell registration requires at least one cell"
        );
        let mut run_nonce = [0_u8; 32];
        rand::rngs::OsRng
            .try_fill_bytes(&mut run_nonce)
            .map_err(|_| anyhow::anyhow!("OS RNG could not mint cellular registration nonce"))?;
        let mut verifying_keys = Vec::with_capacity(cell_count as usize);
        let mut credentials = Vec::with_capacity(cell_count as usize);
        for cell_id in 0..cell_count {
            let mut seed = [0_u8; 32];
            rand::rngs::OsRng
                .try_fill_bytes(&mut seed)
                .map_err(|_| anyhow::anyhow!("OS RNG could not mint cellular registration key"))?;
            let signing_key = SigningKey::from_bytes(&seed);
            verifying_keys.push(signing_key.verifying_key());
            credentials.push(CellRegistrationCredential {
                cell_id,
                run_nonce,
                signing_key,
            });
        }
        Ok((
            Self {
                run_nonce,
                verifying_keys,
            },
            credentials,
        ))
    }

    pub(crate) fn verify(&self, registration: &CellRegister) -> Result<VerifiedCellRegistration> {
        let proof = registration
            .registration_proof
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("cell registration proof is missing"))?;
        ensure!(
            proof.version == REGISTRATION_PROTOCOL_VERSION && proof.run_nonce == self.run_nonce,
            "cell registration proof does not belong to this run"
        );
        let key = self
            .verifying_keys
            .get(registration.cell_id as usize)
            .ok_or_else(|| anyhow::anyhow!("cell registration id is out of range"))?;
        let signature = Signature::from_slice(&proof.signature)
            .map_err(|_| anyhow::anyhow!("cell registration proof is malformed"))?;
        key.verify(
            &registration_transcript(
                registration.cell_id,
                &registration.cell_peer,
                registration.artifact_capability_digest,
                proof.version,
                proof.run_nonce,
            ),
            &signature,
        )
        .map_err(|_| anyhow::anyhow!("cell registration proof is invalid"))?;
        Ok(VerifiedCellRegistration {
            cell_id: registration.cell_id,
        })
    }

    /// Verify a ticket before an AIPerf handler registers a fresh Velo peer.
    pub(crate) fn verify_peer_admission(
        &self,
        cell_id: u32,
        purpose: CellPeerAdmissionPurpose,
        cell_peer: &[u8],
        proof: &CellPeerAdmissionProof,
    ) -> Result<()> {
        ensure!(
            proof.version == REGISTRATION_PROTOCOL_VERSION && proof.run_nonce == self.run_nonce,
            "cell peer admission proof does not belong to this run"
        );
        let key = self
            .verifying_keys
            .get(cell_id as usize)
            .ok_or_else(|| anyhow::anyhow!("cell peer admission id is out of range"))?;
        let signature = Signature::from_slice(&proof.signature)
            .map_err(|_| anyhow::anyhow!("cell peer admission proof is malformed"))?;
        key.verify(
            &peer_admission_transcript(cell_id, purpose, cell_peer, proof.version, proof.run_nonce),
            &signature,
        )
        .map_err(|_| anyhow::anyhow!("cell peer admission proof is invalid"))
    }
}

impl CellRegistrationCredential {
    pub(crate) fn cell_id(&self) -> u32 {
        self.cell_id
    }

    pub(crate) fn encode_launch_value(&self) -> String {
        let mut raw = Vec::with_capacity(68);
        raw.extend_from_slice(&self.cell_id.to_le_bytes());
        raw.extend_from_slice(&self.run_nonce);
        raw.extend_from_slice(&self.signing_key.to_bytes());
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(raw)
    }

    pub(crate) fn from_launch_value(raw: &str) -> Result<Self> {
        let raw = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(raw)
            .map_err(|_| anyhow::anyhow!("cell registration credential is malformed"))?;
        ensure!(
            raw.len() == 68,
            "cell registration credential has invalid length"
        );
        let cell_id: [u8; 4] = raw[..4]
            .try_into()
            .map_err(|_| anyhow::anyhow!("cell registration credential is malformed"))?;
        let run_nonce: [u8; 32] = raw[4..36]
            .try_into()
            .map_err(|_| anyhow::anyhow!("cell registration credential is malformed"))?;
        let signing_key: [u8; 32] = raw[36..]
            .try_into()
            .map_err(|_| anyhow::anyhow!("cell registration credential is malformed"))?;
        Ok(Self {
            cell_id: u32::from_le_bytes(cell_id),
            run_nonce,
            signing_key: SigningKey::from_bytes(&signing_key),
        })
    }
    pub(crate) fn sign_register(
        &self,
        cell_peer: &[u8],
        artifact_capability_digest: Option<[u8; 32]>,
    ) -> Result<CellRegistrationProof> {
        let transcript = registration_transcript(
            self.cell_id,
            cell_peer,
            artifact_capability_digest,
            REGISTRATION_PROTOCOL_VERSION,
            self.run_nonce,
        );
        Ok(CellRegistrationProof {
            version: REGISTRATION_PROTOCOL_VERSION,
            run_nonce: self.run_nonce,
            signature: self.signing_key.sign(&transcript).to_bytes().to_vec(),
        })
    }

    /// Mint a ticket for exactly one ephemeral peer and controller operation.
    pub(crate) fn sign_peer_admission(
        &self,
        purpose: CellPeerAdmissionPurpose,
        cell_peer: &[u8],
    ) -> Result<CellPeerAdmissionProof> {
        let transcript = peer_admission_transcript(
            self.cell_id,
            purpose,
            cell_peer,
            REGISTRATION_PROTOCOL_VERSION,
            self.run_nonce,
        );
        Ok(CellPeerAdmissionProof {
            version: REGISTRATION_PROTOCOL_VERSION,
            run_nonce: self.run_nonce,
            signature: self.signing_key.sign(&transcript).to_bytes().to_vec(),
        })
    }
}

fn registration_transcript(
    cell_id: u32,
    cell_peer: &[u8],
    artifact_capability_digest: Option<[u8; 32]>,
    version: u8,
    run_nonce: [u8; 32],
) -> Vec<u8> {
    let mut transcript = Vec::with_capacity(TRANSCRIPT_DOMAIN.len() + 1 + 32 + 4 + 32 + 1 + 32);
    transcript.extend_from_slice(TRANSCRIPT_DOMAIN);
    transcript.push(version);
    transcript.extend_from_slice(&run_nonce);
    transcript.extend_from_slice(&cell_id.to_le_bytes());
    transcript.extend_from_slice(blake3::hash(cell_peer).as_bytes());
    match artifact_capability_digest {
        Some(digest) => {
            transcript.push(1);
            transcript.extend_from_slice(&digest);
        }
        None => transcript.push(0),
    }
    transcript
}

fn peer_admission_transcript(
    cell_id: u32,
    purpose: CellPeerAdmissionPurpose,
    cell_peer: &[u8],
    version: u8,
    run_nonce: [u8; 32],
) -> Vec<u8> {
    let mut transcript =
        Vec::with_capacity(PEER_ADMISSION_TRANSCRIPT_DOMAIN.len() + 1 + 32 + 4 + 1 + 32);
    transcript.extend_from_slice(PEER_ADMISSION_TRANSCRIPT_DOMAIN);
    transcript.push(version);
    transcript.extend_from_slice(&run_nonce);
    transcript.extend_from_slice(&cell_id.to_le_bytes());
    transcript.push(purpose as u8);
    transcript.extend_from_slice(blake3::hash(cell_peer).as_bytes());
    transcript
}

#[cfg(test)]
mod tests {
    use crate::cellular::transport::CellRegister;

    use super::{CellPeerAdmissionPurpose, CellRegistrationAuthority};

    #[test]
    fn registration_proof_binds_cell_peer_and_capability_digest() {
        let (authority, credentials) = CellRegistrationAuthority::mint(2).unwrap();
        let credential = &credentials[1];
        let peer = b"encoded-peer";
        let digest = [0x11; 32];
        let proof = credential.sign_register(peer, Some(digest)).unwrap();
        let register = CellRegister {
            cell_id: 1,
            cell_peer: peer.to_vec(),
            artifact_capability_digest: Some(digest),
            registration_proof: Some(proof),
        };

        assert!(authority.verify(&register).is_ok());
        let mut changed_digest = register.clone();
        changed_digest.artifact_capability_digest = Some([0x22; 32]);
        assert!(authority.verify(&changed_digest).is_err());
    }

    #[test]
    fn peer_admission_ticket_is_bound_to_peer_cell_and_purpose() {
        let (authority, credentials) = CellRegistrationAuthority::mint(2).unwrap();
        let peer = b"ephemeral-peer";
        let proof = credentials[1]
            .sign_peer_admission(CellPeerAdmissionPurpose::Partition, peer)
            .unwrap();

        assert!(
            authority
                .verify_peer_admission(1, CellPeerAdmissionPurpose::Partition, peer, &proof)
                .is_ok()
        );
        assert!(
            authority
                .verify_peer_admission(1, CellPeerAdmissionPurpose::StorePartition, peer, &proof)
                .is_err()
        );
        assert!(
            authority
                .verify_peer_admission(0, CellPeerAdmissionPurpose::Partition, peer, &proof)
                .is_err()
        );
        assert!(
            authority
                .verify_peer_admission(1, CellPeerAdmissionPurpose::Partition, b"other", &proof)
                .is_err()
        );
        for purpose in [
            CellPeerAdmissionPurpose::StorePartition,
            CellPeerAdmissionPurpose::PhaserSubscribe,
            CellPeerAdmissionPurpose::DatasetSubscribe,
            CellPeerAdmissionPurpose::ArtifactOpen,
            CellPeerAdmissionPurpose::AggregatorStorePartition,
        ] {
            assert!(
                authority
                    .verify_peer_admission(1, purpose, peer, &proof)
                    .is_err(),
                "a partition ticket must not authorize {purpose:?}"
            );
        }
    }
}
