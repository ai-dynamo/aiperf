// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Controller-owned admission credentials for cellular registrations.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use anyhow::{Result, ensure};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::TryRngCore;

use crate::cellular::transport::{CellRegister, CellRegistrationProof};
use crate::engine::cellular_bootstrap::CellularRole;

const REGISTRATION_PROTOCOL_VERSION: u8 = 1;
const TRANSCRIPT_DOMAIN: &[u8] = b"aiperf-cellular-registration-v1\0";
const PEER_ADMISSION_TRANSCRIPT_DOMAIN: &[u8] = b"aiperf-cellular-peer-admission-v1\0";
pub(crate) const ADMISSION_PURPOSE_COUNT: usize = 6;

/// One public role key in the controller's fixed authorization roster.
#[derive(Clone, Copy)]
pub(crate) struct RoleVerifyingKey {
    pub(crate) role: CellularRole,
    pub(crate) verifier: VerifyingKey,
}

/// The sole private cellular authority owned by one process.
pub(crate) struct CellSecurityContext {
    run_nonce: [u8; 32],
    session_nonce: [u8; 32],
    authority: ProcessSecurityAuthority,
    #[allow(dead_code)]
    send_sequences: [AtomicU64; ADMISSION_PURPOSE_COUNT],
}

enum ProcessSecurityAuthority {
    Controller {
        signer: SigningKey,
        role_verifiers: Box<[RoleVerifyingKey]>,
    },
    Worker {
        role: CellularRole,
        signer: SigningKey,
        controller_verifier: VerifyingKey,
    },
}

impl CellSecurityContext {
    pub(crate) fn controller(
        run_nonce: [u8; 32],
        signer: SigningKey,
        role_verifiers: Box<[RoleVerifyingKey]>,
    ) -> Result<Self> {
        ensure!(
            !role_verifiers.is_empty(),
            "controller security roster is empty"
        );
        Ok(Self {
            run_nonce,
            session_nonce: random_nonce("process session nonce")?,
            authority: ProcessSecurityAuthority::Controller {
                signer,
                role_verifiers,
            },
            send_sequences: std::array::from_fn(|_| AtomicU64::new(0)),
        })
    }

    pub(crate) fn worker(
        run_nonce: [u8; 32],
        role: CellularRole,
        signer: SigningKey,
        controller_verifier: VerifyingKey,
    ) -> Result<Self> {
        Ok(Self {
            run_nonce,
            session_nonce: random_nonce("process session nonce")?,
            authority: ProcessSecurityAuthority::Worker {
                role,
                signer,
                controller_verifier,
            },
            send_sequences: std::array::from_fn(|_| AtomicU64::new(0)),
        })
    }

    pub(crate) fn run_nonce(&self) -> [u8; 32] {
        self.run_nonce
    }

    #[allow(dead_code)]
    pub(crate) fn session_nonce(&self) -> [u8; 32] {
        self.session_nonce
    }

    pub(crate) fn role(&self) -> Option<CellularRole> {
        match self.authority {
            ProcessSecurityAuthority::Controller { .. } => None,
            ProcessSecurityAuthority::Worker { role, .. } => Some(role),
        }
    }

    pub(crate) fn role_verifiers(&self) -> Result<Box<[RoleVerifyingKey]>> {
        match &self.authority {
            ProcessSecurityAuthority::Controller { role_verifiers, .. } => {
                Ok(role_verifiers.to_vec().into_boxed_slice())
            }
            ProcessSecurityAuthority::Worker { .. } => {
                anyhow::bail!("worker security context has no controller roster")
            }
        }
    }

    pub(crate) fn controller_verifier(&self) -> Result<VerifyingKey> {
        match &self.authority {
            ProcessSecurityAuthority::Worker {
                controller_verifier,
                ..
            } => Ok(*controller_verifier),
            ProcessSecurityAuthority::Controller { .. } => {
                anyhow::bail!("controller security context has no controller verifier")
            }
        }
    }

    fn sign_worker(&self, transcript: &[u8]) -> Result<Signature> {
        match &self.authority {
            ProcessSecurityAuthority::Worker { signer, .. } => Ok(signer.sign(transcript)),
            ProcessSecurityAuthority::Controller { .. } => {
                anyhow::bail!("controller security context cannot sign worker admission")
            }
        }
    }

    pub(crate) fn registration_credential(self: &Arc<Self>) -> Result<CellRegistrationCredential> {
        let Some(CellularRole::Cell(cell_id)) = self.role() else {
            anyhow::bail!("security context has no cell credential");
        };
        Ok(CellRegistrationCredential {
            cell_id,
            context: Arc::clone(self),
        })
    }

    pub(crate) fn registration_authority(self: &Arc<Self>) -> Result<CellRegistrationAuthority> {
        CellRegistrationAuthority::from_role_keys(self.run_nonce, self.role_verifiers()?)
    }
}

fn random_nonce(class: &str) -> Result<[u8; 32]> {
    let mut nonce = [0_u8; 32];
    rand::rngs::OsRng
        .try_fill_bytes(&mut nonce)
        .map_err(|_| anyhow::anyhow!("OS RNG could not mint cellular {class}"))?;
    Ok(nonce)
}

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
    role_verifiers: Box<[RoleVerifyingKey]>,
}

/// The private, cell-specific signing key delivered only by a trusted launcher.
#[cfg_attr(test, derive(Clone))]
pub(crate) struct CellRegistrationCredential {
    cell_id: u32,
    context: Arc<CellSecurityContext>,
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
    #[cfg(test)]
    pub(crate) fn mint(cell_count: u32) -> Result<(Self, Vec<CellRegistrationCredential>)> {
        ensure!(
            cell_count > 0,
            "cell registration requires at least one cell"
        );
        let mut run_nonce = [0_u8; 32];
        rand::rngs::OsRng
            .try_fill_bytes(&mut run_nonce)
            .map_err(|_| anyhow::anyhow!("OS RNG could not mint cellular registration nonce"))?;
        let controller_signer = SigningKey::from_bytes(&random_nonce("controller reply key")?);
        let controller_verifier = controller_signer.verifying_key();
        let mut role_verifiers = Vec::with_capacity(cell_count as usize);
        let mut credentials = Vec::with_capacity(cell_count as usize);
        for cell_id in 0..cell_count {
            let seed = random_nonce("registration key")?;
            let signing_key = SigningKey::from_bytes(&seed);
            role_verifiers.push(RoleVerifyingKey {
                role: CellularRole::Cell(cell_id),
                verifier: signing_key.verifying_key(),
            });
            let context = Arc::new(CellSecurityContext::worker(
                run_nonce,
                CellularRole::Cell(cell_id),
                signing_key,
                controller_verifier,
            )?);
            credentials.push(CellRegistrationCredential { cell_id, context });
        }
        Ok((
            Self {
                run_nonce,
                role_verifiers: role_verifiers.into_boxed_slice(),
            },
            credentials,
        ))
    }

    #[cfg(test)]
    pub(crate) fn run_nonce(&self) -> [u8; 32] {
        self.run_nonce
    }

    pub(crate) fn from_role_keys(
        run_nonce: [u8; 32],
        role_verifiers: Box<[RoleVerifyingKey]>,
    ) -> Result<Self> {
        ensure!(
            !role_verifiers.is_empty(),
            "cell registration roster requires at least one public key"
        );
        Ok(Self {
            run_nonce,
            role_verifiers,
        })
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
            .role_verifiers
            .iter()
            .find(|entry| entry.role == CellularRole::Cell(registration.cell_id))
            .map(|entry| &entry.verifier)
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
            .role_verifiers
            .iter()
            .find(|entry| entry.role == CellularRole::Cell(cell_id))
            .map(|entry| &entry.verifier)
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
    pub(crate) fn sign_register(
        &self,
        cell_peer: &[u8],
        artifact_capability_digest: Option<[u8; 32]>,
    ) -> Result<CellRegistrationProof> {
        let transcript = registration_transcript(
            self.cell_id(),
            cell_peer,
            artifact_capability_digest,
            REGISTRATION_PROTOCOL_VERSION,
            self.context.run_nonce,
        );
        Ok(CellRegistrationProof {
            version: REGISTRATION_PROTOCOL_VERSION,
            run_nonce: self.context.run_nonce,
            signature: self.context.sign_worker(&transcript)?.to_bytes().to_vec(),
        })
    }

    /// Mint a ticket for exactly one ephemeral peer and controller operation.
    pub(crate) fn sign_peer_admission(
        &self,
        purpose: CellPeerAdmissionPurpose,
        cell_peer: &[u8],
    ) -> Result<CellPeerAdmissionProof> {
        let transcript = peer_admission_transcript(
            self.cell_id(),
            purpose,
            cell_peer,
            REGISTRATION_PROTOCOL_VERSION,
            self.context.run_nonce,
        );
        Ok(CellPeerAdmissionProof {
            version: REGISTRATION_PROTOCOL_VERSION,
            run_nonce: self.context.run_nonce,
            signature: self.context.sign_worker(&transcript)?.to_bytes().to_vec(),
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
