// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Controller-owned admission credentials for cellular registrations.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use anyhow::{Result, ensure};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::TryRngCore;

use crate::cellular::transport::connect::ControllerPeerBinding;
use crate::cellular::transport::{CellRegister, CellRegistrationProof};
use crate::engine::cellular_bootstrap::CellularRole;

const REGISTRATION_PROTOCOL_VERSION: u8 = 1;
const TRANSCRIPT_DOMAIN: &[u8] = b"aiperf-cellular-registration-v1\0";
const PEER_ADMISSION_TRANSCRIPT_DOMAIN: &[u8] = b"aiperf-cellular-peer-admission-v1\0";
pub(crate) const ADMISSION_PURPOSE_COUNT: usize = 6;
const REGISTER_REPLY_TRANSCRIPT_DOMAIN: &[u8] = b"aiperf-cellular-register-reply-v1\0";

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

    fn sign_controller(&self, transcript: &[u8]) -> Result<Signature> {
        match &self.authority {
            ProcessSecurityAuthority::Controller { signer, .. } => Ok(signer.sign(transcript)),
            ProcessSecurityAuthority::Worker { .. } => {
                anyhow::bail!("worker security context cannot attest controller replies")
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
        CellRegistrationAuthority::from_controller_context(self)
    }

    pub(crate) fn reply_attestor(self: &Arc<Self>) -> Result<ControllerRegisterAttestor> {
        ensure!(
            matches!(self.authority, ProcessSecurityAuthority::Controller { .. }),
            "worker has no controller reply authority"
        );
        Ok(ControllerRegisterAttestor {
            context: Arc::clone(self),
        })
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

/// Controller-owned cell verifiers, run nonce, and provisioned reply-attestation capability.
pub(crate) struct CellRegistrationAuthority {
    run_nonce: [u8; 32],
    role_verifiers: Box<[RoleVerifyingKey]>,
    reply_attestor: ControllerRegisterAttestor,
}

/// The private, cell-specific signing key delivered only by a trusted launcher.
#[cfg_attr(test, derive(Clone))]
pub(crate) struct CellRegistrationCredential {
    cell_id: u32,
    context: Arc<CellSecurityContext>,
}

/// Signed controller evidence for one complete registration reply.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ControllerRegisterAttestation {
    /// Fixed protocol version for the signed reply transcript.
    pub version: u8,
    /// Per-run nonce selected by the deployment or local controller.
    pub run_nonce: [u8; 32],
    /// Ed25519 signature over controller peer, registration, and reply material.
    pub signature: Vec<u8>,
}

/// Controller-only signing authority for registration replies.
#[derive(Clone)]
pub(crate) struct ControllerRegisterAttestor {
    context: Arc<CellSecurityContext>,
}

/// Role-local verifier for controller registration replies.
#[derive(Clone)]
pub(crate) struct ControllerRegisterVerifier {
    run_nonce: [u8; 32],
    verifying_key: VerifyingKey,
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
        let context = Arc::new(CellSecurityContext::controller(
            run_nonce,
            controller_signer,
            role_verifiers.into_boxed_slice(),
        )?);
        Ok((context.registration_authority()?, credentials))
    }

    #[cfg(test)]
    pub(crate) fn run_nonce(&self) -> [u8; 32] {
        self.run_nonce
    }

    fn from_controller_context(context: &Arc<CellSecurityContext>) -> Result<Self> {
        let role_verifiers = context.role_verifiers()?;
        ensure!(
            !role_verifiers.is_empty(),
            "cell registration roster requires at least one public key"
        );
        Ok(Self {
            run_nonce: context.run_nonce,
            role_verifiers,
            reply_attestor: context.reply_attestor()?,
        })
    }

    pub(crate) fn reply_attestor(&self) -> ControllerRegisterAttestor {
        self.reply_attestor.clone()
    }

    pub(crate) fn verify(
        &self,
        registration: &CellRegister,
        controller_peer: &velo::PeerInfo,
    ) -> Result<VerifiedCellRegistration> {
        let proof = registration
            .registration_proof
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("cell registration proof is missing"))?;
        ensure!(
            proof.version == REGISTRATION_PROTOCOL_VERSION && proof.run_nonce == self.run_nonce,
            "cell registration proof does not belong to this run"
        );
        ensure!(
            proof.controller_binding.matches_peer(controller_peer)?,
            "cell registration names a different controller peer"
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
                &proof.controller_binding,
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

impl ControllerRegisterAttestor {
    #[cfg(test)]
    pub(crate) fn mint(run_nonce: [u8; 32]) -> Result<Self> {
        let signer = SigningKey::from_bytes(&random_nonce("controller reply key")?);
        let placeholder = RoleVerifyingKey {
            role: CellularRole::Cell(0),
            verifier: signer.verifying_key(),
        };
        Ok(Self {
            context: Arc::new(CellSecurityContext::controller(
                run_nonce,
                signer,
                vec![placeholder].into_boxed_slice(),
            )?),
        })
    }

    #[cfg(test)]
    pub(crate) fn verifier(&self) -> ControllerRegisterVerifier {
        ControllerRegisterVerifier {
            run_nonce: self.context.run_nonce,
            verifying_key: match &self.context.authority {
                ProcessSecurityAuthority::Controller { signer, .. } => signer.verifying_key(),
                ProcessSecurityAuthority::Worker { .. } => unreachable!("attestor is controller"),
            },
        }
    }

    pub(crate) fn attest(
        &self,
        controller_binding: &ControllerPeerBinding,
        registration_frame: &[u8],
        reply_payload: &[u8],
    ) -> Result<ControllerRegisterAttestation> {
        let transcript = register_reply_transcript(
            controller_binding,
            registration_frame,
            reply_payload,
            REGISTRATION_PROTOCOL_VERSION,
            self.context.run_nonce,
        );
        Ok(ControllerRegisterAttestation {
            version: REGISTRATION_PROTOCOL_VERSION,
            run_nonce: self.context.run_nonce,
            signature: self
                .context
                .sign_controller(&transcript)?
                .to_bytes()
                .to_vec(),
        })
    }
}

impl ControllerRegisterVerifier {
    pub(crate) fn from_public_key(run_nonce: [u8; 32], verifying_key: [u8; 32]) -> Result<Self> {
        Ok(Self {
            run_nonce,
            verifying_key: VerifyingKey::from_bytes(&verifying_key)
                .map_err(|_| anyhow::anyhow!("controller verification key is malformed"))?,
        })
    }
    pub(crate) fn verify(
        &self,
        controller_binding: &ControllerPeerBinding,
        registration_frame: &[u8],
        reply_payload: &[u8],
        attestation: &ControllerRegisterAttestation,
    ) -> Result<()> {
        ensure!(
            attestation.version == REGISTRATION_PROTOCOL_VERSION
                && attestation.run_nonce == self.run_nonce,
            "controller registration attestation does not belong to this run"
        );
        let signature = Signature::from_slice(&attestation.signature)
            .map_err(|_| anyhow::anyhow!("controller registration attestation is malformed"))?;
        self.verifying_key
            .verify(
                &register_reply_transcript(
                    controller_binding,
                    registration_frame,
                    reply_payload,
                    attestation.version,
                    attestation.run_nonce,
                ),
                &signature,
            )
            .map_err(|_| anyhow::anyhow!("controller registration attestation is invalid"))
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
        controller_binding: ControllerPeerBinding,
    ) -> Result<CellRegistrationProof> {
        let transcript = registration_transcript(
            self.cell_id(),
            cell_peer,
            artifact_capability_digest,
            &controller_binding,
            REGISTRATION_PROTOCOL_VERSION,
            self.context.run_nonce,
        );
        Ok(CellRegistrationProof {
            version: REGISTRATION_PROTOCOL_VERSION,
            run_nonce: self.context.run_nonce,
            controller_binding,
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
    controller_binding: &ControllerPeerBinding,
    version: u8,
    run_nonce: [u8; 32],
) -> Vec<u8> {
    let mut transcript = Vec::with_capacity(TRANSCRIPT_DOMAIN.len() + 1 + 32 + 4 + 32 + 1 + 32);
    transcript.extend_from_slice(TRANSCRIPT_DOMAIN);
    transcript.push(version);
    transcript.extend_from_slice(&run_nonce);
    transcript.extend_from_slice(&cell_id.to_le_bytes());
    transcript.extend_from_slice(blake3::hash(cell_peer).as_bytes());
    controller_binding.append_transcript(&mut transcript);
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

fn register_reply_transcript(
    controller_binding: &ControllerPeerBinding,
    registration_frame: &[u8],
    reply_payload: &[u8],
    version: u8,
    run_nonce: [u8; 32],
) -> Vec<u8> {
    let mut transcript = Vec::with_capacity(REGISTER_REPLY_TRANSCRIPT_DOMAIN.len() + 1 + 32 + 96);
    transcript.extend_from_slice(REGISTER_REPLY_TRANSCRIPT_DOMAIN);
    transcript.push(version);
    transcript.extend_from_slice(&run_nonce);
    controller_binding.append_transcript(&mut transcript);
    transcript.extend_from_slice(blake3::hash(registration_frame).as_bytes());
    transcript.extend_from_slice(blake3::hash(reply_payload).as_bytes());
    transcript
}

#[cfg(test)]
mod tests {
    use crate::cellular::transport::CellRegister;
    use crate::cellular::transport::connect::{ControllerPeerBinding, DialedControllerAddress};

    use super::{CellPeerAdmissionPurpose, CellRegistrationAuthority, ControllerRegisterAttestor};

    fn controller_binding() -> (velo::PeerInfo, ControllerPeerBinding) {
        let peer = velo::PeerInfo::new(
            velo::InstanceId::new_v4(),
            velo::WorkerAddress::from_encoded(vec![0x80]),
        );
        let binding = ControllerPeerBinding::new(
            &peer,
            DialedControllerAddress::Tcp("127.0.0.1:9500".parse().unwrap()),
        )
        .unwrap();
        (peer, binding)
    }

    #[test]
    fn registration_proof_binds_controller_cell_peer_and_capability_digest() {
        let (authority, credentials) = CellRegistrationAuthority::mint(2).unwrap();
        let credential = &credentials[1];
        let peer = b"encoded-peer";
        let digest = [0x11; 32];
        let (controller_peer, binding) = controller_binding();
        let proof = credential
            .sign_register(peer, Some(digest), binding)
            .unwrap();
        let register = CellRegister {
            cell_id: 1,
            cell_peer: peer.to_vec(),
            artifact_capability_digest: Some(digest),
            registration_proof: Some(proof),
        };

        assert!(authority.verify(&register, &controller_peer).is_ok());
        let changed_controller_peer = velo::PeerInfo::new(
            controller_peer.instance_id(),
            velo::WorkerAddress::from_encoded(vec![0x81]),
        );
        assert!(
            authority
                .verify(&register, &changed_controller_peer)
                .is_err()
        );
        let mut changed_digest = register.clone();
        changed_digest.artifact_capability_digest = Some([0x22; 32]);
        assert!(authority.verify(&changed_digest, &controller_peer).is_err());
    }

    // Catches a cell signature that authenticates only structural peer equality,
    // rather than the exact controller binding carried by the proof.
    #[test]
    fn registration_signature_rejects_consistently_replaced_controller_binding() {
        let (authority, credentials) = CellRegistrationAuthority::mint(1).unwrap();
        let (controller_peer, binding) = controller_binding();
        let proof = credentials[0]
            .sign_register(b"cell-peer", None, binding)
            .unwrap();
        let mut register = CellRegister {
            cell_id: 0,
            cell_peer: b"cell-peer".to_vec(),
            artifact_capability_digest: None,
            registration_proof: Some(proof),
        };
        let changed_controller_peer = velo::PeerInfo::new(
            controller_peer.instance_id(),
            velo::WorkerAddress::from_encoded(vec![0x81]),
        );
        register
            .registration_proof
            .as_mut()
            .unwrap()
            .controller_binding = ControllerPeerBinding::new(
            &changed_controller_peer,
            DialedControllerAddress::Tcp("127.0.0.1:9500".parse().unwrap()),
        )
        .unwrap();

        assert!(
            authority
                .verify(&register, &changed_controller_peer)
                .is_err(),
            "the original signature must not authorize a replacement binding"
        );
    }

    // Catches a cell signature that omits only the resolved dial target.
    #[test]
    fn registration_signature_rejects_dial_only_replacement() {
        let (authority, credentials) = CellRegistrationAuthority::mint(1).unwrap();
        let (controller_peer, binding) = controller_binding();
        let proof = credentials[0]
            .sign_register(b"cell-peer", None, binding)
            .unwrap();
        let mut register = CellRegister {
            cell_id: 0,
            cell_peer: b"cell-peer".to_vec(),
            artifact_capability_digest: None,
            registration_proof: Some(proof),
        };
        register
            .registration_proof
            .as_mut()
            .unwrap()
            .controller_binding = ControllerPeerBinding::new(
            &controller_peer,
            DialedControllerAddress::Tcp("127.0.0.1:9501".parse().unwrap()),
        )
        .unwrap();

        assert!(
            authority.verify(&register, &controller_peer).is_err(),
            "the original cell signature must not authorize a changed dial target"
        );
    }

    // Catches controller attestations that omit any exact registration-reply input.
    #[test]
    fn controller_attestation_covers_binding_registration_frame_and_reply_payload() {
        let attestor = ControllerRegisterAttestor::mint([0x42; 32]).unwrap();
        let verifier = attestor.verifier();
        let (_, binding) = controller_binding();
        let changed_peer = velo::PeerInfo::new(
            velo::InstanceId::new_v4(),
            velo::WorkerAddress::from_encoded(vec![0x82]),
        );
        let changed_binding = ControllerPeerBinding::new(
            &changed_peer,
            DialedControllerAddress::Tcp("127.0.0.1:9501".parse().unwrap()),
        )
        .unwrap();
        let registration_frame = b"exact registration frame";
        let reply_payload = b"exact reply payload";
        let attestation = attestor
            .attest(&binding, registration_frame, reply_payload)
            .unwrap();

        assert!(
            verifier
                .verify(
                    &changed_binding,
                    registration_frame,
                    reply_payload,
                    &attestation,
                )
                .is_err(),
            "the controller binding must be covered"
        );
        assert!(
            verifier
                .verify(
                    &binding,
                    b"changed registration frame",
                    reply_payload,
                    &attestation,
                )
                .is_err(),
            "the exact registration frame must be covered"
        );
        assert!(
            verifier
                .verify(
                    &binding,
                    registration_frame,
                    b"changed reply payload",
                    &attestation,
                )
                .is_err(),
            "the exact reply payload must be covered"
        );
    }

    // Catches a controller attestation that omits only the resolved dial target.
    #[test]
    fn controller_attestation_rejects_dial_only_replacement() {
        let attestor = ControllerRegisterAttestor::mint([0x42; 32]).unwrap();
        let verifier = attestor.verifier();
        let (controller_peer, binding) = controller_binding();
        let changed_binding = ControllerPeerBinding::new(
            &controller_peer,
            DialedControllerAddress::Tcp("127.0.0.1:9501".parse().unwrap()),
        )
        .unwrap();
        let attestation = attestor
            .attest(&binding, b"registration", b"reply")
            .unwrap();

        assert!(
            verifier
                .verify(&changed_binding, b"registration", b"reply", &attestation,)
                .is_err(),
            "the original controller attestation must not authorize a changed dial target"
        );
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
