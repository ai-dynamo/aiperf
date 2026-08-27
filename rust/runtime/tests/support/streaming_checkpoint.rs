// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        AcquisitionHorizon, AdmissionHorizon, BudgetedCheckpointBytes, CheckpointBarrier,
        CheckpointCut, CheckpointEpoch, CheckpointError, CheckpointGeneration,
        CheckpointParticipantId, CommittedParticipantReceipt, CommittedParticipantState,
        DecodeHorizon, DiscoveryHorizon, EventTimeWatermark, OrderedActionHorizon,
        ParticipantInitialization, PreparedParticipantState, StreamingCheckpointParticipant,
        TerminalActionHorizon,
    },
    identity::{ContentDigest, GlobalSequence, SessionCausalFrontier},
    unit::{EventTimeUtc, SourcePosition},
};
use async_trait::async_trait;
use bytes::Bytes;

pub fn cut_at(value: u64) -> CheckpointCut {
    CheckpointCut {
        discovered: DiscoveryHorizon::new(SourcePosition::new(value)),
        acquired: AcquisitionHorizon::new(SourcePosition::new(value)),
        decoded: DecodeHorizon::new(SourcePosition::new(value)),
        ordered: OrderedActionHorizon::new(GlobalSequence::new(value)),
        admitted: AdmissionHorizon::new(GlobalSequence::new(value)),
        terminal: TerminalActionHorizon::new(GlobalSequence::new(value)),
        event_watermark: EventTimeWatermark::Hard {
            through: EventTimeUtc::new(value as i64).expect("non-negative test event time"),
        },
        causal_frontier: SessionCausalFrontier {
            through_sequence: GlobalSequence::new(value),
            event_time: Some(
                EventTimeUtc::new(value as i64).expect("non-negative test event time"),
            ),
            digest: ContentDigest::from_bytes([value as u8; 32]),
        },
    }
}

pub fn barrier_at(value: u64) -> CheckpointBarrier {
    CheckpointBarrier {
        epoch: CheckpointEpoch::new(value),
        cut: cut_at(value),
        plan_digest: ContentDigest::from_bytes([0x55; 32]),
    }
}

pub fn receipt_for(prepared: &PreparedParticipantState) -> CommittedParticipantReceipt {
    CommittedParticipantReceipt {
        generation: CheckpointGeneration::new(
            CheckpointEpoch::new(1),
            ContentDigest::from_bytes([0x77; 32]),
        ),
        participant_id: prepared.descriptor.participant_id.clone(),
        descriptor_digest: prepared
            .descriptor
            .digest()
            .expect("test descriptor is serializable"),
        represented_cut: prepared.descriptor.represented_cut.clone(),
    }
}

async fn checkpoint_payload(bytes: Bytes) -> BudgetedCheckpointBytes {
    let budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: bytes.len().max(1),
    })
    .expect("valid test budget");
    let lease = budget
        .acquire(1, bytes.len())
        .await
        .expect("checkpoint payload budget");
    BudgetedCheckpointBytes::new(bytes, lease).expect("exact payload charge")
}

pub struct CountingParticipant {
    participant_id: CheckpointParticipantId,
    items: u64,
    initialization: ParticipantInitialization,
    released_items: u64,
    committed_descriptor: Option<ContentDigest>,
}

impl CountingParticipant {
    pub fn new(participant_id: &str, items: u64) -> Self {
        Self {
            participant_id: CheckpointParticipantId::new(participant_id),
            items,
            initialization: ParticipantInitialization::default(),
            released_items: 0,
            committed_descriptor: None,
        }
    }

    pub fn released_items(&self) -> u64 {
        self.released_items
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for CountingParticipant {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        let bytes = Bytes::from(self.items.to_le_bytes().to_vec());
        PreparedParticipantState::new(
            self.participant_id.clone(),
            "test.counting",
            1,
            barrier.cut.clone(),
            self.items,
            checkpoint_payload(bytes).await,
        )
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()?;
        if let Some(state) = state {
            let state = state.verify()?;
            let bytes: [u8; 8] = state
                .payload
                .bytes
                .as_ref()
                .try_into()
                .map_err(|_| CheckpointError::ObjectVerification)?;
            self.items = u64::from_le_bytes(bytes);
        }
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.participant_id != self.participant_id {
            return Err(CheckpointError::ParticipantSetMismatch);
        }
        if self.committed_descriptor.as_ref() == Some(&receipt.descriptor_digest) {
            return Ok(());
        }
        self.released_items = self.items;
        self.committed_descriptor = Some(receipt.descriptor_digest);
        Ok(())
    }
}
