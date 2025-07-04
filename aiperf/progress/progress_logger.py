# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging

from tqdm import tqdm

from aiperf.progress.progress_tracker import ProgressTracker


class SimpleProgressLogger:
    """Simple logger for progress updates. It will use tqdm to show a progress bar."""

    def __init__(self, progress_tracker: ProgressTracker):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.progress_tracker = progress_tracker
        self.tqdm_requests: tqdm | None = None
        self.tqdm_records: tqdm | None = None

    async def update_progress(self):
        """Log a progress update."""
        cur_profile = self.progress_tracker.current_profile
        if cur_profile is None:
            return

        # Handle case where total_expected_requests might be None
        total_requests = cur_profile.total_expected_requests or 0
        completed_requests = cur_profile.requests_completed

        self.logger.debug(
            "Requests Completed: %d / %d",
            completed_requests,
            total_requests,
        )

        # Only create tqdm if we have a valid total > 0
        if self.tqdm_requests is None and total_requests > 0:
            self.tqdm_requests = tqdm(
                total=total_requests,
                desc="Requests Completed",
                colour="green",
            )

        if self.tqdm_requests is not None:
            self.tqdm_requests.n = completed_requests
            self.tqdm_requests.refresh()

        # Close tqdm when completed
        if (
            total_requests > 0
            and completed_requests >= total_requests
            and self.tqdm_requests is not None
        ):
            self.tqdm_requests.close()
            self.tqdm_requests = None

    async def update_stats(self):
        """Log a stats update."""
        cur_profile = self.progress_tracker.current_profile
        if cur_profile is None:
            return

        # Handle case where total_expected_requests might be None
        total_requests = cur_profile.total_expected_requests or 0
        processed_requests = cur_profile.requests_processed

        self.logger.debug(
            "Records Processed: %d / %d",
            processed_requests,
            total_requests,
        )

        # Only create tqdm if we have a valid total > 0
        if self.tqdm_records is None and total_requests > 0:
            self.tqdm_records = tqdm(
                total=total_requests,
                desc=" Records Processed",
                colour="blue",
            )

        if self.tqdm_records is not None:
            self.tqdm_records.n = processed_requests
            self.tqdm_records.refresh()

        # Close tqdm when completed
        if (
            total_requests > 0
            and processed_requests >= total_requests
            and self.tqdm_records is not None
        ):
            self.logger.debug(
                "Closing TQDM. Records Processed: %d / %d",
                processed_requests,
                total_requests,
            )
            self.tqdm_records.close()
            self.tqdm_records = None

    async def update_results(self):
        """Log a results update."""
        if self.tqdm_requests is not None:
            self.tqdm_requests.close()
            self.tqdm_requests = None

        if self.tqdm_records is not None:
            self.tqdm_records.close()
            self.tqdm_records = None
