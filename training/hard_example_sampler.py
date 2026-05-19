"""
Hard-example mining / replay buffer for video matting training.

Tracks per-sample loss over a rolling window of epochs.  Samples whose
mean loss exceeds (global_mean + threshold_std * global_std) are flagged
as "hard" and stored in a FIFO buffer.  At training time a configurable
fraction of each batch is replaced with hard examples drawn from the buffer.
"""

from __future__ import annotations

import collections
import logging
import random
from typing import Any, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HardExampleSampler:
    """Tracks per-sample loss history and provides a hard-example replay buffer.

    Usage::

        sampler = HardExampleSampler(buffer_size=5000, loss_window=10)

        # After each forward pass in the training loop:
        sampler.update(sample_ids, per_sample_losses)

        # At the start of the next epoch, fetch hard samples to mix into batches:
        hard_ids = sampler.sample_hard(n=32)

    Args:
        buffer_size:         Maximum number of hard examples to keep (FIFO).
        loss_window:         Rolling window size (in epochs) for smoothing
                             per-sample loss estimates.
        upsample_threshold_std: Samples with loss > mean + N*std are "hard".
        upsample_multiplier: How many extra times to re-add a hard sample
                             to the buffer each time it is flagged.
    """

    def __init__(
        self,
        buffer_size: int = 5000,
        loss_window: int = 10,
        upsample_threshold_std: float = 1.0,
        upsample_multiplier: int = 3,
    ) -> None:
        self.buffer_size = buffer_size
        self.loss_window = loss_window
        self.upsample_threshold_std = upsample_threshold_std
        self.upsample_multiplier = upsample_multiplier

        # sample_id -> deque of recent loss values (length <= loss_window)
        self._loss_history: dict[Any, collections.deque] = {}

        # FIFO buffer of hard sample ids
        self._buffer: collections.deque = collections.deque(maxlen=buffer_size)

        self._epoch: int = 0

    # ------------------------------------------------------------------
    # Recording losses
    # ------------------------------------------------------------------

    def update(
        self,
        sample_ids: list[Any],
        losses: "np.ndarray | list[float]",
    ) -> None:
        """Record per-sample losses for the current step.

        Args:
            sample_ids: List of hashable sample identifiers (e.g. file paths
                        or integer dataset indices).
            losses:     Per-sample scalar loss values, same length as sample_ids.
        """
        if len(sample_ids) != len(losses):
            raise ValueError(
                f"sample_ids ({len(sample_ids)}) and losses ({len(losses)}) "
                f"must have the same length."
            )

        for sid, loss_val in zip(sample_ids, losses):
            if sid not in self._loss_history:
                self._loss_history[sid] = collections.deque(maxlen=self.loss_window)
            self._loss_history[sid].append(float(loss_val))

    # ------------------------------------------------------------------
    # Epoch-level buffer refresh
    # ------------------------------------------------------------------

    def end_epoch(self) -> int:
        """Call once per epoch to refresh the hard-example buffer.

        Computes mean loss per sample over the rolling window, finds
        hard examples (loss > mean + std * threshold), and adds them
        (possibly multiple times) to the FIFO buffer.

        Returns:
            Number of unique hard samples added this epoch.
        """
        if not self._loss_history:
            return 0

        # Aggregate mean loss per sample
        mean_losses: dict[Any, float] = {
            sid: float(np.mean(dq)) for sid, dq in self._loss_history.items()
        }

        values = np.array(list(mean_losses.values()), dtype=np.float32)
        global_mean = float(values.mean())
        global_std = float(values.std())
        threshold = global_mean + self.upsample_threshold_std * global_std

        hard_samples = [
            sid for sid, loss in mean_losses.items() if loss > threshold
        ]

        for sid in hard_samples:
            for _ in range(self.upsample_multiplier):
                self._buffer.append(sid)

        self._epoch += 1
        logger.debug(
            "HardExampleSampler epoch %d: %d hard samples (thresh=%.4f), "
            "buffer size=%d",
            self._epoch,
            len(hard_samples),
            threshold,
            len(self._buffer),
        )
        return len(hard_samples)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_hard(self, n: int) -> list[Any]:
        """Draw up to `n` sample ids from the hard-example buffer.

        Args:
            n: Number of samples to draw (with replacement if buffer < n).

        Returns:
            List of sample ids.  May be empty if the buffer is empty.
        """
        if not self._buffer:
            return []

        buf_list = list(self._buffer)
        if n >= len(buf_list):
            return random.choices(buf_list, k=n)
        return random.sample(buf_list, k=n)

    def __len__(self) -> int:
        """Current number of entries in the hard-example buffer."""
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "epoch": self._epoch,
            "loss_history": {
                sid: list(dq) for sid, dq in self._loss_history.items()
            },
            "buffer": list(self._buffer),
        }

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._loss_history = {
            sid: collections.deque(vals, maxlen=self.loss_window)
            for sid, vals in state["loss_history"].items()
        }
        self._buffer = collections.deque(state["buffer"], maxlen=self.buffer_size)
        logger.info(
            "HardExampleSampler restored: epoch=%d, buffer_size=%d, "
            "tracked_samples=%d",
            self._epoch,
            len(self._buffer),
            len(self._loss_history),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def top_k_hardest(self, k: int = 20) -> list[tuple[Any, float]]:
        """Return the top-k hardest samples by mean loss (descending)."""
        if not self._loss_history:
            return []
        mean_losses = {
            sid: float(np.mean(dq)) for sid, dq in self._loss_history.items()
        }
        sorted_items = sorted(mean_losses.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    def loss_percentile(self, percentile: float = 90.0) -> float:
        """Return the loss value at a given percentile across all tracked samples."""
        if not self._loss_history:
            return 0.0
        means = [float(np.mean(dq)) for dq in self._loss_history.values()]
        return float(np.percentile(means, percentile))
