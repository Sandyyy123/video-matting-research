"""
Curriculum learning scheduler for video matting training.

Stages progress from easiest (synthetic clean) to hardest (real noisy),
with configurable warmup epochs per stage and optional hard-example replay
injected into each batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage.

    Attributes:
        name:               Human-readable label.
        warmup_epochs:      How many epochs to spend in this stage.
                            None means "run until total budget exhausted".
        noise_level:        Additive Gaussian noise std applied to inputs.
        motion_blur:        Whether to apply motion-blur augmentation.
        synthetic_ratio:    Fraction of each batch drawn from synthetic data.
        real_ratio:         Fraction of each batch drawn from real data.
        extra_aug_params:   Any extra augmentation kwargs forwarded to the
                            dataset / collator.
    """
    name: str
    warmup_epochs: Optional[int]
    noise_level: float = 0.0
    motion_blur: bool = False
    synthetic_ratio: float = 1.0
    real_ratio: float = 0.0
    extra_aug_params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = self.synthetic_ratio + self.real_ratio
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Stage '{self.name}': synthetic_ratio + real_ratio must sum to 1.0 "
                f"(got {total:.4f})."
            )


class CurriculumScheduler:
    """Manages progression through curriculum stages during training.

    Usage::

        scheduler = CurriculumScheduler.from_config(cfg["curriculum"])
        for epoch in range(total_epochs):
            stage = scheduler.current_stage
            train_one_epoch(loader, stage, hard_sampler=scheduler.hard_sampler)
            scheduler.step()

    Args:
        stages:                 Ordered list of CurriculumStage objects.
        hard_example_replay_ratio: Fraction of each batch replaced with
                                   hard examples from the replay buffer.
        total_epochs:           Total training budget; used to allocate
                                epochs to the last (open-ended) stage.
    """

    def __init__(
        self,
        stages: list[CurriculumStage],
        hard_example_replay_ratio: float = 0.15,
        total_epochs: int = 120,
    ) -> None:
        if not stages:
            raise ValueError("At least one curriculum stage required.")

        self.stages = stages
        self.hard_example_replay_ratio = hard_example_replay_ratio
        self.total_epochs = total_epochs

        # Resolve open-ended warmup for the last stage
        self._stage_budgets = self._allocate_budgets()
        self._epoch: int = 0
        self._stage_idx: int = 0

        logger.info(
            "CurriculumScheduler initialized with %d stages: %s",
            len(stages),
            [f"{s.name}({b}ep)" for s, b in zip(stages, self._stage_budgets)],
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "CurriculumScheduler":
        """Build from a YAML-parsed config dict.

        Expected structure mirrors configs/base_config.yaml::

            curriculum:
              stages:
                - name: synthetic_clean
                  warmup_epochs: 10
                  ...
              hard_example_replay_ratio: 0.15
        """
        stages = [
            CurriculumStage(
                name=s["name"],
                warmup_epochs=s.get("warmup_epochs"),
                noise_level=s.get("noise_level", 0.0),
                motion_blur=s.get("motion_blur", False),
                synthetic_ratio=s.get("synthetic_ratio", 1.0),
                real_ratio=s.get("real_ratio", 0.0),
                extra_aug_params=s.get("extra_aug_params", {}),
            )
            for s in cfg["stages"]
        ]
        return cls(
            stages=stages,
            hard_example_replay_ratio=cfg.get("hard_example_replay_ratio", 0.15),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _allocate_budgets(self) -> list[int]:
        """Assign an epoch count to every stage, resolving None last stage."""
        budgets: list[int] = []
        fixed_total = sum(
            s.warmup_epochs for s in self.stages if s.warmup_epochs is not None
        )
        open_count = sum(1 for s in self.stages if s.warmup_epochs is None)

        remainder = max(0, self.total_epochs - fixed_total)

        for s in self.stages:
            if s.warmup_epochs is not None:
                budgets.append(s.warmup_epochs)
            else:
                budgets.append(remainder // max(open_count, 1))

        return budgets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> CurriculumStage:
        """The active CurriculumStage for the current epoch."""
        return self.stages[self._stage_idx]

    @property
    def current_epoch(self) -> int:
        return self._epoch

    @property
    def stage_index(self) -> int:
        return self._stage_idx

    @property
    def epochs_in_current_stage(self) -> int:
        """How many epochs have elapsed since entering this stage."""
        elapsed = self._epoch - sum(self._stage_budgets[: self._stage_idx])
        return max(0, elapsed)

    @property
    def stage_progress(self) -> float:
        """Fractional progress [0, 1] through the current stage."""
        budget = self._stage_budgets[self._stage_idx]
        if budget == 0:
            return 1.0
        return min(1.0, self.epochs_in_current_stage / budget)

    def get_augmentation_params(self) -> dict[str, Any]:
        """Return a flat dict of augmentation parameters for the dataset."""
        stage = self.current_stage
        return {
            "noise_level": stage.noise_level,
            "motion_blur": stage.motion_blur,
            "synthetic_ratio": stage.synthetic_ratio,
            "real_ratio": stage.real_ratio,
            **stage.extra_aug_params,
        }

    def step(self) -> None:
        """Advance one epoch. Logs stage transitions."""
        self._epoch += 1

        # Check if we should advance to the next stage
        epochs_at_boundary = sum(self._stage_budgets[: self._stage_idx + 1])
        if (
            self._stage_idx < len(self.stages) - 1
            and self._epoch >= epochs_at_boundary
        ):
            self._stage_idx += 1
            logger.info(
                "Curriculum: advancing to stage [%d] '%s' at epoch %d",
                self._stage_idx,
                self.stages[self._stage_idx].name,
                self._epoch,
            )

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "stage_idx": self._stage_idx}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._stage_idx = state["stage_idx"]
        logger.info(
            "CurriculumScheduler restored: epoch=%d, stage='%s'",
            self._epoch,
            self.current_stage.name,
        )
