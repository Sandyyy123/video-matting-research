from .matting_losses import (
    TemporalConsistencyLoss,
    BoundaryRefinementLoss,
    TriMapGuidedLoss,
    MultiTaskMattingLoss,
)

__all__ = [
    "TemporalConsistencyLoss",
    "BoundaryRefinementLoss",
    "TriMapGuidedLoss",
    "MultiTaskMattingLoss",
]
