"""
Custom loss functions for video matting.

Includes:
  - TemporalConsistencyLoss  : penalizes alpha flicker across adjacent frames
  - BoundaryRefinementLoss   : up-weights gradients near alpha transitions (Sobel)
  - TriMapGuidedLoss         : region-aware L1 + BCE based on trimap labeling
  - MultiTaskMattingLoss     : combines the above with Kendall uncertainty weighting
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sobel_edge_mask(alpha: torch.Tensor, dilation_px: int = 3) -> torch.Tensor:
    """Return a float mask [0,1] that is high near alpha transitions.

    Args:
        alpha: Float tensor of shape (B, 1, H, W) with values in [0, 1].
        dilation_px: How many pixels to dilate the detected edge.

    Returns:
        edge_mask: Float tensor (B, 1, H, W) in [0, 1].
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=alpha.dtype, device=alpha.device
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    grad_x = F.conv2d(alpha, sobel_x, padding=1)
    grad_y = F.conv2d(alpha, sobel_y, padding=1)
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    # Normalize to [0, 1] per sample
    b = magnitude.shape[0]
    flat = magnitude.view(b, -1)
    max_val = flat.max(dim=1).values.view(b, 1, 1, 1).clamp(min=1e-6)
    edge_mask = (magnitude / max_val).clamp(0.0, 1.0)

    if dilation_px > 0:
        k = 2 * dilation_px + 1
        dilate_kernel = torch.ones(1, 1, k, k, dtype=alpha.dtype, device=alpha.device)
        edge_mask = (
            F.conv2d(edge_mask, dilate_kernel, padding=dilation_px).clamp(0.0, 1.0)
        )

    return edge_mask


def _warp_by_flow(tensor: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Bilinear warp `tensor` using an optical flow field.

    Args:
        tensor: (B, C, H, W)
        flow:   (B, 2, H, W) in pixel units (dx, dy)

    Returns:
        Warped tensor (B, C, H, W).
    """
    B, C, H, W = tensor.shape
    device = tensor.device

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)

    # Displaced grid in pixel coords
    displaced = base_grid + flow  # (B, 2, H, W)

    # Normalize to [-1, 1]
    displaced[:, 0] = 2.0 * displaced[:, 0] / (W - 1) - 1.0
    displaced[:, 1] = 2.0 * displaced[:, 1] / (H - 1) - 1.0

    grid = displaced.permute(0, 2, 3, 1)  # (B, H, W, 2)
    return F.grid_sample(tensor, grid, mode="bilinear", align_corners=True, padding_mode="border")


# ---------------------------------------------------------------------------
# TemporalConsistencyLoss
# ---------------------------------------------------------------------------

class TemporalConsistencyLoss(nn.Module):
    """Penalize alpha matte flickering between adjacent frames.

    Given predicted alpha mattes at times t and t+1 and the optical flow
    from t to t+1, the loss warps alpha_t into the coordinate frame of t+1
    and computes an L1 difference against alpha_{t+1}.

    Args:
        reduction: "mean" | "sum" | "none"
        occlusion_threshold: Flow magnitude above which pixels are treated as
            occluded (and masked out from the loss).
    """

    def __init__(
        self,
        reduction: str = "mean",
        occlusion_threshold: float = 20.0,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.occlusion_threshold = occlusion_threshold

    def forward(
        self,
        alpha_t: torch.Tensor,
        alpha_t1: torch.Tensor,
        flow_t_to_t1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            alpha_t:        (B, 1, H, W) predicted alpha at time t
            alpha_t1:       (B, 1, H, W) predicted alpha at time t+1
            flow_t_to_t1:   (B, 2, H, W) optical flow (pixel units).
                            If None, computes naive frame-diff (no warp).

        Returns:
            Scalar loss tensor.
        """
        if flow_t_to_t1 is not None:
            alpha_t_warped = _warp_by_flow(alpha_t, flow_t_to_t1)

            # Build occlusion mask: mask out large-motion regions
            flow_magnitude = flow_t_to_t1.norm(dim=1, keepdim=True)  # (B, 1, H, W)
            valid_mask = (flow_magnitude < self.occlusion_threshold).float()
        else:
            alpha_t_warped = alpha_t
            valid_mask = torch.ones_like(alpha_t)

        diff = torch.abs(alpha_t_warped - alpha_t1) * valid_mask

        if self.reduction == "mean":
            n_valid = valid_mask.sum().clamp(min=1.0)
            return diff.sum() / n_valid
        elif self.reduction == "sum":
            return diff.sum()
        else:
            return diff


# ---------------------------------------------------------------------------
# BoundaryRefinementLoss
# ---------------------------------------------------------------------------

class BoundaryRefinementLoss(nn.Module):
    """Apply higher loss weight near detected alpha transitions (Sobel-based).

    Pixels near the alpha boundary tend to be the hardest to predict
    accurately. This loss derives an edge mask from the *ground truth* alpha
    and up-weights the per-pixel L1 reconstruction loss in that region.

    Args:
        edge_weight_factor: Multiplier applied to edge-region pixels.
        dilation_px: Dilation radius for the edge mask.
        reduction: "mean" | "sum"
    """

    def __init__(
        self,
        edge_weight_factor: float = 5.0,
        dilation_px: int = 3,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.edge_weight_factor = edge_weight_factor
        self.dilation_px = dilation_px
        self.reduction = reduction

    def forward(
        self,
        pred_alpha: torch.Tensor,
        gt_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_alpha: (B, 1, H, W) predicted alpha in [0, 1]
            gt_alpha:   (B, 1, H, W) ground-truth alpha in [0, 1]

        Returns:
            Scalar boundary-aware loss.
        """
        with torch.no_grad():
            edge_mask = _sobel_edge_mask(gt_alpha, self.dilation_px)

        # Per-pixel weight: edges get extra weight, flat regions get weight=1
        pixel_weight = 1.0 + (self.edge_weight_factor - 1.0) * edge_mask

        diff = torch.abs(pred_alpha - gt_alpha) * pixel_weight

        if self.reduction == "mean":
            return diff.mean()
        else:
            return diff.sum()


# ---------------------------------------------------------------------------
# TriMapGuidedLoss
# ---------------------------------------------------------------------------

class TriMapGuidedLoss(nn.Module):
    """Region-aware loss that treats unknown / fg / bg differently.

    - Unknown region (trimap == 0.5): L1 loss, upweighted
    - Known FG (trimap == 1): binary cross-entropy against gt
    - Known BG (trimap == 0): binary cross-entropy against gt

    Trimap convention:
        0   -> definite background
        0.5 -> unknown / transition
        1   -> definite foreground

    Args:
        unknown_weight: Extra multiplier on L1 in the unknown region.
        bce_weight:     Weight for fg/bg BCE term relative to unknown L1.
        unknown_tol:    Pixels with |trimap - 0.5| < tol are "unknown".
    """

    def __init__(
        self,
        unknown_weight: float = 2.0,
        bce_weight: float = 0.5,
        unknown_tol: float = 0.1,
    ) -> None:
        super().__init__()
        self.unknown_weight = unknown_weight
        self.bce_weight = bce_weight
        self.unknown_tol = unknown_tol

    def forward(
        self,
        pred_alpha: torch.Tensor,
        gt_alpha: torch.Tensor,
        trimap: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_alpha: (B, 1, H, W) in [0, 1]
            gt_alpha:   (B, 1, H, W) in [0, 1]
            trimap:     (B, 1, H, W) with values in {0, 0.5, 1}

        Returns:
            Scalar combined loss.
        """
        unknown_mask = (torch.abs(trimap - 0.5) < self.unknown_tol).float()
        known_mask = 1.0 - unknown_mask

        # L1 in unknown region
        l1_unknown = (
            torch.abs(pred_alpha - gt_alpha) * unknown_mask * self.unknown_weight
        )
        n_unknown = unknown_mask.sum().clamp(min=1.0)
        loss_unknown = l1_unknown.sum() / n_unknown

        # BCE in known fg/bg region
        pred_clamped = pred_alpha.clamp(1e-6, 1 - 1e-6)
        bce_known = F.binary_cross_entropy(pred_clamped, gt_alpha, reduction="none")
        bce_known = (bce_known * known_mask).sum() / known_mask.sum().clamp(min=1.0)

        return loss_unknown + self.bce_weight * bce_known


# ---------------------------------------------------------------------------
# MultiTaskMattingLoss  (Kendall et al. uncertainty weighting)
# ---------------------------------------------------------------------------

class MultiTaskMattingLoss(nn.Module):
    """Combines all three matting losses using learnable uncertainty weighting.

    Following Kendall et al. (2018) "Multi-Task Learning Using Uncertainty
    to Weigh Losses in Deep Learning":

        L_total = sum_i [ exp(-s_i) * L_i + s_i ]

    where s_i = log(sigma_i^2) are learnable log-variance parameters.

    Args:
        temporal_cfg:   kwargs passed to TemporalConsistencyLoss
        boundary_cfg:   kwargs passed to BoundaryRefinementLoss
        trimap_cfg:     kwargs passed to TriMapGuidedLoss
        init_log_var:   Initial value for log-variance scalars.
    """

    def __init__(
        self,
        temporal_cfg: Optional[dict] = None,
        boundary_cfg: Optional[dict] = None,
        trimap_cfg: Optional[dict] = None,
        init_log_var: float = 0.0,
    ) -> None:
        super().__init__()

        self.temporal_loss = TemporalConsistencyLoss(**(temporal_cfg or {}))
        self.boundary_loss = BoundaryRefinementLoss(**(boundary_cfg or {}))
        self.trimap_loss = TriMapGuidedLoss(**(trimap_cfg or {}))

        # Learnable log-variance for each task (Kendall uncertainty)
        self.log_var_temporal = nn.Parameter(torch.tensor(init_log_var))
        self.log_var_boundary = nn.Parameter(torch.tensor(init_log_var))
        self.log_var_trimap = nn.Parameter(torch.tensor(init_log_var))

    def _uncertainty_weight(
        self, loss: torch.Tensor, log_var: nn.Parameter
    ) -> torch.Tensor:
        """Apply Kendall weighting: exp(-s)*L + s."""
        return torch.exp(-log_var) * loss + log_var

    def forward(
        self,
        pred_alpha_t: torch.Tensor,
        pred_alpha_t1: torch.Tensor,
        gt_alpha_t: torch.Tensor,
        trimap_t: torch.Tensor,
        flow_t_to_t1: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred_alpha_t:   (B, 1, H, W) prediction at frame t
            pred_alpha_t1:  (B, 1, H, W) prediction at frame t+1
            gt_alpha_t:     (B, 1, H, W) ground truth at frame t
            trimap_t:       (B, 1, H, W) trimap at frame t
            flow_t_to_t1:   (B, 2, H, W) optional optical flow

        Returns:
            Dict with keys "total", "temporal", "boundary", "trimap",
            "log_var_temporal", "log_var_boundary", "log_var_trimap".
        """
        l_temporal = self.temporal_loss(pred_alpha_t, pred_alpha_t1, flow_t_to_t1)
        l_boundary = self.boundary_loss(pred_alpha_t, gt_alpha_t)
        l_trimap = self.trimap_loss(pred_alpha_t, gt_alpha_t, trimap_t)

        total = (
            self._uncertainty_weight(l_temporal, self.log_var_temporal)
            + self._uncertainty_weight(l_boundary, self.log_var_boundary)
            + self._uncertainty_weight(l_trimap, self.log_var_trimap)
        )

        return {
            "total": total,
            "temporal": l_temporal.detach(),
            "boundary": l_boundary.detach(),
            "trimap": l_trimap.detach(),
            "log_var_temporal": self.log_var_temporal.detach(),
            "log_var_boundary": self.log_var_boundary.detach(),
            "log_var_trimap": self.log_var_trimap.detach(),
        }
