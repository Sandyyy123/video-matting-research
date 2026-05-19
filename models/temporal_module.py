"""
Recurrent temporal refinement head for video matting.

RecurrentMattingHead takes per-frame encoder features and a GRU hidden state,
and produces a refined alpha matte together with an updated hidden state.
This enables temporally coherent predictions without requiring optical flow
at inference time.

Architecture sketch:
    features (B, C_in, H, W)  +  hidden (B, C_h, H/s, W/s)
        |
    [spatial pooling / projection]
        |
    ConvGRU cell  ->  h_new (B, C_h, H/s, W/s)
        |
    [upsample + decode]
        |
    alpha (B, 1, H, W)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ConvGRU cell  (spatially-aware recurrence at reduced resolution)
# ---------------------------------------------------------------------------

class ConvGRUCell(nn.Module):
    """Convolutional GRU cell operating on 2-D feature maps.

    Replaces the matrix multiply in a standard GRU with 2-D convolutions,
    allowing the hidden state to preserve spatial structure.

    Args:
        input_dim:   Number of channels in the input feature map.
        hidden_dim:  Number of channels in the hidden state.
        kernel_size: Convolution kernel size (spatial).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2

        # Reset and update gates combined into one conv for efficiency
        self.gates = nn.Conv2d(
            input_dim + hidden_dim,
            2 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )
        # Candidate hidden state
        self.candidate = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features  (B, input_dim, H, W)
            h: Hidden state    (B, hidden_dim, H, W)

        Returns:
            h_new: Updated hidden state (B, hidden_dim, H, W)
        """
        combined = torch.cat([x, h], dim=1)  # (B, input_dim+hidden_dim, H, W)

        gates = torch.sigmoid(self.gates(combined))
        r, z = gates.chunk(2, dim=1)  # reset, update  each (B, hidden_dim, H, W)

        candidate_input = torch.cat([x, r * h], dim=1)
        n = torch.tanh(self.candidate(candidate_input))

        h_new = (1.0 - z) * h + z * n
        return h_new


# ---------------------------------------------------------------------------
# Lightweight feature projection + spatial downsampler
# ---------------------------------------------------------------------------

class FeatureAdapter(nn.Module):
    """Projects backbone features to the hidden dimension used by ConvGRU.

    Also optionally downsamples if the backbone feature stride differs from
    the hidden state stride.

    Args:
        in_channels:  Backbone output channels.
        out_channels: Target channels (= ConvGRU hidden_dim).
        downsample_factor: 1 = no downsample, 2 = 2x spatial reduction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_factor: int = 1,
    ) -> None:
        super().__init__()
        stride = downsample_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Decoder head (recurrent hidden state -> alpha)
# ---------------------------------------------------------------------------

class MattingDecoder(nn.Module):
    """Decodes GRU hidden state into an alpha matte at full resolution.

    Applies a series of bilinear upsampling + convolution blocks, with
    optional skip connections from encoder features.

    Args:
        hidden_dim:      Input channel count (= ConvGRU hidden_dim).
        decoder_channels: Channel counts for each upsampling stage.
        upsample_factor: Total spatial upscaling needed (product of 2x steps).
    """

    def __init__(
        self,
        hidden_dim: int,
        decoder_channels: list[int],
        upsample_factor: int = 4,
    ) -> None:
        super().__init__()
        n_steps = int(torch.tensor(upsample_factor).float().log2().round().item())

        layers: list[nn.Module] = []
        in_ch = hidden_dim
        for i, out_ch in enumerate(decoder_channels):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=min(16, out_ch), num_channels=out_ch),
                    nn.GELU(),
                )
            )
            if i < n_steps:
                layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            in_ch = out_ch

        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: GRU hidden state (B, hidden_dim, H/s, W/s)

        Returns:
            alpha: (B, 1, H_full, W_full) in [0, 1]
        """
        feat = self.body(h)
        return torch.sigmoid(self.head(feat))


# ---------------------------------------------------------------------------
# RecurrentMattingHead  (public API)
# ---------------------------------------------------------------------------

class RecurrentMattingHead(nn.Module):
    """GRU-based temporal refinement head for video matting.

    Wraps ConvGRU + decoder into a single module.  Designed to slot in
    after any backbone / feature-pyramid encoder.

    Args:
        in_channels:     Encoder feature channels (e.g. 256 from FPN).
        hidden_dim:      ConvGRU hidden state channels.
        num_layers:      Number of stacked ConvGRU cells.
        decoder_channels: Per-stage channel counts in the upsampling decoder.
        feature_stride:  Spatial stride of encoder features relative to input.
                         E.g. stride=4 means features are 4x smaller than input.

    Input / Output::

        features:  (B, in_channels, H/stride, W/stride)
        hidden:    (B, hidden_dim, H/stride, W/stride) or None (auto-init zeros)
        --
        alpha:     (B, 1, H, W) in [0, 1]
        hidden:    (B, hidden_dim, H/stride, W/stride)  <- pass back next frame
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        decoder_channels: Optional[list[int]] = None,
        feature_stride: int = 4,
    ) -> None:
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [128, 64, 32, 16]

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.feature_stride = feature_stride

        # Project encoder features to hidden_dim
        self.adapter = FeatureAdapter(in_channels, hidden_dim)

        # Stack of ConvGRU cells
        self.gru_cells = nn.ModuleList(
            [ConvGRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Decoder: hidden state -> full-res alpha
        self.decoder = MattingDecoder(
            hidden_dim=hidden_dim,
            decoder_channels=decoder_channels,
            upsample_factor=feature_stride,
        )

    def init_hidden(
        self,
        batch_size: int,
        feature_h: int,
        feature_w: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Return a list of zero hidden states (one per GRU layer)."""
        return [
            torch.zeros(batch_size, self.hidden_dim, feature_h, feature_w, device=device)
            for _ in range(self.num_layers)
        ]

    def forward(
        self,
        features: torch.Tensor,
        hidden: Optional[list[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            features: (B, in_channels, Hf, Wf) encoder output
            hidden:   List of per-layer hidden states, or None for zero-init.

        Returns:
            alpha:      (B, 1, H, W) in [0, 1]
            new_hidden: Updated list of hidden states (pass back on next frame)
        """
        B, _, Hf, Wf = features.shape

        if hidden is None:
            hidden = self.init_hidden(B, Hf, Wf, features.device)

        x = self.adapter(features)  # (B, hidden_dim, Hf, Wf)

        new_hidden: list[torch.Tensor] = []
        for i, cell in enumerate(self.gru_cells):
            x = cell(x, hidden[i])
            new_hidden.append(x)

        alpha = self.decoder(x)  # (B, 1, H, W)
        return alpha, new_hidden

    def process_sequence(
        self,
        frame_features: torch.Tensor,
        hidden: Optional[list[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """Convenience method: process a temporal batch of frames.

        Args:
            frame_features: (B, T, C, Hf, Wf) — T frames per clip
            hidden:         Initial hidden state list (or None)

        Returns:
            alphas:     (B, T, 1, H, W)
            last_hidden: Hidden state after the final frame
        """
        B, T, C, Hf, Wf = frame_features.shape
        alphas: list[torch.Tensor] = []

        for t in range(T):
            alpha_t, hidden = self.forward(frame_features[:, t], hidden)
            alphas.append(alpha_t)

        return torch.stack(alphas, dim=1), hidden  # type: ignore[return-value]
