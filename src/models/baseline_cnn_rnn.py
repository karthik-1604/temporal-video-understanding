"""Baseline 2: pretrained CNN frame features -> LSTM/GRU over the frame
sequence -> classifier head. Same backbone convention as Baseline 1
(frame_pool), but replaces temporal average pooling with a recurrent model
that can actually use frame *order* — directly testing whether that recovers
accuracy on cases average pooling can't distinguish (e.g. Baseline 1's
Basketball/BasketballDunk confusion, see journey.md Phase 6).
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.backbones import build_backbone

_RNN_TYPES = {"lstm": nn.LSTM, "gru": nn.GRU}


class CNNRNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        rnn_type: str = "lstm",
        hidden_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if rnn_type not in _RNN_TYPES:
            raise ValueError(f"Unknown rnn_type {rnn_type!r}; expected one of {sorted(_RNN_TYPES)}")

        self.backbone, feature_dim = build_backbone(backbone, pretrained)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.rnn = _RNN_TYPES[rnn_type](
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_out_dim, num_classes),
        )

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, num_frames, feature_dim), already backbone-extracted
        (e.g. cached to skip re-running a frozen backbone every epoch) ->
        (batch, num_classes).
        """
        rnn_out, _ = self.rnn(features)
        last_step = rnn_out[:, -1, :]
        return self.classifier(last_step)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """clips: (batch, num_frames, channels, height, width) -> (batch, num_classes)."""
        if clips.ndim != 5:
            raise ValueError(
                f"Expected 5D input (batch, num_frames, channels, height, width), "
                f"got shape {tuple(clips.shape)}"
            )
        batch, num_frames, channels, height, width = clips.shape
        frames = clips.reshape(batch * num_frames, channels, height, width)
        features = self.backbone(frames)
        features = features.reshape(batch, num_frames, -1)
        return self.forward_from_features(features)
