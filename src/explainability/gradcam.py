"""Grad-CAM for FramePoolClassifier (Baseline 1): per-frame class-activation
heatmaps showing which spatial regions of each frame drove the prediction.
Hooks the backbone's last conv block, so the mechanics (activation/gradient
capture, weighting, normalization) are fully testable locally with a random-
weight model and random input — real semantics only matter once run against
real frames (Kaggle-only, see kaggle_kernel/explainability/).
"""

from __future__ import annotations

from typing import Optional

import torch

from src.models.baseline_frame_pool import FramePoolClassifier


class GradCAM:
    def __init__(self, model: FramePoolClassifier, target_layer_name: str = "layer4"):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        modules = dict(model.backbone.named_modules())
        if target_layer_name not in modules:
            raise ValueError(
                f"No layer {target_layer_name!r} in backbone; available top-level "
                f"modules: {sorted(n for n in modules if '.' not in n and n)}"
            )
        target_layer = modules[target_layer_name]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(
        self, clip: torch.Tensor, target_class: Optional[int] = None
    ) -> tuple[torch.Tensor, int]:
        """clip: (1, num_frames, channels, height, width) -> (heatmaps, predicted_class).
        heatmaps: (num_frames, h, w) in [0, 1], one per frame, spatial size matching
        the target layer's output (e.g. 7x7 for resnet18 layer4 on 224x224 input).
        """
        if clip.ndim != 5 or clip.shape[0] != 1:
            raise ValueError(f"Expected (1, num_frames, C, H, W), got shape {tuple(clip.shape)}")

        self.model.zero_grad()
        logits = self.model(clip)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (num_frames, C, 1, 1)
        cam = torch.relu((weights * self.activations).sum(dim=1))  # (num_frames, h, w)

        peak = cam.amax(dim=(1, 2), keepdim=True)
        cam = cam / (peak + 1e-8)

        return cam.detach(), target_class
