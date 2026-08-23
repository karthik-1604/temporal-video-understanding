import pytest
import torch

from src.explainability.gradcam import GradCAM
from src.models.baseline_frame_pool import FramePoolClassifier

# pretrained=False: mechanics-only tests (activation/gradient hooks, shapes,
# normalization) with random weights -- no real images needed for these.


def _make_model(num_classes=10, freeze_backbone=True):
    return FramePoolClassifier(
        num_classes=num_classes, backbone="resnet18", pretrained=False,
        freeze_backbone=freeze_backbone,
    )


def test_generate_returns_heatmap_shape_matching_num_frames():
    model = _make_model()
    gradcam = GradCAM(model)
    clip = torch.randn(1, 4, 3, 64, 64, requires_grad=True)

    heatmaps, _ = gradcam.generate(clip)

    assert heatmaps.shape[0] == 4  # one heatmap per frame
    assert heatmaps.shape[1] == heatmaps.shape[2] == 2  # 64 / 32 (resnet18 stride) = 2


def test_heatmap_values_in_unit_range():
    model = _make_model()
    gradcam = GradCAM(model)
    clip = torch.randn(1, 3, 3, 64, 64, requires_grad=True)

    heatmaps, _ = gradcam.generate(clip)

    assert heatmaps.min() >= 0.0
    assert heatmaps.max() <= 1.0 + 1e-6


def test_predicted_class_matches_argmax_when_target_none():
    model = _make_model(num_classes=5)
    model.eval()
    gradcam = GradCAM(model)
    clip = torch.randn(1, 3, 3, 64, 64, requires_grad=True)

    with torch.no_grad():
        expected_class = int(model(clip).argmax(dim=1).item())

    _, pred_class = gradcam.generate(clip)

    assert pred_class == expected_class


def test_explicit_target_class_is_used():
    model = _make_model(num_classes=5)
    gradcam = GradCAM(model)
    clip = torch.randn(1, 3, 3, 64, 64, requires_grad=True)

    _, returned_class = gradcam.generate(clip, target_class=2)

    assert returned_class == 2


def test_rejects_batch_size_other_than_one():
    model = _make_model()
    gradcam = GradCAM(model)
    clip = torch.randn(2, 3, 3, 64, 64, requires_grad=True)

    with pytest.raises(ValueError):
        gradcam.generate(clip)


def test_rejects_non_5d_input():
    model = _make_model()
    gradcam = GradCAM(model)
    clip = torch.randn(1, 3, 64, 64, requires_grad=True)

    with pytest.raises(ValueError):
        gradcam.generate(clip)


def test_unknown_target_layer_raises():
    model = _make_model()
    with pytest.raises(ValueError):
        GradCAM(model, target_layer_name="not_a_real_layer")


def test_works_with_frozen_backbone():
    model = _make_model(num_classes=6, freeze_backbone=True)
    gradcam = GradCAM(model)
    clip = torch.randn(1, 2, 3, 64, 64, requires_grad=True)

    heatmaps, _ = gradcam.generate(clip)

    assert heatmaps.shape[0] == 2


def test_works_with_unfrozen_backbone():
    model = _make_model(num_classes=6, freeze_backbone=False)
    gradcam = GradCAM(model)
    clip = torch.randn(1, 2, 3, 64, 64, requires_grad=True)

    heatmaps, _ = gradcam.generate(clip)

    assert heatmaps.shape[0] == 2
