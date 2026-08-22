import pytest
import torch

from src.models.baseline_frame_pool import FramePoolClassifier

# pretrained=False everywhere here: local tests only check the architecture's
# wiring (shapes, gradient flow, freezing) with random weights, never real data
# or downloaded pretrained checkpoints.


def test_output_shape_matches_batch_and_num_classes():
    model = FramePoolClassifier(num_classes=10, backbone="resnet18", pretrained=False)
    clips = torch.randn(2, 4, 3, 32, 32)

    logits = model(clips)

    assert logits.shape == (2, 10)


def test_rejects_non_5d_input():
    model = FramePoolClassifier(num_classes=10, backbone="resnet18", pretrained=False)
    clips = torch.randn(2, 3, 32, 32)

    with pytest.raises(ValueError):
        model(clips)


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        FramePoolClassifier(num_classes=10, backbone="not_a_real_backbone", pretrained=False)


def test_freeze_backbone_disables_backbone_grad():
    model = FramePoolClassifier(
        num_classes=10, backbone="resnet18", pretrained=False, freeze_backbone=True
    )

    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_unfrozen_backbone_keeps_grad():
    model = FramePoolClassifier(
        num_classes=10, backbone="resnet18", pretrained=False, freeze_backbone=False
    )

    assert all(p.requires_grad for p in model.backbone.parameters())


def test_gradient_flows_to_classifier():
    model = FramePoolClassifier(num_classes=5, backbone="resnet18", pretrained=False)
    clips = torch.randn(1, 3, 3, 32, 32)

    logits = model(clips)
    logits.sum().backward()

    grads = [p.grad for p in model.classifier.parameters()]
    assert all(g is not None and torch.any(g != 0) for g in grads)


def test_efficientnet_backbone_produces_correct_shape():
    model = FramePoolClassifier(num_classes=7, backbone="efficientnet_b0", pretrained=False)
    clips = torch.randn(1, 2, 3, 64, 64)

    logits = model(clips)

    assert logits.shape == (1, 7)


def test_single_frame_clip_still_averages_correctly():
    model = FramePoolClassifier(num_classes=3, backbone="resnet18", pretrained=False)
    clips = torch.randn(2, 1, 3, 32, 32)

    logits = model(clips)

    assert logits.shape == (2, 3)
