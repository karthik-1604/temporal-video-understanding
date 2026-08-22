import pytest
import torch

from src.models.backbones import build_backbone


def test_returns_correct_feature_dim_for_resnet18():
    backbone, feature_dim = build_backbone("resnet18", pretrained=False)

    assert feature_dim == 512
    out = backbone(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 512)


def test_returns_correct_feature_dim_for_efficientnet_b0():
    backbone, feature_dim = build_backbone("efficientnet_b0", pretrained=False)

    assert feature_dim == 1280
    out = backbone(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 1280)


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        build_backbone("not_a_real_backbone", pretrained=False)
