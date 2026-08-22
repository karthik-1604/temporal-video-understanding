import pytest
import torch

from src.models.baseline_cnn_rnn import CNNRNNClassifier

# pretrained=False everywhere: architecture/wiring checks only, random weights,
# no real data or downloaded checkpoints.


def test_output_shape_lstm():
    model = CNNRNNClassifier(num_classes=10, backbone="resnet18", pretrained=False, rnn_type="lstm")
    clips = torch.randn(2, 4, 3, 32, 32)

    logits = model(clips)

    assert logits.shape == (2, 10)


def test_output_shape_gru():
    model = CNNRNNClassifier(num_classes=6, backbone="resnet18", pretrained=False, rnn_type="gru")
    clips = torch.randn(2, 5, 3, 32, 32)

    logits = model(clips)

    assert logits.shape == (2, 6)


def test_bidirectional_lstm_shape():
    model = CNNRNNClassifier(
        num_classes=6, backbone="resnet18", pretrained=False, rnn_type="lstm",
        bidirectional=True, hidden_dim=32,
    )
    clips = torch.randn(2, 4, 3, 32, 32)

    logits = model(clips)

    assert logits.shape == (2, 6)


def test_multi_layer_rnn_shape():
    model = CNNRNNClassifier(
        num_classes=4, backbone="resnet18", pretrained=False, rnn_type="gru",
        num_layers=2, hidden_dim=16,
    )
    clips = torch.randn(1, 3, 3, 32, 32)

    logits = model(clips)

    assert logits.shape == (1, 4)


def test_unknown_rnn_type_raises():
    with pytest.raises(ValueError):
        CNNRNNClassifier(num_classes=10, backbone="resnet18", pretrained=False, rnn_type="not_real")


def test_rejects_non_5d_input():
    model = CNNRNNClassifier(num_classes=10, backbone="resnet18", pretrained=False)
    clips = torch.randn(2, 3, 32, 32)

    with pytest.raises(ValueError):
        model(clips)


def test_freeze_backbone_disables_backbone_grad():
    model = CNNRNNClassifier(
        num_classes=10, backbone="resnet18", pretrained=False, freeze_backbone=True
    )

    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.rnn.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_gradient_flows_to_rnn_and_classifier():
    model = CNNRNNClassifier(num_classes=5, backbone="resnet18", pretrained=False)
    clips = torch.randn(1, 3, 3, 32, 32)

    logits = model(clips)
    logits.sum().backward()

    rnn_grads = [p.grad for p in model.rnn.parameters()]
    classifier_grads = [p.grad for p in model.classifier.parameters()]
    assert all(g is not None and torch.any(g != 0) for g in rnn_grads)
    assert all(g is not None and torch.any(g != 0) for g in classifier_grads)


def test_forward_from_features_matches_forward():
    model = CNNRNNClassifier(num_classes=7, backbone="resnet18", pretrained=False)
    model.eval()
    clips = torch.randn(2, 4, 3, 32, 32)

    with torch.no_grad():
        batch, num_frames, channels, height, width = clips.shape
        frames = clips.reshape(batch * num_frames, channels, height, width)
        features = model.backbone(frames).reshape(batch, num_frames, -1)

        logits_from_clips = model(clips)
        logits_from_features = model.forward_from_features(features)

    assert torch.allclose(logits_from_clips, logits_from_features)


def test_forward_from_features_shape():
    model = CNNRNNClassifier(num_classes=9, backbone="resnet18", pretrained=False, hidden_dim=32)
    features = torch.randn(3, 6, 512)  # resnet18 feature_dim = 512

    logits = model.forward_from_features(features)

    assert logits.shape == (3, 9)


def test_sensitive_to_frame_order():
    """The whole point of Baseline 2 over Baseline 1: shuffling frame order
    should change the prediction, since an RNN (unlike average pooling) is
    order-sensitive.
    """
    model = CNNRNNClassifier(num_classes=20, backbone="resnet18", pretrained=False)
    model.eval()
    clips = torch.randn(1, 8, 3, 32, 32)
    reversed_clips = clips.flip(dims=[1])

    with torch.no_grad():
        logits_forward = model(clips)
        logits_reversed = model(reversed_clips)

    assert not torch.allclose(logits_forward, logits_reversed)
