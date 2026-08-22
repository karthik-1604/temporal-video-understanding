import torch

from src.models.clip_zero_shot import camel_case_to_words, class_name_to_prompt, cosine_classify


def test_camel_case_splits_simple_words():
    assert camel_case_to_words("ApplyEyeMakeup") == "apply eye makeup"
    assert camel_case_to_words("ApplyLipstick") == "apply lipstick"
    assert camel_case_to_words("BasketballDunk") == "basketball dunk"


def test_camel_case_handles_repeated_capitals():
    assert camel_case_to_words("YoYo") == "yo yo"


def test_camel_case_handles_all_caps_acronym_prefix():
    assert camel_case_to_words("HDVideo") == "hd video"


def test_camel_case_handles_single_word():
    assert camel_case_to_words("Archery") == "archery"


def test_camel_case_handles_digits():
    assert camel_case_to_words("Punch2Kick") == "punch 2 kick"


def test_class_name_to_prompt_wraps_words():
    assert class_name_to_prompt("ApplyEyeMakeup") == "a video of a person apply eye makeup"


def test_cosine_classify_shape():
    image_features = torch.randn(5, 16)
    text_features = torch.randn(10, 16)

    logits = cosine_classify(image_features, text_features)

    assert logits.shape == (5, 10)


def test_cosine_classify_picks_matching_direction():
    text_features = torch.eye(4)  # 4 orthonormal "classes"
    image_features = torch.eye(4)[[2, 0, 3]]  # samples matching classes 2, 0, 3

    logits = cosine_classify(image_features, text_features)

    assert logits.argmax(dim=1).tolist() == [2, 0, 3]


def test_cosine_classify_is_scale_invariant():
    image_features = torch.randn(3, 8)
    text_features = torch.randn(6, 8)

    logits_a = cosine_classify(image_features, text_features)
    logits_b = cosine_classify(image_features * 100.0, text_features * 0.01)

    assert torch.allclose(logits_a, logits_b, atol=1e-4)
