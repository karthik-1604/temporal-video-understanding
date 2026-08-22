"""Baseline 3: CLIP zero-shot classification, no training. Closes the JD's
LLM/VLM gap (see 02_Sony_AI_Video_Analysis_Research.md's "JD-gap additions") —
slots into the same results table as Baselines 1/2 as the vision-language
comparison point.

Split into pure logic (prompt construction, cosine-similarity classification —
tested locally with no real model) and the real CLIP wrapper (`open_clip`
lazy-imported, only runs where the actual pretrained weights are available,
i.e. inside a Kaggle kernel), same pattern as src/data/video_io.py.
"""

from __future__ import annotations

import re
from typing import Sequence

import torch
import torch.nn.functional as F


def camel_case_to_words(name: str) -> str:
    """"ApplyEyeMakeup" -> "apply eye makeup". Handles all-caps runs and
    digits as their own tokens (e.g. "YoYo" -> "yo yo").
    """
    tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", name)
    return " ".join(t.lower() for t in tokens)


def class_name_to_prompt(class_name: str) -> str:
    return f"a video of a person {camel_case_to_words(class_name)}"


def cosine_classify(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    """image_features: (N, D), text_features: (C, D) -> similarity logits (N, C).
    Both are L2-normalized internally, so callers don't need pre-normalized
    embeddings and input scale doesn't affect the result.
    """
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    return image_features @ text_features.T


class CLIPZeroShotClassifier:
    """Thin wrapper around a real `open_clip` model. Only instantiable where
    `open_clip` and pretrained weights are available (Kaggle) — the class body
    above (prompt construction, cosine_classify) is what's unit-tested locally.
    """

    def __init__(
        self,
        class_names: Sequence[str],
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        import open_clip  # local import: only required where real CLIP runs

        self.device = device
        self.class_names = list(class_names)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        prompts = [class_name_to_prompt(name) for name in self.class_names]
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(tokens)

    def classify_clip(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (T, C, H, W), already CLIP-preprocessed -> (num_classes,) logits."""
        with torch.no_grad():
            image_features = self.model.encode_image(frames.to(self.device))
            clip_feature = image_features.mean(dim=0, keepdim=True)
            logits = cosine_classify(clip_feature, self.text_features)
        return logits.squeeze(0)
