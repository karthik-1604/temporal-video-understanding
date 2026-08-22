import pytest

from src.data.config import VideoDatasetConfig


def test_defaults_are_applied():
    cfg = VideoDatasetConfig(
        dataset_name="ucf101", root_dir="/fake/root", split="train", num_frames=16
    )
    assert cfg.sampling_strategy == "uniform"
    assert cfg.image_size == 224
    assert cfg.seed == 42


def test_rejects_unknown_sampling_strategy():
    with pytest.raises(ValueError):
        VideoDatasetConfig(
            dataset_name="ucf101",
            root_dir="/fake/root",
            split="train",
            num_frames=16,
            sampling_strategy="not_a_real_strategy",
        )


def test_rejects_non_positive_num_frames():
    with pytest.raises(ValueError):
        VideoDatasetConfig(
            dataset_name="ucf101", root_dir="/fake/root", split="train", num_frames=0
        )


def test_from_yaml_round_trips(tmp_path):
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(
        "dataset_name: ucf101\n"
        "root_dir: /fake/root\n"
        "split: val\n"
        "num_frames: 8\n"
        "sampling_strategy: random\n"
        "image_size: 112\n"
        "seed: 7\n",
        encoding="utf-8",
    )

    cfg = VideoDatasetConfig.from_yaml(yaml_path)

    assert cfg.dataset_name == "ucf101"
    assert cfg.split == "val"
    assert cfg.num_frames == 8
    assert cfg.sampling_strategy == "random"
    assert cfg.image_size == 112
    assert cfg.seed == 7


def test_from_yaml_still_validates(tmp_path):
    yaml_path = tmp_path / "bad_dataset.yaml"
    yaml_path.write_text(
        "dataset_name: ucf101\n"
        "root_dir: /fake/root\n"
        "split: train\n"
        "num_frames: -5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        VideoDatasetConfig.from_yaml(yaml_path)


def test_real_project_config_file_loads():
    cfg = VideoDatasetConfig.from_yaml("configs/dataset/ucf101.yaml")
    assert cfg.dataset_name == "ucf101"
    assert cfg.num_frames == 16
