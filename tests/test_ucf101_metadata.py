import pytest

from src.data.ucf101_metadata import (
    build_class_index,
    load_split_metadata,
    load_ucf101_splits,
    resolve_video_path,
)

TRAIN_CSV = """clip_name,clip_path,label
v_Swing_g05_c02,/train/Swing/v_Swing_g05_c02.avi,Swing
v_ApplyEyeMakeup_g01_c01,/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi,ApplyEyeMakeup
v_Archery_g02_c03,/train/Archery/v_Archery_g02_c03.avi,Archery
"""

TEST_CSV = """clip_name,clip_path,label
v_Swing_g21_c02,/test/Swing/v_Swing_g21_c02.avi,Swing
v_Archery_g10_c01,/test/Archery/v_Archery_g10_c01.avi,Archery
"""


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_build_class_index_is_sorted_and_deduplicated():
    index = build_class_index(["Swing", "Archery", "Swing", "ApplyEyeMakeup"])
    assert index == {"ApplyEyeMakeup": 0, "Archery": 1, "Swing": 2}


def test_load_split_metadata_parses_rows(tmp_path):
    csv_path = _write(tmp_path, "train.csv", TRAIN_CSV)
    class_to_index = {"ApplyEyeMakeup": 0, "Archery": 1, "Swing": 2}

    samples = load_split_metadata(csv_path, class_to_index)

    assert len(samples) == 3
    swing = next(s for s in samples if s.clip_name == "v_Swing_g05_c02")
    assert swing.class_name == "Swing"
    assert swing.label == 2
    # leading slash in clip_path stripped so it composes cleanly with root_dir
    assert swing.relative_path == "train/Swing/v_Swing_g05_c02.avi"


def test_load_split_metadata_rejects_unknown_class(tmp_path):
    csv_path = _write(tmp_path, "train.csv", TRAIN_CSV)
    class_to_index = {"Swing": 0}  # missing ApplyEyeMakeup, Archery

    with pytest.raises(ValueError):
        load_split_metadata(csv_path, class_to_index)


def test_load_ucf101_splits_shares_label_ids_across_splits(tmp_path):
    _write(tmp_path, "train.csv", TRAIN_CSV)
    _write(tmp_path, "test.csv", TEST_CSV)

    splits = load_ucf101_splits(tmp_path)

    assert set(splits.keys()) == {"train", "test"}
    assert len(splits["train"]) == 3
    assert len(splits["test"]) == 2

    train_swing = next(s for s in splits["train"] if s.class_name == "Swing")
    test_swing = next(s for s in splits["test"] if s.class_name == "Swing")
    assert train_swing.label == test_swing.label


def test_load_ucf101_splits_raises_when_no_csvs_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ucf101_splits(tmp_path)


def test_load_ucf101_splits_falls_back_when_index_source_missing(tmp_path):
    _write(tmp_path, "test.csv", TEST_CSV)

    splits = load_ucf101_splits(tmp_path, class_index_source="train")

    assert set(splits.keys()) == {"test"}
    assert len(splits["test"]) == 2


def test_resolve_video_path_joins_root_and_relative_path(tmp_path):
    csv_path = _write(tmp_path, "train.csv", TRAIN_CSV)
    class_to_index = {"ApplyEyeMakeup": 0, "Archery": 1, "Swing": 2}
    sample = load_split_metadata(csv_path, class_to_index)[0]

    resolved = resolve_video_path("/kaggle/input/ucf101", sample)

    assert str(resolved).replace("\\", "/") == "/kaggle/input/ucf101/train/Swing/v_Swing_g05_c02.avi"
