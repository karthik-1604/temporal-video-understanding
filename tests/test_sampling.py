import pytest

from src.data.sampling import (
    random_frame_indices,
    sample_frame_indices,
    uniform_frame_indices,
)


def test_uniform_spans_full_range_and_is_sorted():
    indices = uniform_frame_indices(total_frames=100, num_frames=8)
    assert len(indices) == 8
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert indices[-1] == 99


def test_uniform_exact_match_returns_every_frame():
    indices = uniform_frame_indices(total_frames=8, num_frames=8)
    assert indices == list(range(8))


def test_uniform_pads_short_clip_by_repeating_last_frame():
    indices = uniform_frame_indices(total_frames=3, num_frames=8)
    assert len(indices) == 8
    assert indices[:3] == [0, 1, 2]
    assert indices[3:] == [2] * 5


def test_uniform_is_deterministic():
    a = uniform_frame_indices(total_frames=50, num_frames=16)
    b = uniform_frame_indices(total_frames=50, num_frames=16)
    assert a == b


@pytest.mark.parametrize("total_frames,num_frames", [(0, 8), (-1, 8), (10, 0), (10, -1)])
def test_invalid_counts_raise(total_frames, num_frames):
    with pytest.raises(ValueError):
        uniform_frame_indices(total_frames, num_frames)


def test_random_returns_sorted_unique_indices_within_range():
    indices = random_frame_indices(total_frames=100, num_frames=16, seed=0)
    assert len(indices) == 16
    assert len(set(indices)) == 16
    assert indices == sorted(indices)
    assert all(0 <= i < 100 for i in indices)


def test_random_is_reproducible_given_same_seed():
    a = random_frame_indices(total_frames=100, num_frames=16, seed=7)
    b = random_frame_indices(total_frames=100, num_frames=16, seed=7)
    assert a == b


def test_random_differs_across_seeds_in_general():
    a = random_frame_indices(total_frames=100, num_frames=16, seed=1)
    b = random_frame_indices(total_frames=100, num_frames=16, seed=2)
    assert a != b


def test_random_pads_short_clip_same_as_uniform():
    indices = random_frame_indices(total_frames=3, num_frames=8, seed=0)
    assert indices[:3] == [0, 1, 2]
    assert indices[3:] == [2] * 5


def test_sample_frame_indices_dispatches_by_strategy():
    uniform = sample_frame_indices(100, 8, strategy="uniform")
    assert uniform == uniform_frame_indices(100, 8)

    random_result = sample_frame_indices(100, 8, strategy="random", seed=0)
    assert random_result == random_frame_indices(100, 8, seed=0)


def test_sample_frame_indices_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        sample_frame_indices(100, 8, strategy="not_a_real_strategy")
