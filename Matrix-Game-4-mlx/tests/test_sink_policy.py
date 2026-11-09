import pytest

from matrix_game_mlx.inference.sink_policy import (FUTURE_SINK_OFFSET, SINK_ROPE_MODES, check_sink_rope_mode,
                                                   rope_positions_needed, rotated_sink_position)


def test_positions():
    window = 21
    assert SINK_ROPE_MODES == ("absolute", "fixed", "future") and FUTURE_SINK_OFFSET == 11
    for position in (0, 1, window):
        for mode in SINK_ROPE_MODES:
            assert rotated_sink_position(mode, position, window) is None
    for position in (window + 1, 1000, 8191):
        assert rotated_sink_position("absolute", position, window) is None
        assert rotated_sink_position("fixed", position, window) == position - window
        assert rotated_sink_position("future", position, window) == position + FUTURE_SINK_OFFSET


def test_rope_rows():
    assert rope_positions_needed("absolute", 8192) == 8192
    assert rope_positions_needed("fixed", 8192) == 8192
    assert rope_positions_needed("future", 8192) == 8192 + FUTURE_SINK_OFFSET


def test_none_is_absolute():
    assert rotated_sink_position(None, 30, 21) is None and rope_positions_needed(None, 5) == 5


def test_invalid():
    with pytest.raises(ValueError):
        rotated_sink_position("dual", 30, 21)
    with pytest.raises(ValueError):
        check_sink_rope_mode("dual")
    with pytest.raises(ValueError):
        rope_positions_needed("dual", 5)
