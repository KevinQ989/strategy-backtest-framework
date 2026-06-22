from __future__ import annotations
from strategy_backtester.validation import RollingWindowScheme


def test_split_is_length_is_win_in_for_all_folds(price_df):
    """Every IS window must span exactly win_in trading days."""
    win_in, win_out = 100, 50
    scheme = RollingWindowScheme(win_in=win_in, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for is_start, is_end, _, _ in folds:
        start_idx = dates.get_loc(is_start)
        end_idx = dates.get_loc(is_end)
        assert end_idx - start_idx + 1 == win_in


def test_split_fold1_is_starts_at_dates_0(price_df):
    """Fold 1 IS start must be the first trading date."""
    win_in, win_out = 100, 50
    scheme = RollingWindowScheme(win_in=win_in, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    is_start, _, _, _ = folds[0]
    assert is_start == dates[0]


def test_split_is_start_slides_by_win_out_each_fold(price_df):
    """IS start index should increase by exactly win_out at each fold."""
    win_in, win_out = 100, 50
    scheme = RollingWindowScheme(win_in=win_in, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for i in range(1, len(folds)):
        prev_is_start = folds[i - 1][0]
        curr_is_start = folds[i][0]
        prev_idx = dates.get_loc(prev_is_start)
        curr_idx = dates.get_loc(curr_is_start)
        assert curr_idx - prev_idx == win_out
        

def test_split_is_window_length_constant_across_folds(price_df):
    """
    The key invariant of rolling vs expanding: IS window size must be
    identical across all folds, not grow with each iteration.
    """
    win_in, win_out = 100, 50
    scheme = RollingWindowScheme(win_in=win_in, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    is_lengths = []
    for is_start, is_end, _, _ in folds:
        start_idx = dates.get_loc(is_start)
        end_idx = dates.get_loc(is_end)
        is_lengths.append(end_idx - start_idx + 1)
    assert all(length == win_in for length in is_lengths)