from __future__ import annotations
from strategy_backtester.validation import ExpandingWindowScheme


def test_split_is_start_always_first_date(price_df):
    """IS start is anchored at the first trading date for every fold."""
    win_in, win_out = 100, 50
    scheme = ExpandingWindowScheme(win_in=win_in, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for is_start, _, _, _ in folds:
        assert is_start == dates[0]


def test_split_is_end_grows_by_win_out_each_fold(price_df):
    """is_end should step forward by exactly win_out trading days each fold."""
    win_in, win_out = 100, 50
    scheme = ExpandingWindowScheme(win_in=win_in, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for i in range(1, len(folds)):
        prev_is_end = folds[i - 1][1]
        curr_is_end = folds[i][1]
        prev_idx = dates.get_loc(prev_is_end)
        curr_idx = dates.get_loc(curr_is_end)
        assert curr_idx - prev_idx == win_out


def test_split_fold1_is_end_at_correct_index(price_df):
    """Fold 1 IS end should be at index win_in - 1."""
    win_in, win_out = 100, 50
    scheme = ExpandingWindowScheme(win_in=win_in, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    _, is_end, _, _ = folds[0]
    assert is_end == dates[win_in - 1]
