from __future__ import annotations
import pandas as pd
from strategy_backtester.validation import RollingWindowScheme


# ---------------------------------------------------------------------------
# Empty result when data too short for one complete fold
# ---------------------------------------------------------------------------

def test_split_returns_empty_when_too_short(price_df):
    """win_in + win_out > len(dates) -> no complete fold -> empty list."""
    scheme = RollingWindowScheme(win_in=200, win_out=200)
    assert scheme.split(price_df) == []


def test_split_returns_empty_when_exactly_win_in_days(price_df):
    """Exactly win_in days of data: no room for any OOS window."""
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    win_in = len(dates)
    scheme = RollingWindowScheme(win_in=win_in, win_out=50)
    assert scheme.split(price_df) == []


# ---------------------------------------------------------------------------
# Fold count
# ---------------------------------------------------------------------------

def test_split_fold_count_exact_fit(price_df):
    """
    With 300 dates, win_in=100, win_out=50:
    fold k requires (k-1)*50 + 100 + 50 - 1 < 300
    k_max = floor((300 - 100) / 50) = 4
    """
    scheme = RollingWindowScheme(win_in=100, win_out=50)
    folds = scheme.split(price_df)
    assert len(folds) == 4


def test_split_fold_count_with_remainder(price_df):
    """
    Partial final OOS window is dropped.
    With 300 dates, win_in=100, win_out=70:
    floor((300-100)/70) = 2, remainder 60 < 70 -> 2 folds, not 3.
    """
    scheme = RollingWindowScheme(win_in=100, win_out=70)
    folds = scheme.split(price_df)
    assert len(folds) == 2


def test_split_single_fold_when_exactly_win_in_plus_win_out(price_df):
    """Exactly win_in + win_out dates -> exactly one fold."""
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    n = len(dates)
    win_in = n // 2
    win_out = n - win_in
    scheme = RollingWindowScheme(win_in=win_in, win_out=win_out)
    folds = scheme.split(price_df)
    assert len(folds) == 1


# ---------------------------------------------------------------------------
# IS window: fixed length win_in, slides by win_out each fold
# ---------------------------------------------------------------------------

def test_split_is_length_is_win_in_for_all_folds(price_df):
    """Every IS window must span exactly win_in trading days."""
    win_in = 100
    scheme = RollingWindowScheme(win_in=win_in, win_out=50)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for is_start, is_end, _, _ in folds:
        start_idx = dates.get_loc(is_start)
        end_idx = dates.get_loc(is_end)
        assert end_idx - start_idx + 1 == win_in


def test_split_fold1_is_starts_at_dates_0(price_df):
    """Fold 1 IS start must be the first trading date."""
    scheme = RollingWindowScheme(win_in=100, win_out=50)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    is_start, _, _, _ = folds[0]
    assert is_start == dates[0]


def test_split_is_start_slides_by_win_out_each_fold(price_df):
    """IS start index should increase by exactly win_out at each fold."""
    win_out = 50
    scheme = RollingWindowScheme(win_in=100, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for i in range(1, len(folds)):
        prev_is_start = folds[i - 1][0]
        curr_is_start = folds[i][0]
        prev_idx = dates.get_loc(prev_is_start)
        curr_idx = dates.get_loc(curr_is_start)
        assert curr_idx - prev_idx == win_out


def test_split_is_start_differs_from_expanding(price_df):
    """
    Unlike expanding scheme, rolling IS start is NOT anchored at dates[0]
    for folds beyond the first.
    """
    scheme = RollingWindowScheme(win_in=100, win_out=50)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    # Fold 2 onwards: IS start must have moved forward
    for is_start, _, _, _ in folds[1:]:
        assert is_start != dates[0]


# ---------------------------------------------------------------------------
# OOS window: fixed length win_out, steps forward each fold
# ---------------------------------------------------------------------------

def test_split_oos_length_is_win_out_for_all_folds(price_df):
    """Every OOS window must span exactly win_out trading days."""
    win_out = 50
    scheme = RollingWindowScheme(win_in=100, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for _, _, oos_start, oos_end in folds:
        start_idx = dates.get_loc(oos_start)
        end_idx = dates.get_loc(oos_end)
        assert end_idx - start_idx + 1 == win_out


def test_split_fold1_oos_starts_at_win_in(price_df):
    """Fold 1 OOS start must be at index win_in."""
    win_in, win_out = 100, 50
    scheme = RollingWindowScheme(win_in=win_in, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    _, _, oos_start, _ = folds[0]
    assert oos_start == dates[win_in]


# ---------------------------------------------------------------------------
# No gap and no overlap between IS end and OOS start
# ---------------------------------------------------------------------------

def test_split_oos_start_is_day_after_is_end(price_df):
    """oos_start must immediately follow is_end with no gap."""
    scheme = RollingWindowScheme(win_in=100, win_out=50)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for _, is_end, oos_start, _ in folds:
        is_end_idx = dates.get_loc(is_end)
        oos_start_idx = dates.get_loc(oos_start)
        assert oos_start_idx == is_end_idx + 1


# ---------------------------------------------------------------------------
# No overlap between consecutive fold OOS windows
# ---------------------------------------------------------------------------

def test_split_consecutive_oos_windows_do_not_overlap(price_df):
    """OOS windows across folds must be non-overlapping and chronological."""
    scheme = RollingWindowScheme(win_in=100, win_out=50)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for i in range(1, len(folds)):
        prev_oos_end = folds[i - 1][3]
        curr_oos_start = folds[i][2]
        prev_idx = dates.get_loc(prev_oos_end)
        curr_idx = dates.get_loc(curr_oos_start)
        assert curr_idx == prev_idx + 1


# ---------------------------------------------------------------------------
# All boundary dates exist in the price index
# ---------------------------------------------------------------------------

def test_split_all_boundaries_in_price_index(price_df):
    """Every boundary date returned must be a real trading date in prices."""
    scheme = RollingWindowScheme(win_in=100, win_out=50)
    trading_dates = set(price_df.index.get_level_values("Date").unique())
    folds = scheme.split(price_df)
    for is_start, is_end, oos_start, oos_end in folds:
        assert is_start in trading_dates
        assert is_end in trading_dates
        assert oos_start in trading_dates
        assert oos_end in trading_dates


# ---------------------------------------------------------------------------
# Returned tuples have correct structure
# ---------------------------------------------------------------------------

def test_split_returns_list_of_4_tuples(price_df):
    scheme = RollingWindowScheme(win_in=100, win_out=50)
    folds = scheme.split(price_df)
    assert isinstance(folds, list)
    for fold in folds:
        assert isinstance(fold, tuple)
        assert len(fold) == 4
        assert all(isinstance(ts, pd.Timestamp) for ts in fold)


def test_split_chronological_order(price_df):
    """Within each fold: is_start <= is_end < oos_start <= oos_end."""
    scheme = RollingWindowScheme(win_in=100, win_out=50)
    folds = scheme.split(price_df)
    for is_start, is_end, oos_start, oos_end in folds:
        assert is_start <= is_end
        assert is_end < oos_start
        assert oos_start <= oos_end


# ---------------------------------------------------------------------------
# Scheme-specific: IS window does NOT grow (unlike expanding)
# ---------------------------------------------------------------------------

def test_split_is_window_length_constant_across_folds(price_df):
    """
    The key invariant of rolling vs expanding: IS window size must be
    identical across all folds, not grow with each iteration.
    """
    win_in = 100
    scheme = RollingWindowScheme(win_in=win_in, win_out=50)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    is_lengths = []
    for is_start, is_end, _, _ in folds:
        start_idx = dates.get_loc(is_start)
        end_idx = dates.get_loc(is_end)
        is_lengths.append(end_idx - start_idx + 1)
    assert all(length == win_in for length in is_lengths)