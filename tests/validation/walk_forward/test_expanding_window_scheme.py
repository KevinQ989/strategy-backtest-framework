from __future__ import annotations
import pytest
import pandas as pd
from strategy_backtester.validation import ExpandingWindowScheme


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scheme():
    """win_in=100, win_out=50: first OOS window ends at index 149."""
    return ExpandingWindowScheme(win_in=100, win_out=50)


# ---------------------------------------------------------------------------
# Empty result when data too short for one complete fold
# ---------------------------------------------------------------------------

def test_split_returns_empty_when_too_short(price_df):
    """win_in + win_out > len(dates) -> no complete fold -> empty list."""
    scheme = ExpandingWindowScheme(win_in=200, win_out=200)
    # price_df has 300 dates; 200+200=400 > 300 -> no folds
    assert scheme.split(price_df) == []


def test_split_returns_empty_when_exactly_win_in_days(price_df):
    """Exactly win_in days of data: no room for any OOS window."""
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    win_in = len(dates)
    scheme = ExpandingWindowScheme(win_in=win_in, win_out=50)
    assert scheme.split(price_df) == []


# ---------------------------------------------------------------------------
# Fold count
# ---------------------------------------------------------------------------

def test_split_fold_count_exact_fit(price_df):
    """
    With 300 dates, win_in=100, win_out=50:
    fold k requires index win_in + k*win_out - 1 < 300
    k_max = floor((300 - 100) / 50) = 4
    """
    scheme = ExpandingWindowScheme(win_in=100, win_out=50)
    folds = scheme.split(price_df)
    assert len(folds) == 4


def test_split_fold_count_with_remainder(price_df):
    """
    Partial final OOS window is dropped.
    With 300 dates, win_in=100, win_out=70:
    floor((300-100)/70) = 2, remainder 60 < 70 -> 2 folds, not 3
    """
    scheme = ExpandingWindowScheme(win_in=100, win_out=70)
    folds = scheme.split(price_df)
    assert len(folds) == 2


def test_split_single_fold_when_exactly_win_in_plus_win_out(price_df):
    """Exactly win_in + win_out dates -> exactly one fold."""
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    n = len(dates)
    win_in = n // 2
    win_out = n - win_in
    scheme = ExpandingWindowScheme(win_in=win_in, win_out=win_out)
    folds = scheme.split(price_df)
    assert len(folds) == 1


# ---------------------------------------------------------------------------
# IS window: anchored at dates[0], grows each fold
# ---------------------------------------------------------------------------

def test_split_is_start_always_first_date(price_df):
    """IS start is anchored at the first trading date for every fold."""
    scheme = ExpandingWindowScheme(win_in=100, win_out=50)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for is_start, _, _, _ in folds:
        assert is_start == dates[0]


def test_split_is_end_grows_by_win_out_each_fold(price_df):
    """is_end should step forward by exactly win_out trading days each fold."""
    win_out = 50
    scheme = ExpandingWindowScheme(win_in=100, win_out=win_out)
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


# ---------------------------------------------------------------------------
# OOS window: fixed length win_out, steps forward each fold
# ---------------------------------------------------------------------------

def test_split_oos_length_is_win_out_for_all_folds(price_df):
    """Every OOS window must span exactly win_out trading days."""
    win_out = 50
    scheme = ExpandingWindowScheme(win_in=100, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for _, _, oos_start, oos_end in folds:
        start_idx = dates.get_loc(oos_start)
        end_idx = dates.get_loc(oos_end)
        assert end_idx - start_idx + 1 == win_out


def test_split_fold1_oos_starts_at_win_in(price_df):
    """Fold 1 OOS start should be at index win_in."""
    win_in, win_out = 100, 50
    scheme = ExpandingWindowScheme(win_in=win_in, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    _, _, oos_start, _ = folds[0]
    assert oos_start == dates[win_in]


# ---------------------------------------------------------------------------
# No gap and no overlap between IS end and OOS start
# ---------------------------------------------------------------------------

def test_split_oos_start_is_day_after_is_end(price_df):
    """oos_start must immediately follow is_end with no gap."""
    scheme = ExpandingWindowScheme(win_in=100, win_out=50)
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
    scheme = ExpandingWindowScheme(win_in=100, win_out=50)
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
    scheme = ExpandingWindowScheme(win_in=100, win_out=50)
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
    scheme = ExpandingWindowScheme(win_in=100, win_out=50)
    folds = scheme.split(price_df)
    assert isinstance(folds, list)
    for fold in folds:
        assert isinstance(fold, tuple)
        assert len(fold) == 4
        assert all(isinstance(ts, pd.Timestamp) for ts in fold)


def test_split_chronological_order(price_df):
    """Within each fold: is_start <= is_end < oos_start <= oos_end."""
    scheme = ExpandingWindowScheme(win_in=100, win_out=50)
    folds = scheme.split(price_df)
    for is_start, is_end, oos_start, oos_end in folds:
        assert is_start <= is_end
        assert is_end < oos_start
        assert oos_start <= oos_end