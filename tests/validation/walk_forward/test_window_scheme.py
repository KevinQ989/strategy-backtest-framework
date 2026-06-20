from __future__ import annotations
import pytest
import pandas as pd
from strategy_backtester.validation import ExpandingWindowScheme, RollingWindowScheme, WindowScheme

schemes = [
    ExpandingWindowScheme, 
    RollingWindowScheme
]

# ---------------------------------------------------------------------------
# Test __init__ validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme_cls", schemes)
def test_init_stores_win_in_and_win_out(scheme_cls):
    scheme = scheme_cls(win_in=100, win_out=50)
    assert scheme.win_in == 100
    assert scheme.win_out == 50


@pytest.mark.parametrize("scheme_cls", schemes)
def test_init_zero_win_in_raises(scheme_cls):
    with pytest.raises(ValueError, match="win_in"):
        scheme_cls(win_in=0, win_out=50)


@pytest.mark.parametrize("scheme_cls", schemes)
def test_init_negative_win_in_raises(scheme_cls):
    with pytest.raises(ValueError, match="win_in"):
        scheme_cls(win_in=-10, win_out=50)


@pytest.mark.parametrize("scheme_cls", schemes)
def test_init_zero_win_out_raises(scheme_cls):
    with pytest.raises(ValueError, match="win_out"):
        scheme_cls(win_in=100, win_out=0)


@pytest.mark.parametrize("scheme_cls", schemes)
def test_init_negative_win_out_raises(scheme_cls):
    with pytest.raises(ValueError, match="win_out"):
        scheme_cls(win_in=100, win_out=-5)


# ---------------------------------------------------------------------------
# Test split common properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme_cls", schemes)
def test_split_returns_empty_when_too_short(scheme_cls, price_df):
    """win_in + win_out > len(dates) -> no complete fold -> empty list."""
    scheme = scheme_cls(win_in=200, win_out=200)
    # price_df has 300 dates; 200+200=400 > 300 -> no folds
    assert scheme.split(price_df) == []


@pytest.mark.parametrize("scheme_cls", schemes)
def test_split_returns_empty_when_exactly_win_in_days(scheme_cls, price_df):
    """Exactly win_in days of data: no room for any OOS window."""
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    win_in = len(dates)
    scheme = scheme_cls(win_in=win_in, win_out=50)
    assert scheme.split(price_df) == []


@pytest.mark.parametrize("scheme_cls", schemes)
def test_split_returns_list(scheme_cls, price_df):
    """
    With 300 dates, win_in=100, win_out=50:
    fold k requires (k-1)*50 + 100 + 50 - 1 < 300
    k_max = floor((300 - 100) / 50) = 4
    """
    win_in, win_out = 100, 50
    scheme = scheme_cls(win_in=win_in, win_out=win_out)
    folds = scheme.split(price_df)
    assert len(folds) == 4


@pytest.mark.parametrize("scheme_cls", schemes)
def test_split_fold_count_with_remainder(scheme_cls, price_df):
    """
    Partial final OOS window is dropped.
    With 300 dates, win_in=100, win_out=70:
    floor((300-100)/70) = 2, remainder 60 < 70 -> 2 folds, not 3.
    """
    win_in, win_out = 100, 70
    scheme = scheme_cls(win_in=win_in, win_out=win_out)
    folds = scheme.split(price_df)
    assert len(folds) == 2


@pytest.mark.parametrize("scheme_cls", schemes)
def test_split_single_fold_when_exactly_win_in_plus_win_out(scheme_cls, price_df):
    """Exactly win_in + win_out dates -> exactly one fold."""
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    n = len(dates)
    win_in = n // 2
    win_out = n - win_in
    scheme = scheme_cls(win_in=win_in, win_out=win_out)
    folds = scheme.split(price_df)
    assert len(folds) == 1


@pytest.mark.parametrize("scheme_cls", schemes)
def test_split_oos_length_is_win_out_for_all_folds(scheme_cls, price_df):
    """Every OOS window must span exactly win_out trading days."""
    win_in, win_out = 100, 50
    scheme = scheme_cls(win_in=win_in, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for _, _, oos_start, oos_end in folds:
        start_idx = dates.get_loc(oos_start)
        end_idx = dates.get_loc(oos_end)
        assert end_idx - start_idx + 1 == win_out


@pytest.mark.parametrize("scheme_cls", schemes)
def test_split_fold1_oos_starts_at_win_in(scheme_cls, price_df):
    """Fold 1 OOS start must be at index win_in."""
    win_in, win_out = 100, 50
    scheme = scheme_cls(win_in=win_in, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    _, _, oos_start, _ = folds[0]
    assert oos_start == dates[win_in]


@pytest.mark.parametrize("scheme_cls", schemes)
def test_split_oos_start_is_day_after_is_end(scheme_cls, price_df):
    """oos_start must immediately follow is_end with no gap."""
    win_in, win_out = 100, 50
    scheme = scheme_cls(win_in=win_in, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for _, is_end, oos_start, _ in folds:
        is_end_idx = dates.get_loc(is_end)
        oos_start_idx = dates.get_loc(oos_start)
        assert oos_start_idx == is_end_idx + 1


@pytest.mark.parametrize("scheme_cls", schemes)
def test_split_consecutive_oos_windows_do_not_overlap(scheme_cls, price_df):
    """OOS windows across folds must be non-overlapping and chronological."""
    win_in, win_out = 100, 50
    scheme = scheme_cls(win_in=win_in, win_out=win_out)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    folds = scheme.split(price_df)
    for i in range(1, len(folds)):
        prev_oos_end = folds[i - 1][3]
        curr_oos_start = folds[i][2]
        prev_idx = dates.get_loc(prev_oos_end)
        curr_idx = dates.get_loc(curr_oos_start)
        assert curr_idx == prev_idx + 1


@pytest.mark.parametrize("scheme_cls", schemes)
def test_split_all_boundaries_in_price_index(scheme_cls, price_df):
    """Every boundary date returned must be a real trading date in prices."""
    win_in, win_out = 100, 50
    scheme = scheme_cls(win_in=win_in, win_out=win_out)
    trading_dates = set(price_df.index.get_level_values("Date").unique())
    folds = scheme.split(price_df)
    for is_start, is_end, oos_start, oos_end in folds:
        assert is_start in trading_dates
        assert is_end in trading_dates
        assert oos_start in trading_dates
        assert oos_end in trading_dates


@pytest.mark.parametrize("scheme_cls", schemes)
def test_split_returns_list_of_4_tuples(scheme_cls, price_df):
    win_in, win_out = 100, 50
    scheme = scheme_cls(win_in=win_in, win_out=win_out)
    folds = scheme.split(price_df)
    assert isinstance(folds, list)
    for fold in folds:
        assert isinstance(fold, tuple)
        assert len(fold) == 4
        assert all(isinstance(ts, pd.Timestamp) for ts in fold)


@pytest.mark.parametrize("scheme_cls", schemes)
def test_split_chronological_order(scheme_cls, price_df):
    """Within each fold: is_start <= is_end < oos_start <= oos_end."""
    win_in, win_out = 100, 50
    scheme = scheme_cls(win_in=win_in, win_out=win_out)
    folds = scheme.split(price_df)
    for is_start, is_end, oos_start, oos_end in folds:
        assert is_start <= is_end
        assert is_end < oos_start
        assert oos_start <= oos_end


# ---------------------------------------------------------------------------
# Test WindowScheme is abstract — cannot be instantiated directly
# ---------------------------------------------------------------------------

def test_window_scheme_is_abstract():
    with pytest.raises(TypeError):
        WindowScheme(win_in=100, win_out=50)  # type: ignore