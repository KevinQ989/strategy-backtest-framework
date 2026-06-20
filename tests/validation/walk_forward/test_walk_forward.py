from __future__ import annotations
import pytest
import pandas as pd
from unittest.mock import patch
from strategy_backtester.core import ParamResult, WalkForwardFold, WalkForwardResult
from strategy_backtester.strategies import CrossSectionalMomentumStrategy
from strategy_backtester.validation import (
    ExpandingWindowScheme,
    RollingWindowScheme,
)
from strategy_backtester.validation.walk_forward import WalkForwardTest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# small_price_df: 50 days, 5 tickers
# win_in=30, win_out=10 -> floor((50-30)/10) = 2 folds
WIN_IN = 30
WIN_OUT = 10

# Param grid whose lookbacks fit within WIN_IN=30
# 2 x 2 = 4 combinations — small enough for tests to complete quickly
PARAM_GRID = {
    "lookback": [20, 25],
    "percent":  [0.4, 0.5],
}

# Fixed strategy params that fit within WIN_IN for single-strategy fixtures
SMALL_PARAMS = dict(lookback=20, skip=1, percent=0.4, rebalance_freq=5)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def expanding_scheme():
    return ExpandingWindowScheme(win_in=WIN_IN, win_out=WIN_OUT)


@pytest.fixture
def rolling_scheme():
    return RollingWindowScheme(win_in=WIN_IN, win_out=WIN_OUT)


@pytest.fixture
def wf_expanding(small_price_df, expanding_scheme):
    return WalkForwardTest(
        prices=small_price_df,
        strategy_cls=CrossSectionalMomentumStrategy,
        param_grid=PARAM_GRID,
        scheme=expanding_scheme,
        metric="sharpe",
        initial_capital=100_000.0,
        n_jobs=1,
    )


@pytest.fixture
def wf_rolling(small_price_df, rolling_scheme):
    return WalkForwardTest(
        prices=small_price_df,
        strategy_cls=CrossSectionalMomentumStrategy,
        param_grid=PARAM_GRID,
        scheme=rolling_scheme,
        metric="sharpe",
        initial_capital=100_000.0,
        n_jobs=1,
    )


@pytest.fixture
def result_expanding(wf_expanding):
    return wf_expanding.run()


@pytest.fixture
def result_rolling(wf_rolling):
    return wf_rolling.run()


# ---------------------------------------------------------------------------
# Test __init__: valid construction
# ---------------------------------------------------------------------------

def test_init_stores_attributes(small_price_df, expanding_scheme):
    wf = WalkForwardTest(
        prices=small_price_df,
        strategy_cls=CrossSectionalMomentumStrategy,
        param_grid=PARAM_GRID,
        scheme=expanding_scheme,
    )
    assert wf.prices is small_price_df
    assert wf.strategy_cls is CrossSectionalMomentumStrategy
    assert wf.scheme is expanding_scheme
    assert wf.metric == "sharpe"
    assert wf.initial_capital == 100_000.0


def test_init_default_metric_and_capital(small_price_df, expanding_scheme):
    wf = WalkForwardTest(
        prices=small_price_df,
        strategy_cls=CrossSectionalMomentumStrategy,
        param_grid=PARAM_GRID,
        scheme=expanding_scheme,
    )
    assert wf.metric == "sharpe"
    assert wf.initial_capital == 100_000.0


def test_init_param_combinations_cartesian_product(small_price_df, expanding_scheme):
    """Cartesian product of PARAM_GRID should produce 2x2=4 combinations."""
    wf = WalkForwardTest(
        prices=small_price_df,
        strategy_cls=CrossSectionalMomentumStrategy,
        param_grid=PARAM_GRID,
        scheme=expanding_scheme,
    )
    assert len(wf.param_combinations) == 4
    # Every combination must contain all keys
    for combo in wf.param_combinations:
        assert set(combo.keys()) == set(PARAM_GRID.keys())


def test_init_n_jobs_minus_one_sets_none(small_price_df, expanding_scheme):
    """n_jobs=-1 should map to None (os.cpu_count() semantics)."""
    wf = WalkForwardTest(
        prices=small_price_df,
        strategy_cls=CrossSectionalMomentumStrategy,
        param_grid=PARAM_GRID,
        scheme=expanding_scheme,
        n_jobs=-1,
    )
    assert wf.n_jobs is None


# ---------------------------------------------------------------------------
# Test __init__: validation errors
# ---------------------------------------------------------------------------

def test_init_raises_on_unsupported_metric(small_price_df, expanding_scheme):
    with pytest.raises(ValueError, match="Unsupported metric"):
        WalkForwardTest(
            prices=small_price_df,
            strategy_cls=CrossSectionalMomentumStrategy,
            param_grid=PARAM_GRID,
            scheme=expanding_scheme,
            metric="calmar",
        )


def test_init_raises_on_empty_param_grid(small_price_df, expanding_scheme):
    with pytest.raises(ValueError, match="param_grid"):
        WalkForwardTest(
            prices=small_price_df,
            strategy_cls=CrossSectionalMomentumStrategy,
            param_grid={},
            scheme=expanding_scheme,
        )


def test_init_raises_when_win_in_less_than_max_lookback(small_price_df):
    """win_in=30 < max(lookback)=35 -> ValueError."""
    scheme = ExpandingWindowScheme(win_in=30, win_out=10)
    with pytest.raises(ValueError, match="win_in"):
        WalkForwardTest(
            prices=small_price_df,
            strategy_cls=CrossSectionalMomentumStrategy,
            param_grid={"lookback": [25, 35], "percent": [0.4]},
            scheme=scheme,
        )


def test_init_raises_when_price_history_too_short(small_price_df):
    """Price history too short for even one fold."""
    scheme = ExpandingWindowScheme(win_in=40, win_out=40)
    # small_price_df has 50 days; 40+40=80 > 50
    with pytest.raises(ValueError, match="too short|insufficient"):
        WalkForwardTest(
            prices=small_price_df,
            strategy_cls=CrossSectionalMomentumStrategy,
            param_grid=PARAM_GRID,
            scheme=scheme,
        )


# ---------------------------------------------------------------------------
# Test _slice_prices
# ---------------------------------------------------------------------------

def test_slice_prices_returns_correct_date_range(wf_expanding, small_price_df):
    dates = small_price_df.index.get_level_values("Date").unique().sort_values()
    start, end = dates[5], dates[14]
    sliced = wf_expanding._slice_prices(start, end)
    sliced_dates = sliced.index.get_level_values("Date").unique().sort_values()
    assert sliced_dates[0] == start
    assert sliced_dates[-1] == end


def test_slice_prices_raises_on_empty_range(wf_expanding):
    with pytest.raises(ValueError):
        wf_expanding._slice_prices(
            pd.Timestamp("1990-01-01"),
            pd.Timestamp("1990-12-31"),
        )


def test_slice_prices_inclusive_boundaries(wf_expanding, small_price_df):
    """Both start and end dates must be included in the slice."""
    dates = small_price_df.index.get_level_values("Date").unique().sort_values()
    start, end = dates[0], dates[-1]
    sliced = wf_expanding._slice_prices(start, end)
    sliced_dates = sliced.index.get_level_values("Date").unique()
    assert start in sliced_dates
    assert end in sliced_dates


# ---------------------------------------------------------------------------
# Test run: result structure
# ---------------------------------------------------------------------------

def test_run_returns_walk_forward_result(result_expanding):
    assert isinstance(result_expanding, WalkForwardResult)


def test_run_scheme_name_matches_scheme_class(result_expanding, result_rolling):
    assert result_expanding.scheme == "ExpandingWindowScheme"
    assert result_rolling.scheme == "RollingWindowScheme"


def test_run_metric_stored_on_result(result_expanding):
    assert result_expanding.metric == "sharpe"


def test_run_fold_count_matches_scheme(
    small_price_df, expanding_scheme, rolling_scheme, result_expanding, result_rolling
):
    expected_expanding = len(expanding_scheme.split(small_price_df))
    expected_rolling = len(rolling_scheme.split(small_price_df))
    assert len(result_expanding.folds) == expected_expanding
    assert len(result_rolling.folds) == expected_rolling


def test_run_folds_are_walk_forward_fold_instances(result_expanding):
    for fold in result_expanding.folds:
        assert isinstance(fold, WalkForwardFold)


# ---------------------------------------------------------------------------
# Test WalkForwardFold structure
# ---------------------------------------------------------------------------

def test_fold_idx_is_1_based(result_expanding):
    for i, fold in enumerate(result_expanding.folds, start=1):
        assert fold.fold_idx == i


def test_fold_dates_are_timestamps(result_expanding):
    for fold in result_expanding.folds:
        assert isinstance(fold.is_start, pd.Timestamp)
        assert isinstance(fold.is_end, pd.Timestamp)
        assert isinstance(fold.oos_start, pd.Timestamp)
        assert isinstance(fold.oos_end, pd.Timestamp)


def test_fold_dates_chronological(result_expanding):
    """Within each fold: is_start <= is_end < oos_start <= oos_end."""
    for fold in result_expanding.folds:
        assert fold.is_start <= fold.is_end
        assert fold.is_end < fold.oos_start
        assert fold.oos_start <= fold.oos_end


def test_fold_param_results_count_matches_grid(result_expanding):
    """Each fold must have one ParamResult per grid combination."""
    n_combinations = 4  # 2 lookbacks x 2 percents
    for fold in result_expanding.folds:
        assert len(fold.param_results) == n_combinations


def test_fold_param_results_are_param_result_instances(result_expanding):
    for fold in result_expanding.folds:
        for pr in fold.param_results:
            assert isinstance(pr, ParamResult)


def test_fold_param_result_keys_match_grid(result_expanding):
    """Every ParamResult.params must have the same keys as PARAM_GRID."""
    for fold in result_expanding.folds:
        for pr in fold.param_results:
            assert set(pr.params.keys()) == set(PARAM_GRID.keys())


def test_fold_param_result_values_from_grid(result_expanding):
    """Every ParamResult.params value must come from the corresponding grid list."""
    for fold in result_expanding.folds:
        for pr in fold.param_results:
            for key, val in pr.params.items():
                assert val in PARAM_GRID[key]


def test_fold_is_metric_is_float(result_expanding):
    for fold in result_expanding.folds:
        for pr in fold.param_results:
            assert isinstance(pr.is_metric, float)


def test_fold_selected_params_keys_match_grid(result_expanding):
    for fold in result_expanding.folds:
        assert set(fold.selected_params.keys()) == set(PARAM_GRID.keys())


def test_fold_selected_params_values_from_grid(result_expanding):
    for fold in result_expanding.folds:
        for key, val in fold.selected_params.items():
            assert val in PARAM_GRID[key]


def test_fold_selected_params_is_best_is_metric(result_expanding):
    """selected_params must correspond to the highest IS metric in param_results."""
    for fold in result_expanding.folds:
        best = max(fold.param_results, key=lambda pr: pr.is_metric)
        assert fold.selected_params == best.params


def test_fold_oos_metric_is_float(result_expanding):
    for fold in result_expanding.folds:
        assert isinstance(fold.oos_metric, float)


# ---------------------------------------------------------------------------
# Test run: folds are in chronological order
# ---------------------------------------------------------------------------

def test_folds_chronological_across_folds(result_expanding):
    """OOS end of fold k must precede OOS start of fold k+1."""
    folds = result_expanding.folds
    for i in range(1, len(folds)):
        assert folds[i - 1].oos_end < folds[i].oos_start


# ---------------------------------------------------------------------------
# Test run: both schemes produce structurally equivalent results
# ---------------------------------------------------------------------------

def test_expanding_and_rolling_same_fold_count(result_expanding, result_rolling):
    """Both schemes produce the same number of folds for this dataset."""
    assert len(result_expanding.folds) == len(result_rolling.folds)


def test_expanding_is_start_anchored(result_expanding, small_price_df):
    """Expanding scheme: IS start must be the first trading date for all folds."""
    first_date = small_price_df.index.get_level_values("Date").unique().sort_values()[0]
    for fold in result_expanding.folds:
        assert fold.is_start == first_date


def test_rolling_is_start_not_anchored(result_rolling, small_price_df):
    """Rolling scheme: IS start slides forward, so not all folds start at dates[0]."""
    first_date = small_price_df.index.get_level_values("Date").unique().sort_values()[0]
    # At least one fold (beyond first) should have a later IS start
    later_starts = [f.is_start for f in result_rolling.folds if f.is_start != first_date]
    assert len(later_starts) > 0


# ---------------------------------------------------------------------------
# Test run raises when no folds producible
# ---------------------------------------------------------------------------

def test_run_raises_when_scheme_produces_no_folds(small_price_df):
    """
    If scheme.split() returns empty (data too short), run() should raise
    rather than silently returning a result with zero folds.
    """
    # Use an expanding scheme whose windows exceed the data length
    # We patch split to return [] to isolate run() from split() logic
    scheme = ExpandingWindowScheme(win_in=WIN_IN, win_out=WIN_OUT)
    wf = WalkForwardTest(
        prices=small_price_df,
        strategy_cls=CrossSectionalMomentumStrategy,
        param_grid=PARAM_GRID,
        scheme=scheme,
    )
    with patch.object(scheme, "split", return_value=[]):
        with pytest.raises(ValueError, match="No complete folds"):
            wf.run()


# ---------------------------------------------------------------------------
# Test _calculate_metric
# ---------------------------------------------------------------------------

def test_calculate_metric_unsupported_raises(wf_expanding):
    from unittest.mock import MagicMock
    from strategy_backtester.core import BacktestResult
    from strategy_backtester.validation.walk_forward.walk_forward import _compute_metric
    dummy = MagicMock(spec=BacktestResult)
    dummy.returns = pd.Series([0.01, -0.005, 0.002])
    with pytest.raises(ValueError, match="Unsupported metric"):
        _compute_metric(dummy, "calmar")