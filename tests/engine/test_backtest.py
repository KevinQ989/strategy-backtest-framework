from __future__ import annotations
import pandas as pd
import pytest
from strategy_backtester.core import BacktestResult, PortfolioWeights
from strategy_backtester.engine import BacktestEngine
from strategy_backtester.strategies import BaseStrategy, RandomStrategy, CrossSectionalMomentumStrategy
from strategy_backtester.results import calc_final_value

# ---------------------------------------------------------------------------
# Slicing regression: historical_data.loc[:current_date] must return ALL
# tickers' rows for ALL dates up to and including current_date — not a
# row-count-based slice (the original .iloc bug).
# ---------------------------------------------------------------------------

def test_date_slicing_returns_all_tickers_for_all_prior_dates(price_df, tickers):
    """
    For a rectangular (Date, Ticker) panel with T tickers per date,
    .loc[:current_date] at the i-th date must contain exactly (i+1) * T rows
    (all tickers, for every date up to and including current_date) —
    not (i+1) rows, which is what an .iloc[:i+1] bug would produce.
    """
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    n_tickers = len(tickers)

    for i in (0, 1, 10, len(dates) - 1):
        current_date = dates[i]
        sliced = price_df.loc[:current_date]

        expected_rows = (i + 1) * n_tickers
        assert len(sliced) == expected_rows

        # Every ticker must be present for every date in the slice
        sliced_dates = sliced.index.get_level_values("Date").unique()
        assert len(sliced_dates) == i + 1
        for d in sliced_dates:
            assert set(sliced.xs(d, level="Date").index) == set(tickers)


def test_date_slicing_excludes_future_dates(price_df):
    """.loc[:current_date] must not include any date after current_date."""
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    current_date = dates[50]
    sliced = price_df.loc[:current_date]
    sliced_dates = sliced.index.get_level_values("Date").unique()
    assert sliced_dates.max() == current_date
    assert (sliced_dates <= current_date).all()


# ---------------------------------------------------------------------------
# A strategy that records exactly what `prices` it was given on each call,
# so we can assert the engine passes correctly-sliced, growing history.
# ---------------------------------------------------------------------------

class RecordingStrategy(BaseStrategy):
    """Records (as_of, n_dates_seen, n_rows_seen) on every generate() call
    and always rebalances, but returns flat (empty) weights so the engine
    runs without executing trades."""

    def __init__(self):
        self.calls = []

    def should_rebalance(self, date, last_rebalance, current_weights, prices):
        return True

    def generate(self, prices, as_of, current_weights):
        dates_seen = prices.index.get_level_values("Date").unique()
        self.calls.append({
            "as_of": as_of,
            "n_dates_seen": len(dates_seen),
            "n_rows_seen": len(prices),
            "max_date_seen": dates_seen.max(),
        })
        return PortfolioWeights(
            date=as_of,
            long_weights=pd.Series(dtype=float),
            short_weights=pd.Series(dtype=float),
        )


def test_engine_passes_growing_history_with_correct_row_counts(price_df, tickers):
    """
    Each generate() call (from day 2 onward) must see exactly
    (number of dates up to and including as_of) * n_tickers rows, and the
    max date seen must equal as_of — confirming date-based (not row-count
    based) slicing end-to-end through the engine.
    """
    strategy = RecordingStrategy()
    engine = BacktestEngine(price_df, strategy)
    result = engine.run_backtest()

    assert isinstance(result, BacktestResult)
    assert len(strategy.calls) > 0

    n_tickers = len(tickers)
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    date_to_idx = {d: i for i, d in enumerate(dates)}

    for call in strategy.calls:
        i = date_to_idx[call["as_of"]]
        assert call["n_dates_seen"] == i + 1
        assert call["n_rows_seen"] == (i + 1) * n_tickers
        assert call["max_date_seen"] == call["as_of"]


# ---------------------------------------------------------------------------
# BacktestEngine.__init__
# ---------------------------------------------------------------------------

def test_init_calls_prepare(price_df):
    """historical_data must be strategy.prepare(prices), not prices itself,
    even for the default no-op prepare (identity check)."""
    strategy = RandomStrategy()
    engine = BacktestEngine(price_df, strategy)
    assert engine.historical_data is strategy.prepare(price_df)


def test_init_stores_initial_capital_and_metadata(price_df):
    strategy = RandomStrategy()
    metadata = {"config": "test"}
    engine = BacktestEngine(price_df, strategy, metadata=metadata, initial_capital=50_000.0)
    assert engine.initial_capital == 50_000.0
    assert engine.metadata is metadata


def test_init_default_initial_capital(price_df):
    strategy = RandomStrategy()
    engine = BacktestEngine(price_df, strategy)
    assert engine.initial_capital == 100_000.0
    assert engine.metadata is None


# ---------------------------------------------------------------------------
# _has_run guard (single-use engine contract)
# ---------------------------------------------------------------------------

def test_run_backtest_sets_has_run_flag(price_df):
    strategy = RandomStrategy()
    engine = BacktestEngine(price_df, strategy)
    assert engine._has_run is False
    engine.run_backtest()
    assert engine._has_run is True


def test_run_backtest_second_call_raises(price_df):
    strategy = RandomStrategy()
    engine = BacktestEngine(price_df, strategy)
    engine.run_backtest()
    with pytest.raises(RuntimeError):
        engine.run_backtest()


def test_run_backtest_second_call_does_not_mutate_first_result(price_df):
    """
    A rejected second run must not have side-effects on previously-returned
    results (i.e. the RuntimeError is raised before any state mutation).
    """
    strategy = RandomStrategy()
    engine = BacktestEngine(price_df, strategy)
    result1 = engine.run_backtest()
    returns1 = result1.returns.copy()

    with pytest.raises(RuntimeError):
        engine.run_backtest()

    pd.testing.assert_series_equal(result1.returns, returns1)


# ---------------------------------------------------------------------------
# Result structure / basic invariants
# ---------------------------------------------------------------------------

@pytest.fixture
def momentum_result(price_df):
    strategy = CrossSectionalMomentumStrategy(lookback=252, skip=21, percent=0.1, rebalance_freq=21)
    engine = BacktestEngine(price_df, strategy)
    return engine.run_backtest()


def test_no_unique_dates(price_df):
    """If the input data has no dates, the engine should raise a ValueError."""
    empty_price_df = price_df.iloc[0:0]
    strategy = RandomStrategy()
    engine = BacktestEngine(empty_price_df, strategy)
    with pytest.raises(ValueError, match="No data downloaded"):
        engine.run_backtest()


def test_result_returns_indexed_by_all_dates(momentum_result, price_df):
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    assert len(momentum_result.returns) == len(dates)
    assert momentum_result.returns.index.equals(dates)


def test_result_first_day_return_is_zero(momentum_result):
    """Day 1 has no prior value to compute a return against."""
    assert momentum_result.returns.iloc[0] == 0.0


def test_result_costs_and_turnover_nonnegative(momentum_result):
    assert (momentum_result.costs >= 0).all()
    assert (momentum_result.turnover >= 0).all()


def test_result_costs_and_turnover_zero_on_non_rebalance_days(momentum_result):
    """
    With rebalance_freq=21, most days should have zero cost/turnover —
    confirms the engine only trades on rebalance days, not every day.
    """
    n_zero_cost_days = (momentum_result.costs == 0.0).sum()
    assert n_zero_cost_days > len(momentum_result.costs) // 2


def test_result_positions_columns_are_tickers(momentum_result, tickers):
    assert set(momentum_result.positions.columns).issubset(set(tickers))


def test_result_starting_capital_matches_engine(price_df):
    strategy = RandomStrategy()
    engine = BacktestEngine(price_df, strategy, initial_capital=75_000.0)
    result = engine.run_backtest()
    assert result.starting_capital == 75_000.0


# ---------------------------------------------------------------------------
# No-rebalance edge case: strategy that never rebalances
# ---------------------------------------------------------------------------

class NeverRebalanceStrategy(BaseStrategy):
    def should_rebalance(self, date, last_rebalance, current_weights, prices):
        return False

    def generate(self, prices, as_of, current_weights):
        raise AssertionError("generate() should never be called if should_rebalance is always False")


def test_never_rebalance_strategy_produces_flat_zero_returns(price_df):
    """
    If should_rebalance is always False, the portfolio never takes positions,
    so daily returns must be all zero and costs/turnover all zero.
    """
    strategy = NeverRebalanceStrategy()
    engine = BacktestEngine(price_df, strategy)
    result = engine.run_backtest()

    assert (result.returns == 0.0).all()
    assert (result.costs == 0.0).all()
    assert (result.turnover == 0.0).all()
    assert calc_final_value(result.starting_capital, result.returns) == pytest.approx(result.starting_capital)


# ---------------------------------------------------------------------------
# Engine does not mutate input prices
# ---------------------------------------------------------------------------

def test_run_backtest_does_not_mutate_input_prices(price_df):
    before = price_df.copy(deep=True)
    strategy = CrossSectionalMomentumStrategy(lookback=252, skip=21, percent=0.1, rebalance_freq=21)
    engine = BacktestEngine(price_df, strategy)
    engine.run_backtest()
    pd.testing.assert_frame_equal(price_df, before)