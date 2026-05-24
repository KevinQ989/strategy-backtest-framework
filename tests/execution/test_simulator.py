from __future__ import annotations
import pytest
import pandas as pd
from strategy_backtester.core.types import PortfolioWeights, ExecutionResult
from strategy_backtester.portfolio import PortfolioState
from strategy_backtester.execution.simulator import (
    execute,
    _compute_adv,
    _compute_commission,
    _compute_spread,
    _compute_slippage
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CAPITAL = 1_000_000.0
SIGNAL_DATE = pd.Timestamp("2022-05-31")
COMMISSION_BPS = 5.0
SPREAD_BPS = 5.0
SLIPPAGE_K = 0.1
ADV_WINDOW = 20
BPS = 1e-4


def _make_weights(
    long_weights: dict[str, float] | None = None,
    short_weights: dict[str, float] | None = None,
) -> PortfolioWeights:
    return PortfolioWeights(
        date=SIGNAL_DATE,
        long_weights=pd.Series(long_weights or {}, dtype=float),
        short_weights=pd.Series(short_weights or {}, dtype=float),
    )
 
 
def _make_state(trading_dates, price_df, capital: float = CAPITAL) -> PortfolioState:
    """Fresh PortfolioState marked to market at trading_dates[20]."""
    state = PortfolioState(date=trading_dates[20], starting_capital=capital)
    state.update_to_market(price_df, trading_dates[20])
    return state
 
 
def _make_state_with_positions(
    trading_dates,
    price_df,
    tickers: list[str],
    weights: dict[str, float],
    capital: float = CAPITAL,
) -> PortfolioState:
    """
    PortfolioState with existing positions.
    weights: ticker -> target weight (positive = long, negative = short).
    """
    state = PortfolioState(date=trading_dates[20], starting_capital=capital)
    state.update_to_market(price_df, trading_dates[20])
    prices = price_df.xs(trading_dates[20], level="Date")["Close"]
    for ticker, weight in weights.items():
        dollar_val = capital * weight
        state.positions[ticker] = dollar_val / prices[ticker]
    state.cash = capital - sum(
        capital * w for w in weights.values()
    )
    return state


def _exec_date(trading_dates):
    """Execution occurs at T+1 open, so date is trading_dates[21]."""
    return trading_dates[21]


def _call_execute(
    pending,
    state,
    price_df,
    trading_dates,
    tickers,
    commission_bps=COMMISSION_BPS,
    spread_bps=SPREAD_BPS,
    slippage_k=SLIPPAGE_K,
    adv_window=ADV_WINDOW,
):
    """Thin wrapper to reduce boilerplate in tests."""
    exec_date = _exec_date(trading_dates)
    open_prices = price_df.xs(exec_date, level="Date")["Open"].reindex(tickers)
    return execute(
        pending=pending,
        state=state,
        open_prices=open_prices,
        hist_prices=price_df,
        date=exec_date,
        commission_bps=commission_bps,
        spread_bps=spread_bps,
        slippage_k=slippage_k,
        adv_window=adv_window,
    )

# ---------------------------------------------------------------------------
# Test execute
# ---------------------------------------------------------------------------

def test_execute_returns_execution_result(trading_dates, price_df, tickers):
    state = _make_state(trading_dates, price_df)
    pending = _make_weights(long_weights={tickers[0]: 0.5})
    result = _call_execute(pending, state, price_df, trading_dates, tickers)
    assert isinstance(result, ExecutionResult)


def test_execute_correct_date(trading_dates, price_df, tickers):
    state = _make_state(trading_dates, price_df)
    pending = _make_weights(long_weights={tickers[0]: 0.5})
    result = _call_execute(pending, state, price_df, trading_dates, tickers)
    assert result.date == _exec_date(trading_dates)


def test_execute_long_and_short_fills_correct_signs(trading_dates, price_df, tickers):
    state = _make_state(trading_dates, price_df)
    long_tickers = tickers[:2]
    short_tickers = tickers[5:7]
    pending = _make_weights(
        long_weights={t: 0.2 for t in long_tickers},
        short_weights={t: -0.2 for t in short_tickers},
    )
    result = _call_execute(pending, state, price_df, trading_dates, tickers)
    assert (result.fills[long_tickers] > 0).all()
    assert (result.fills[short_tickers] < 0).all()


def test_execute_execution_prices_match_open(trading_dates, price_df, tickers):
    state = _make_state(trading_dates, price_df)
    subset_tickers = tickers[:3]
    pending = _make_weights(long_weights={t: 0.5 for t in subset_tickers})
    exec_date = _exec_date(trading_dates)
    open_prices = price_df.xs(exec_date, level="Date")["Open"].reindex(tickers)
    result = _call_execute(pending, state, price_df, trading_dates, tickers)
    for ticker in subset_tickers:
        assert result.execution_prices[ticker] == pytest.approx(open_prices[ticker])


def test_execute_turnover_equals_sum_abs_weight_deltas(trading_dates, price_df, tickers):
    state = _make_state(trading_dates, price_df)
    pending = _make_weights(
        long_weights={tickers[0]: 0.1, tickers[1]: 0.2},
        short_weights={tickers[5]: -0.1}
    )
    result = _call_execute(pending, state, price_df, trading_dates, tickers)
    assert result.turnover == pytest.approx(0.4, rel=1e-4)


def test_execute_no_trade_when_empty_weights(trading_dates, price_df, tickers):
    state = _make_state(trading_dates, price_df)
    pending = _make_weights()
    result = _call_execute(pending, state, price_df, trading_dates, tickers)
    assert result.fills.empty
    assert result.turnover == 0.0
    assert result.total_cost == 0.0


def test_execute_no_trade_when_weight_deltas_below_threshold(trading_dates, price_df, tickers):
    state = _make_state(trading_dates, price_df)
    pending = _make_weights(long_weights={tickers[0]: BPS / 2})
    result = _call_execute(pending, state, price_df, trading_dates, tickers)
    assert result.fills.empty
    assert result.turnover == 0.0
    assert result.total_cost == 0.0


def test_execute_no_trade_when_weights_equal_current(trading_dates, price_df, tickers):
    state = _make_state_with_positions(
        trading_dates,
        price_df,
        tickers,
        weights={tickers[0]: 0.1, tickers[1]: -0.1}
    )
    pending = _make_weights(long_weights={tickers[0]: 0.1}, short_weights={tickers[1]: -0.1})
    result = _call_execute(pending, state, price_df, trading_dates, tickers)
    assert result.fills.empty
    assert result.turnover == 0.0
    assert result.total_cost == 0.0


def test_execute_raises_on_missing_open_price(trading_dates, price_df, tickers):
    state = _make_state(trading_dates, price_df)
    pending = _make_weights(long_weights={tickers[0]: 0.5})
    exec_date = _exec_date(trading_dates)
    open_prices = price_df.xs(exec_date, level="Date")["Open"].reindex(tickers[1:])
    with pytest.raises(ValueError, match="Missing open prices for tickers:"):
        execute(
            pending=pending,
            state=state,
            open_prices=open_prices,
            hist_prices=price_df,
            date=exec_date,
            commission_bps=COMMISSION_BPS,
            spread_bps=SPREAD_BPS,
            slippage_k=SLIPPAGE_K,
            adv_window=ADV_WINDOW,
        )


# ---------------------------------------------------------------------------
# Test _compute_adv
# ---------------------------------------------------------------------------

def test_adv_positive_for_known_tickers(price_df, tickers):
    target = pd.Index(tickers[:5])
    adv = _compute_adv(price_df, target, ADV_WINDOW)
    assert (adv > 0).all()


def test_adv_fallback_for_unknown_tickers(price_df):
    fallback = 1_000_000.0
    target = pd.Index(["DoesNotExist"])
    adv = _compute_adv(price_df, target, ADV_WINDOW)
    assert adv["DoesNotExist"] == fallback


# ---------------------------------------------------------------------------
# Test _compute_commission
# ---------------------------------------------------------------------------

def test_commission_non_negative():
    abs_trade_values = pd.Series([10_000.0, 20_000.0])
    result = _compute_commission(abs_trade_values, CAPITAL, COMMISSION_BPS)
    assert result >= 0.0


def test_commission_zero_for_zero_trade():
    result = _compute_commission(pd.Series([0.0]), CAPITAL, COMMISSION_BPS)
    assert result == pytest.approx(0.0)
 
 
def test_commission_scales_with_bps():
    base = _compute_commission(pd.Series([100_000.0]), CAPITAL, 5.0)
    doubled = _compute_commission(pd.Series([100_000.0]), CAPITAL, 10.0)
    assert doubled == pytest.approx(2 * base)


# ---------------------------------------------------------------------------
# Test _compute_spread
# ---------------------------------------------------------------------------

def test_spread_non_negative():
    abs_trade_values = pd.Series([10_000.0, 20_000.0])
    result = _compute_spread(abs_trade_values, CAPITAL, SPREAD_BPS)
    assert result >= 0.0


def test_spread_zero_for_zero_trade():
    result = _compute_spread(pd.Series([0.0]), CAPITAL, SPREAD_BPS)
    assert result == pytest.approx(0.0)
 
 
def test_spread_scales_with_bps():
    base = _compute_spread(pd.Series([100_000.0]), CAPITAL, 5.0)
    doubled = _compute_spread(pd.Series([100_000.0]), CAPITAL, 10.0)
    assert doubled == pytest.approx(2 * base)


# ---------------------------------------------------------------------------
# Test _compute_slippage
# ---------------------------------------------------------------------------

def test_slippage_non_negative(tickers):
    abs_trade_values = pd.Series({tickers[0]: 100_000.0})
    adv = pd.Series({tickers[0]: 5_000_000.0})
    result = _compute_slippage(abs_trade_values, adv, CAPITAL, SLIPPAGE_K)
    assert result >= 0.0


def test_slippage_zero_for_zero_trade(tickers):
    ticker = tickers[0]
    adv = pd.Series({ticker: 5_000_000.0})
    result = _compute_slippage(pd.Series({ticker: 0.0}), adv, CAPITAL, SLIPPAGE_K)
    assert result == pytest.approx(0.0)


def test_slippage_increases_with_order_size(tickers):
    ticker = tickers[0]
    adv = pd.Series({ticker: 5_000_000.0})
    small = _compute_slippage(pd.Series({ticker: 50_000.0}), adv, CAPITAL, SLIPPAGE_K)
    large = _compute_slippage(pd.Series({ticker: 200_000.0}), adv, CAPITAL, SLIPPAGE_K)
    assert large > small