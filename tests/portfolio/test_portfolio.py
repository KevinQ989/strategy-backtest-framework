from __future__ import annotations
import pytest
import pandas as pd
from strategy_backtester.portfolio.portfolio import PortfolioState
from strategy_backtester.execution.simulator import _no_trade_result
from strategy_backtester.core.types import ExecutionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CAPITAL = 100_000.0


def _make_state(trading_dates, price_df, capital: float = CAPITAL) -> PortfolioState:
    """Fresh PortfolioState with no positions."""
    state = PortfolioState(date=trading_dates[20], starting_capital=capital)
    state.update_to_market(
        price_df.xs(trading_dates[20], level="Date")["Close"],
        trading_dates[20]
    )
    return state



def _make_execution_result(
    date: pd.Timestamp,
    fills: dict[str, float],
    prices: dict[str, float],
    turnover: float = 0.0,
    slippage: float = 0.0,
    commission: float = 0.0,
    spread: float = 0.0,
) -> ExecutionResult:
    """Build an ExecutionResult from plain dicts."""
    return ExecutionResult(
        date=date,
        fills=pd.Series(fills, dtype=float),
        execution_prices=pd.Series(prices, dtype=float),
        turnover=turnover,
        slippage=slippage,
        commission=commission,
        spread=spread,
    )


# ---------------------------------------------------------------------------
# Test __init__
# ---------------------------------------------------------------------------

def test_init(trading_dates):
    state = PortfolioState(date=trading_dates[0], starting_capital=CAPITAL)
    assert state.cash == CAPITAL
    assert state.starting_capital == CAPITAL
    assert state.positions.empty
    assert state._last_prices.empty
    assert state.date == trading_dates[0]


# ---------------------------------------------------------------------------
# Test total_value
# ---------------------------------------------------------------------------

def test_total_value_no_position(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    assert state.total_value == CAPITAL


def test_total_value_with_positions(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    ticker = price_df.index.get_level_values("Ticker").unique()[:3].tolist()
    shares = [5.0, 10.0, -3.0]
    for t, s in zip(ticker, shares):
        state.positions[t] = s
    state._invalidate_cache()
    prices = price_df.xs(trading_dates[0], level="Date")["Close"]
    expected = CAPITAL + sum(s * prices[t] for t, s in zip(ticker, shares))
    assert state.total_value == pytest.approx(expected)


def test_total_value_missing_prices(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    state.positions["DOES_NOT_EXIST"] = 10.0
    state._invalidate_cache()
    with pytest.raises(ValueError, match="No price available for positions"):
        _ = state.total_value


# ---------------------------------------------------------------------------
# Test current_weights
# ---------------------------------------------------------------------------

def test_current_weights_empty_no_positions(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    assert state.current_weights.empty


def test_current_weights_empty_total_value_zero(trading_dates, price_df):
    state = _make_state(trading_dates, price_df, capital=0.0)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    assert state.current_weights.empty


def test_current_weights_positive_for_longs(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    ticker = price_df.index.get_level_values("Ticker").unique()[0]
    state.positions[ticker] = 10.0
    state._invalidate_cache()
    assert state.current_weights[ticker] > 0


def test_current_weights_negative_for_shorts(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    ticker = price_df.index.get_level_values("Ticker").unique()[0]
    state.positions[ticker] = -10.0
    state._invalidate_cache()
    assert state.current_weights[ticker] < 0


def test_current_weights_scale_with_price(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    ticker = price_df.index.get_level_values("Ticker").unique()[0]
    price = price_df.loc[(trading_dates[0], ticker), "Close"]
    shares = 5.0
    state.positions[ticker] = shares
    state._invalidate_cache()
    expected_weight = (shares * price) / state.total_value
    assert state.current_weights[ticker] == pytest.approx(expected_weight)


# ---------------------------------------------------------------------------
# Test update_to_market
# ---------------------------------------------------------------------------

def test_update_to_market_advances_date(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[5], level="Date")["Close"], trading_dates[5])
    assert state.date == trading_dates[5]


def test_update_to_market_sets_last_prices(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    assert not state._last_prices.empty


def test_update_to_market_last_prices_match_close(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    date = trading_dates[10]
    state.update_to_market(price_df.xs(date, level="Date")["Close"], date)
    expected_close = price_df.xs(date, level="Date")["Close"]
    pd.testing.assert_series_equal(
        state._last_prices.sort_index(),
        expected_close.sort_index(),
        check_names=False,
    )


def test_update_to_market_does_not_change_positions(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    ticker = price_df.index.get_level_values("Ticker").unique()[0]
    state.positions[ticker] = 10.0
    state._invalidate_cache()
    state.update_to_market(price_df.xs(trading_dates[1], level="Date")["Close"], trading_dates[1])
    assert state.positions[ticker] == 10.0


def test_update_to_market_does_not_change_cash(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    assert state.cash == CAPITAL


def test_update_to_market_updates_total_value(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    ticker = price_df.index.get_level_values("Ticker").unique()[0]
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    price_t0 = price_df.loc[(trading_dates[0], ticker), "Close"]
    state.positions[ticker] = 10.0
    state.cash = 0.0
    state._invalidate_cache()
    value_t0 = state.total_value

    state.update_to_market(price_df.xs(trading_dates[1], level="Date")["Close"], trading_dates[1])
    price_t1 = price_df.loc[(trading_dates[1], ticker), "Close"]
    value_t1 = state.total_value

    assert value_t0 == pytest.approx(10.0 * price_t0)
    assert value_t1 == pytest.approx(10.0 * price_t1)


# ---------------------------------------------------------------------------
# Test update_to_execution
# ---------------------------------------------------------------------------

def test_update_to_execution_no_trade_leaves_state_unchanged(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    cash_before = state.cash
    result = _no_trade_result(trading_dates[0])
    state.update_to_execution(result)
    assert state.cash == pytest.approx(cash_before)
    assert state.positions.empty


def test_update_to_execution_long(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    ticker = price_df.index.get_level_values("Ticker").unique()[0]
    price = price_df.loc[(trading_dates[0], ticker), "Close"]
    result = _make_execution_result(
        date=trading_dates[0],
        fills={ticker: 10.0},
        prices={ticker: price},
    )
    state.update_to_execution(result)
    assert state.positions[ticker] == pytest.approx(10.0)
    assert state.cash == pytest.approx(CAPITAL - 10.0 * price)


def test_update_to_execution_short(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    ticker = price_df.index.get_level_values("Ticker").unique()[0]
    price = price_df.loc[(trading_dates[0], ticker), "Close"]
    result = _make_execution_result(
        date=trading_dates[0],
        fills={ticker: -10.0},
        prices={ticker: price},
    )
    state.update_to_execution(result)
    assert state.positions[ticker] == pytest.approx(-10.0)
    assert state.cash == pytest.approx(CAPITAL + 10.0 * price)


def test_update_to_execution_costs_deducted_from_cash(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    slippage = 1_000
    commission = 500
    spread = 200
    result = ExecutionResult(
        date=trading_dates[0],
        fills=pd.Series(dtype=float),
        execution_prices=pd.Series(dtype=float),
        turnover=0.0,
        slippage=slippage,
        commission=commission,
        spread=spread,
    )
    state.update_to_execution(result)
    total_cost = slippage + commission + spread
    assert state.cash == pytest.approx(CAPITAL - total_cost)


def test_update_to_execution_zero_position_removed(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    ticker = price_df.index.get_level_values("Ticker").unique()[0]
    price = price_df.loc[(trading_dates[0], ticker), "Close"]
    state.positions[ticker] = 10.0
    state._invalidate_cache()
    result = _make_execution_result(
        date=trading_dates[0],
        fills={ticker: -10.0},
        prices={ticker: price},
    )
    state.update_to_execution(result)
    assert ticker not in state.positions.index


def test_update_to_execution_accumulates_across_calls(trading_dates, price_df):
    state = _make_state(trading_dates, price_df)
    state.update_to_market(price_df.xs(trading_dates[0], level="Date")["Close"], trading_dates[0])
    ticker1 = price_df.index.get_level_values("Ticker").unique()[0]
    ticker2 = price_df.index.get_level_values("Ticker").unique()[1]
    price1 = price_df.loc[(trading_dates[0], ticker1), "Close"]
    price2 = price_df.loc[(trading_dates[0], ticker2), "Close"]
    result1 = _make_execution_result(
        date=trading_dates[0],
        fills={ticker1: 5.0},
        prices={ticker1: price1},
    )
    result2 = _make_execution_result(
        date=trading_dates[0],
        fills={ticker1: 3.0, ticker2: 2.0},
        prices={ticker1: price1, ticker2: price2},
    )
    state.update_to_execution(result1)
    state.update_to_execution(result2)
    assert state.positions[ticker1] == pytest.approx(8.0)
    assert state.positions[ticker2] == pytest.approx(2.0)