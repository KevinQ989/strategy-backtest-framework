import numpy as np
import pandas as pd
import pytest
from strategy_backtester.data import make_price_dataframe, PriceDataFrame
from strategy_backtester.engine import BacktestEngine
from strategy_backtester.strategies import CrossSectionalMomentumStrategy
from strategy_backtester.results.metrics import calc_sharpe_ratio


def _make_synthetic_prices() -> PriceDataFrame:
    np.random.seed(42)

    winners = [f"WIN_{i}" for i in range(4)]
    losers  = [f"LOS_{i}" for i in range(4)]
    neutral = [f"NEU_{i}" for i in range(12)]
    tickers = winners + losers + neutral
    dates   = pd.date_range("2020-01-01", "2022-12-31", freq="B")

    records = []
    for ticker in tickers:
        if ticker.startswith("WIN"):   drift = 0.002
        elif ticker.startswith("LOS"): drift = -0.002
        else:                          drift = 0.0

        daily_returns = np.random.normal(drift, 0.005, len(dates))
        prices = 100.0 * np.cumprod(1 + daily_returns)

        for i, date in enumerate(dates):
            records.append({
                "Date": date, "Ticker": ticker,
                "Open": prices[i], "High": prices[i],
                "Low":  prices[i], "Close": prices[i],
                "Adj_Close": prices[i], "Volume": 1_000_000,
            })

    df = pd.DataFrame(records).set_index(["Date", "Ticker"])
    df = df.sort_index()
    price_cols = ["Open", "High", "Low", "Close", "Adj_Close"]
    df[price_cols] = df[price_cols].astype("float64")
    df["Volume"]   = df["Volume"].astype("int64")
    return make_price_dataframe(df)


@pytest.fixture(scope="module")
def synthetic_backtest_result():
    prices = _make_synthetic_prices()
    strategy = CrossSectionalMomentumStrategy(
        lookback=63, skip=0, percent=0.2, rebalance_freq=21
    )
    engine = BacktestEngine(
        prices=prices,
        strategy=strategy,
        metadata={},
        initial_capital=1_000_000.0,
    )
    return engine.run_backtest()


def test_detects_momentum_signal_on_synthetic_data(synthetic_backtest_result):
    """
    Sanity check: the engine must produce Sharpe > 1.0 on price data with
    manufactured persistent momentum.

    Synthetic data parameters (see _make_synthetic_prices):
        - 4 winners:  +0.2% daily drift
        - 4 losers:   -0.2% daily drift
        - 12 neutral:  0.0% daily drift
        - Daily noise: 0.5% std (i.i.d. Gaussian, seed=42)
        - Lookback:   63 trading days (~3 months)

    The expected long/short spread is 0.4%/day before costs. With 0.5%
    daily noise across 8 positions, the strategy has ample signal to
    produce a Sharpe well above 1.0 over the 3-year test window.

    PASS: Sharpe > 1.0 — the engine detects and exploits the signal.

    FAIL interpretations:
        - No positions taken: lookback or percent params misconfigured —
          the strategy never generates a signal.
        - Sharpe <= 1.0: the engine or ranking logic is broken. The signal
          is too strong to miss legitimately. This is not a data problem.
          Use test_winners_longed_losers_shorted_on_synthetic_data to
          distinguish a broken ranking from costs killing the edge.
    """
    result = synthetic_backtest_result

    assert result.positions.abs().sum(axis=1).max() > 0, \
        "Strategy never took any positions — check percent/lookback params"

    sharpe = calc_sharpe_ratio(result.returns)
    assert sharpe > 1.0, \
        f"Sharpe = {sharpe:.4f} on synthetic data with clear momentum signal. " \
        f"Engine or ranking logic is not detecting signal correctly."
    

def test_winners_longed_losers_shorted_on_synthetic_data(synthetic_backtest_result):
    """
    Checks that the ranking logic correctly identifies winners as long
    candidates and losers as short candidates on active rebalance days.
    Use this alongside test_detects_momentum_signal_on_synthetic_data
    to isolate the failure mode when the Sharpe test fails.

    Synthetic data parameters (see _make_synthetic_prices):
        - 4 winners:  +0.2% daily drift
        - 4 losers:   -0.2% daily drift
        - 12 neutral:  0.0% daily drift
        - Daily noise: 0.5% std (i.i.d. Gaussian, seed=42)
        - Lookback:   63 trading days (~3 months)
        - percent=0.2 on 20 tickers → top 4 longed, bottom 4 shorted

    Threshold derivation:
        Over a 63-day lookback, winner vs neutral cumulative return gap:
            Expected: 0.002 × 63 = 12.6%
            Noise std: 0.005 × sqrt(63) ≈ 3.97%
            Separation: 12.6% / (3.97% × sqrt(2)) ≈ 2.24 sigma

        P(a single neutral beats a specific winner) = Φ(-2.24) ≈ 1.2%
        P(at least one of 12 neutrals beats it)     ≈ 1 - (1-0.012)^12 ≈ 13.6%

        For a winner to miss the long leg, it must also rank last among
        the 4 winners (probability 1/4 = 25%, since all share equal drift):
            P(winner excluded) ≈ 0.25 × 0.136 ≈ 3.4%
            P(winner longed)   ≈ 96.6%

        By symmetry, P(loser shorted) ≈ 96.6%.
        Threshold of 0.90 sits well below the theoretical ~96.6%,
        giving headroom for rebalance-to-rebalance variance without
        producing flaky failures.

    PASS: winner_long_rate > 0.90 and loser_short_rate > 0.90 —
        the ranking logic correctly separates winners from losers.

    FAIL interpretations (check which assertion fails and by how much):
        - Rate ≈ 0.10: long and short legs are inverted — the strategy
          is going long losers and short winners.
        - Rate ≈ 0.50: signal computation is broken — the strategy is
          ranking randomly.
        - Rate between 0.50 and 0.90: partial signal, possibly a lookback
          off-by-one or skip parameter is consuming too much of the window.

    Diagnostic use:
        If this test PASSES but the Sharpe test FAILS, the ranking is
        correct — transaction costs or position sizing are killing the edge.
        If both tests FAIL, the signal computation itself is wrong.
    """
    result = synthetic_backtest_result
    positions = result.positions

    # Only examine days where the strategy holds positions
    active = positions[positions.abs().sum(axis=1) > 0]

    winner_cols = [c for c in positions.columns if c.startswith("WIN_")]
    loser_cols  = [c for c in positions.columns if c.startswith("LOS_")]

    winner_long_rate = (active[winner_cols] > 0).mean().mean()
    loser_short_rate = (active[loser_cols]  < 0).mean().mean()

    assert winner_long_rate > 0.90, \
        f"Winners longed only {winner_long_rate:.1%} of active days (expected >90%). " \
        f"Ranking logic may be inverted or signal is noisy."
    assert loser_short_rate > 0.90, \
        f"Losers shorted only {loser_short_rate:.1%} of active days (expected >90%). " \
        f"Ranking logic may be inverted or signal is noisy."