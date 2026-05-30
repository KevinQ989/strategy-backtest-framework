from __future__ import annotations
import numpy as np
from strategy_backtester.engine import BacktestEngine
from strategy_backtester.strategies import RandomStrategy, CrossSectionalMomentumStrategy
from strategy_backtester.data import load_data
from strategy_backtester.validation import PermutationTest, IIDScheme, RanksScheme, BlockScheme


def print_result(label: str, result) -> None:
    print(f"\n--- {label} ---")
    print(f"  Final portfolio value:  ${result.final_value:>12,.2f}")
    print(f"  Cumulative return:      {result.cumulative_return:>12.2%}")
    print(f"  Annualised return:      {result.annualised_return:>12.2%}")
    print(f"  Annualised volatility:  {result.annualised_volatility:>12.2%}")
    print(f"  Sharpe ratio:           {result.sharpe_ratio:>12.2f}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Universe and date range
    # ------------------------------------------------------------------
    tickers = [
        "AAPL", "MSFT", "GOOG", "AMZN", "TSLA",
        "NVDA", "META", "JPM",  "JNJ",  "XOM",
        "UNH",  "V",    "PG",   "HD",   "MA",
        "BAC",  "ABBV", "CVX",  "MRK",  "LLY",
    ]
    start_date      = "2018-01-01"
    end_date        = "2023-12-31"
    initial_capital = 1_000_000.0

    # ------------------------------------------------------------------
    # Load data once — reused by engine and permutation test
    # ------------------------------------------------------------------
    print("Loading price data...")
    prices = load_data(tickers, start_date, end_date)
    print(f"Loaded {len(prices.index.get_level_values('Date').unique())} trading days "
          f"for {len(tickers)} tickers.\n")

    metadata = {
        "tickers":         tickers,
        "start_date":      start_date,
        "end_date":        end_date,
        "initial_capital": initial_capital,
        "strategy":        "CrossSectionalMomentum",
    }

    # ------------------------------------------------------------------
    # Strategy configuration
    # ------------------------------------------------------------------
    strategy = CrossSectionalMomentumStrategy(
        lookback=252,
        skip=21,
        decile=0.2,   # top and bottom 20% — 4 tickers per leg from 20-stock universe
        rebalance_freq=21,
    )

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------
    print("Running backtest...")
    engine = BacktestEngine(prices, metadata=metadata, initial_capital=initial_capital)
    result = engine.run_backtest(strategy)
    print_result("Cross-Sectional Momentum — Full Period", result)

    # ------------------------------------------------------------------
    # Permutation test
    # ------------------------------------------------------------------
    print("\n\nRunning permutation test...")
    print("This will take a few minutes.\n")

    perm_test = PermutationTest(
        prices=prices,
        strategy=CrossSectionalMomentumStrategy(
            lookback=252,
            skip=21,
            decile=0.2,
            rebalance_freq=21,
        ),
        scheme=RanksScheme(),
        N=1000,
        metric="sharpe",
        initial_capital=initial_capital,
        seed=42,
    )
    perm_results = perm_test.run()
    
    print("\n--- Permutation Test Results ---")
    print(f"  Scheme:                 RanksScheme")
    print(f"  N permutations:         {perm_results.N}")
    print(f"  Baseline Sharpe:        {perm_results.baseline.sharpe_ratio:>8.2f}")
    null_sharpes = [r.sharpe_ratio for r in perm_results.null_distribution]
    print(f"  Mean null Sharpe:       {np.mean(null_sharpes):>8.2f}")
    print(f"  Null Sharpe std:        {np.std(null_sharpes):>8.2f}")
    print(f"  p-value (one-tailed):   {perm_results.p_value:>8.4f}")
    if perm_results.p_value < 0.05:
        print("  Interpretation: Statistically significant at the 5% level.")
        print("                  The momentum ranking criterion has predictive power.")
    elif perm_results.p_value < 0.10:
        print("  Interpretation: Marginal significance at the 10% level.")
    else:
        print("  Interpretation: Not statistically significant.")
        print("                  Cannot reject the null hypothesis of no predictive power.")