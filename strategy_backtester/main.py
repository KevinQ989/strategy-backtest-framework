from __future__ import annotations
import numpy as np
import pandas as pd
import yaml
import os
import io
import time
import urllib.request
from strategy_backtester.engine import BacktestEngine
from strategy_backtester.strategies import RandomStrategy, CrossSectionalMomentumStrategy
from strategy_backtester.data import load_data
from strategy_backtester.validation import PermutationTest, RankPermutationStrategy, IIDPermutationStrategy, BlockPermutationStrategy


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)
    

def build_universe(cfg: dict) -> list[str]:
    universe = cfg["backtest"].get("universe", "")
    if universe == "sp500":
        print("Fetching S&P 500 constituents from Wikipedia...")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; strategy-backtest-framework)"}
        )
        with urllib.request.urlopen(req) as response:
            html = response.read()
        table = pd.read_html(io.BytesIO(html), attrs={"id": "constituents"})[0]
        return table["Symbol"].str.replace(".", "-", regex=False).tolist()
    tickers = cfg["backtest"].get("tickers", [])
    if not tickers:
        raise ValueError(
            "No tickers specified. Set 'universe: sp500' or provide "
            "a list of tickers in config.yaml."
        )
    return tickers


def build_strategy(cfg: dict):
    name = cfg["backtest"]["strategy"]
    params = cfg["strategies"][name]
    if name == "random":
        return RandomStrategy()
    elif name == "cross_sectional_momentum":
        return CrossSectionalMomentumStrategy(**params)
    else:
        raise ValueError(f"Unknown strategy: {name}")


def build_scheme(cfg: dict):
    scheme_name = cfg["permutation"]["scheme"]
    if scheme_name == "ranks":
        return RankPermutationStrategy, {}
    elif scheme_name == "iid":
        return IIDPermutationStrategy, {}
    elif scheme_name == "block":
        return BlockPermutationStrategy, {"block_size": cfg["permutation"]["block_size"]}
    else:
        raise ValueError(f"Unknown permutation scheme: {scheme_name}")


def to_label(name: str) -> str:
    return name.replace("_", " ").title()


def print_result(label: str, result) -> None:
    print(f"\n--- {label} ---")
    print(f"  Final portfolio value:  ${result.final_value:<12,.2f}")
    print(f"  Cumulative return:      {result.cumulative_return:<12.2%}")
    print(f"  Annualised return:      {result.annualised_return:<12.2%}")
    print(f"  Annualised volatility:  {result.annualised_volatility:<12.2%}")
    print(f"  Sharpe ratio:           {result.sharpe_ratio:<12.2f}")


if __name__ == "__main__":
    cfg = load_config()
    bt = cfg["backtest"]
    st = cfg["strategies"]
    pm = cfg["permutation"]
    ex = cfg["execution"]

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading price data...")
    t0 = time.perf_counter()
    tickers = build_universe(cfg)
    prices = load_data(
        tickers = tickers,
        start_date = bt["start_date"],
        end_date = bt["end_date"]
    )
    print(f"Loaded {len(prices.index.get_level_values('Date').unique())} trading days "
          f"for {len(tickers)} tickers.\n")
    print(f"Data loaded in {time.perf_counter() - t0:.2f} seconds.")
    metadata = {
        "tickers":         tickers,
        "start_date":      bt["start_date"],
        "end_date":        bt["end_date"],
        "initial_capital": bt["initial_capital"],
        "strategy":        bt["strategy"],
    }

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------
    print("Running backtest...")
    t1 = time.perf_counter()
    engine = BacktestEngine(
        prices = prices,
        strategy = build_strategy(cfg),
        metadata = metadata,
        initial_capital = bt["initial_capital"]
    )
    result = engine.run_backtest()
    print_result(to_label(bt["strategy"]), result)
    print(f"Backtest completed in {time.perf_counter() - t1:.2f} seconds.")

    # ------------------------------------------------------------------
    # Permutation test
    # ------------------------------------------------------------------
    print(f"\n\nRunning permutation test ({pm['scheme']}, N={pm['n']})...")
    print("This will take a few minutes.\n")
    t2 = time.perf_counter()
    scheme_cls, scheme_kwargs = build_scheme(cfg)

    perm_test = PermutationTest(
        prices = prices,
        strategy = build_strategy(cfg),
        scheme_cls = scheme_cls,
        scheme_kwargs = scheme_kwargs,
        N = pm["n"],
        metric = pm["metric"],
        initial_capital=bt["initial_capital"],
        seed = pm["seed"],
        n_jobs = pm["n_jobs"]
    )
    perm_results = perm_test.run()
    
    null_sharpes = [r.sharpe_ratio for r in perm_results.null_distribution]
    print("\n--- Permutation Test Results ---")
    print(f"  Scheme:                 {perm_results.scheme}")
    print(f"  N permutations:         {perm_results.N:<8d}")
    print(f"  Baseline Sharpe:        {perm_results.baseline.sharpe_ratio:<8.2f}")
    print(f"  Mean null Sharpe:       {np.mean(null_sharpes):<8.2f}")
    print(f"  Null Sharpe std:        {np.std(null_sharpes):<8.2f}")
    print(f"  p-value (one-tailed):   {perm_results.p_value:<8.4f}")
    if perm_results.p_value < 0.05:
        print("  Interpretation: Statistically significant at the 5% level.")
    elif perm_results.p_value < 0.10:
        print("  Interpretation: Marginal significance at the 10% level.")
    else:
        print("  Interpretation: Not statistically significant. Cannot reject the null hypothesis of no predictive power.")
    print(f"Permutation test completed in {time.perf_counter() - t2:.2f} seconds.")