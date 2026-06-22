from __future__ import annotations
import numpy as np
import pandas as pd
import yaml
import os
import io
import time
import pickle
import urllib.request
from strategy_backtester.data import load_data
from strategy_backtester.results.plots import plot_dashboard
from strategy_backtester.engine import BacktestEngine
from strategy_backtester.core import (
    BacktestResult,
    PermutationResult,
    WalkForwardResult
)
from strategy_backtester.strategies import(
    RandomStrategy,
    CrossSectionalMomentumStrategy
)
from strategy_backtester.validation import (
    PermutationTest,
    RankPermutationStrategy,
    IIDPermutationStrategy,
    BlockPermutationStrategy,
    WalkForwardTest,
    RollingWindowScheme,
    ExpandingWindowScheme
)


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
_STRATEGY_CLASSES = {
    "random": RandomStrategy,
    "cross_sectional_momentum": CrossSectionalMomentumStrategy
}


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
    if name not in _STRATEGY_CLASSES:
        raise ValueError(
            f"Unknown strategy: {name}\n"
            f"Available strategies: {list(_STRATEGY_CLASSES.keys())}"
        )
    params = cfg["strategies"].get(name, {})
    return _STRATEGY_CLASSES[name](**params)


def build_perm_scheme(cfg: dict):
    scheme_name = cfg["permutation"]["scheme"]
    if scheme_name == "ranks":
        return RankPermutationStrategy, {}
    elif scheme_name == "iid":
        return IIDPermutationStrategy, {}
    elif scheme_name == "block":
        return BlockPermutationStrategy, {"block_size": cfg["permutation"]["block_size"]}
    else:
        raise ValueError(
            f"Unknown permutation scheme: {scheme_name}\n"
            f"Available schemes: 'ranks', 'iid', 'block'"
        )


def build_wf_scheme(cfg: dict):
    scheme_name = cfg["walk_forward"]["scheme"]
    win_in = cfg["walk_forward"]["win_in"]
    win_out = cfg["walk_forward"]["win_out"]
    if scheme_name == "expanding":
        return ExpandingWindowScheme(win_in=win_in, win_out=win_out)
    elif scheme_name == "rolling":
        return RollingWindowScheme(win_in=win_in, win_out=win_out)
    else:
        raise ValueError(
            f"Unknown walk-forward scheme: {scheme_name}\n"
            f"Available schemes: 'expanding', 'rolling'"
        )
    

def build_param_grid(cfg: dict) -> dict[str, list]:
    strategy_name = cfg["backtest"]["strategy"]
    wf = cfg["walk_forward"]
    if strategy_name not in wf:
        raise ValueError(
            f"No parameter grid found for strategy '{strategy_name}' in walk_forward config."
        )
    return wf[strategy_name]


# ------------------------------------------------------------------
# Print helpers
# ------------------------------------------------------------------


def to_label(name: str) -> str:
    return name.replace("_", " ").title()


def print_backtest_result(label: str, result: BacktestResult) -> None:
    print(f"\n--- {label} ---")
    print(f"  Final portfolio value:  ${result.final_value:<12,.2f}")
    print(f"  Cumulative return:      {result.cumulative_return:<12.2%}")
    print(f"  Annualised return:      {result.annualised_return:<12.2%}")
    print(f"  Annualised volatility:  {result.annualised_volatility:<12.2%}")
    print(f"  Sharpe ratio:           {result.sharpe_ratio:<12.2f}")


def print_permutation_result(result: PermutationResult) -> None:
    null_sharpes = [r.sharpe_ratio for r in result.null_distribution]
    print("\n--- Permutation Test Results ---")
    print(f"  Scheme:                 {result.scheme}")
    print(f"  N permutations:         {result.N:<8d}")
    print(f"  Baseline Sharpe:        {result.baseline.sharpe_ratio:<8.2f}")
    print(f"  Mean null Sharpe:       {np.mean(null_sharpes):<8.2f}")
    print(f"  Null Sharpe std:        {np.std(null_sharpes):<8.2f}")
    print(f"  p-value (one-tailed):   {result.p_value:<8.4f}")
    if result.p_value < 0.05:
        print("  Interpretation: Statistically significant at the 5% level.")
    elif result.p_value < 0.10:
        print("  Interpretation: Marginal significance at the 10% level.")
    else:
        print("  Interpretation: Not statistically significant. Cannot reject the null hypothesis of no predictive power.")


def print_wf_result(result: WalkForwardResult) -> None:
    print("\n--- Walk-Forward Validation Results ---")
    print(f"  Scheme:  {result.scheme}")
    print(f"  Folds:   {len(result.folds):<8d}")
    print(f"  Metric:  {result.metric:<8s}")
    print()
    header = f"  {'Fold':<6} {'IS Start':<12} {'IS End':<12} {'OOS Start':<12} {'OOS End':<12} {'IS Sharpe':<12} {'OOS Sharpe':<12} Selected Params"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for fold in result.folds:
        best_is = max(fold.param_results, key=lambda x: x.is_metric)
        params_str = ", ".join(f"{k}={v}" for k, v in fold.selected_params.items())
        print(
            f"  {fold.fold_idx:<6d} "
            f"{fold.is_start.date().strftime("%Y-%m-%d"):<12} "
            f"{fold.is_end.date().strftime("%Y-%m-%d"):<12} "
            f"{fold.oos_start.date().strftime("%Y-%m-%d"):<12} "
            f"{fold.oos_end.date().strftime("%Y-%m-%d"):<12} "
            f"{best_is.is_metric:<12.2f} "
            f"{fold.oos_metric:<12.2f} "
            f"{params_str}"
        )
    oos_metrics = [f.oos_metric for f in result.folds]
    print()
    print(f"  Mean OOS {result.metric}: {np.mean(oos_metrics):.2f}")
    print(f"  Std OOS {result.metric}:  {np.std(oos_metrics):.2f}")
    positive_folds = sum(1 for m in oos_metrics if m > 0)
    print(f"  Folds with positive OOS metric: {positive_folds} / {len(result.folds)}")


if __name__ == "__main__":
    cfg = load_config()
    bt = cfg["backtest"]
    pm = cfg["permutation"]
    wf = cfg["walk_forward"]

    # ------------------------------------------------------------------
    # Create cache directory if it doesn't exist
    # ------------------------------------------------------------------
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created cache directory at {cache_dir}\n")

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
    print_backtest_result(to_label(bt["strategy"]), result)
    print(f"Backtest completed in {time.perf_counter() - t1:.2f} seconds.")


    # ------------------------------------------------------------------
    # Permutation test
    # ------------------------------------------------------------------
    print(f"\n\nRunning permutation test ({pm['scheme']}, N={pm['n']})...")
    print("This will take a few minutes.\n")
    t2 = time.perf_counter()
    scheme_cls, scheme_kwargs = build_perm_scheme(cfg)

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

    # Cache permutation test results
    perm_test_data_path = os.path.join(cache_dir, "perm_results.pkl")
    with open(perm_test_data_path, "wb") as file:
        pickle.dump(perm_results, file)
    print("Permutation test data saved successfully.")

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    print("Generating Dashboard...")
    # plot_dashboard(perm_results, cfg['dashboard']['rolling_sharpe_window'])
    print_permutation_result(perm_results)
    print(f"Permutation test completed in {time.perf_counter() - t2:.2f} seconds.")


    # ------------------------------------------------------------------
    # Walk forward validation
    # ------------------------------------------------------------------
    print(f"\n\nRunning walk-forward validation...")
    print("This will take a few minutes.\n")
    t3 = time.perf_counter()
    wfv = WalkForwardTest(
        prices = prices,
        strategy_cls = _STRATEGY_CLASSES[bt["strategy"]],
        param_grid = build_param_grid(cfg),
        scheme = build_wf_scheme(cfg),
        metric = wf["metric"],
        initial_capital = bt["initial_capital"],
        n_jobs = wf["n_jobs"]
    )
    wfv_results = wfv.run()

    # Cache walk-forward validation results
    wfv_data_path = os.path.join(cache_dir, "wfv_results.pkl")
    with open(wfv_data_path, "wb") as file:
        pickle.dump(wfv_results, file)
    print("Walk-forward validation data saved successfully.")

    print_wf_result(wfv_results)
    print(f"Walk-forward validation completed in {time.perf_counter() - t3:.2f} seconds.")
