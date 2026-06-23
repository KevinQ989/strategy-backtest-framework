from __future__ import annotations
import os
import io
import time
import yaml
import pickle
import urllib.request
import argparse
from strategy_backtester.data import load_data, PriceDataFrame
from strategy_backtester.engine import BacktestEngine
from strategy_backtester.core import (
    BacktestResult,
    PermutationResult,
    WalkForwardResult,
)
from strategy_backtester.strategies import (
    RandomStrategy,
    CrossSectionalMomentumStrategy,
)
from strategy_backtester.validation import (
    PermutationTest,
    RankPermutationStrategy,
    IIDPermutationStrategy,
    BlockPermutationStrategy,
    WalkForwardTest,
    RollingWindowScheme,
    ExpandingWindowScheme,
)
from strategy_backtester.results import (
    generate_dashboard,
    generate_tear_sheet,
)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
CACHE_DIR   = os.path.join(os.path.dirname(__file__), "cache")

_STRATEGY_CLASSES = {
    "random":                   RandomStrategy,
    "cross_sectional_momentum": CrossSectionalMomentumStrategy,
}

_CACHE_FILES = {
    "backtest":    "backtest_results.pkl",
    "permutation": "perm_results.pkl",
    "walk_forward": "wfv_results.pkl",
}


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strategy backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py                         # run everything\n"
            "  python main.py --skip                  # load all from cache\n"
            "  python main.py --skip-perm --skip-wf   # re-run backtest only\n"
        ),
    )
    parser.add_argument(
        "--skip", "-s",
        action="store_true",
        help="Skip all computation and load all results from cache. "
             "Equivalent to --skip-backtest --skip-perm --skip-wf.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip backtest and load result from cache.",
    )
    parser.add_argument(
        "--skip-perm",
        action="store_true",
        help="Skip permutation test and load result from cache.",
    )
    parser.add_argument(
        "--skip-wf",
        action="store_true",
        help="Skip walk-forward validation and load result from cache.",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------

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
            headers={"User-Agent": "Mozilla/5.0 (compatible; strategy-backtest-framework)"},
        )
        with urllib.request.urlopen(req) as response:
            html = response.read()
        import pandas as pd
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
            f"Unknown strategy: {name!r}\n"
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
    raise ValueError(
        f"Unknown permutation scheme: {scheme_name!r}\n"
        f"Available schemes: 'ranks', 'iid', 'block'"
    )


def build_wf_scheme(cfg: dict):
    scheme_name = cfg["walk_forward"]["scheme"]
    win_in  = cfg["walk_forward"]["win_in"]
    win_out = cfg["walk_forward"]["win_out"]
    if scheme_name == "expanding":
        return ExpandingWindowScheme(win_in=win_in, win_out=win_out)
    elif scheme_name == "rolling":
        return RollingWindowScheme(win_in=win_in, win_out=win_out)
    raise ValueError(
        f"Unknown walk-forward scheme: {scheme_name!r}\n"
        f"Available schemes: 'expanding', 'rolling'"
    )


def build_param_grid(cfg: dict) -> dict[str, list]:
    strategy_name = cfg["backtest"]["strategy"]
    wf = cfg["walk_forward"]
    if strategy_name not in wf:
        raise ValueError(
            f"No parameter grid found for strategy {strategy_name!r} "
            f"in walk_forward config."
        )
    return wf[strategy_name]


# ------------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------------

def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, _CACHE_FILES[key])


def _load_cache(key: str):
    path = _cache_path(key)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"--skip-{key.replace('_', '-')} was specified but cache file not found: {path}\n"
            f"Run without the skip flag first to generate it."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_cache(key: str, obj) -> None:
    path = _cache_path(key)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Cached to {os.path.basename(path)}")


# ------------------------------------------------------------------
# Runners
# ------------------------------------------------------------------

def backtest_runner(
    tickers: list[str],
    prices: PriceDataFrame,
    cfg: dict,
    skip: bool,
) -> BacktestResult:
    if skip:
        print("Loading backtest result from cache...")
        return _load_cache("backtest")

    bt = cfg["backtest"]
    metadata = {
        "tickers":         tickers,
        "start_date":      bt["start_date"],
        "end_date":        bt["end_date"],
        "initial_capital": bt["initial_capital"],
        "strategy":        bt["strategy"],
    }
    print("Running backtest...")
    t = time.perf_counter()
    engine = BacktestEngine(
        prices          = prices,
        strategy        = build_strategy(cfg),
        metadata        = metadata,
        initial_capital = bt["initial_capital"],
    )
    result = engine.run_backtest()
    print(f"Backtest completed in {time.perf_counter() - t:.2f}s")
    _save_cache("backtest", result)
    return result


def permutation_runner(
    prices: PriceDataFrame,
    cfg: dict,
    skip: bool,
) -> PermutationResult:
    if skip:
        print("Loading permutation result from cache...")
        return _load_cache("permutation")

    pm = cfg["permutation"]
    print(f"\nRunning permutation test ({pm['scheme']}, N={pm['n']})...")
    print("This will take a few minutes.\n")
    t = time.perf_counter()
    scheme_cls, scheme_kwargs = build_perm_scheme(cfg)
    perm_test = PermutationTest(
        prices          = prices,
        strategy        = build_strategy(cfg),
        scheme_cls      = scheme_cls,
        scheme_kwargs   = scheme_kwargs,
        N               = pm["n"],
        metric          = pm["metric"],
        initial_capital = cfg["backtest"]["initial_capital"],
        seed            = pm["seed"],
        n_jobs          = pm["n_jobs"],
    )
    result = perm_test.run()
    print(f"Permutation test completed in {time.perf_counter() - t:.2f}s")
    _save_cache("permutation", result)
    return result


def walk_forward_runner(
    prices: PriceDataFrame,
    cfg: dict,
    skip: bool,
) -> WalkForwardResult:
    if skip:
        print("Loading walk-forward result from cache...")
        return _load_cache("walk_forward")

    wf = cfg["walk_forward"]
    bt = cfg["backtest"]
    print("\nRunning walk-forward validation...")
    print("This will take a few minutes.\n")
    t = time.perf_counter()
    wfv = WalkForwardTest(
        prices          = prices,
        strategy_cls    = _STRATEGY_CLASSES[bt["strategy"]],
        param_grid      = build_param_grid(cfg),
        scheme          = build_wf_scheme(cfg),
        metric          = wf["metric"],
        initial_capital = bt["initial_capital"],
        n_jobs          = wf["n_jobs"],
    )
    result = wfv.run()
    print(f"Walk-forward validation completed in {time.perf_counter() - t:.2f}s")
    _save_cache("walk_forward", result)
    return result


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config()

    # Resolve effective skip flags — --skip sets all three
    skip_bt   = args.skip or args.skip_backtest
    skip_perm = args.skip or args.skip_perm
    skip_wf   = args.skip or args.skip_wf

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Only fetch price data if at least one runner needs it
    prices  = None
    tickers = None
    if not (skip_bt and skip_perm and skip_wf):
        bt = cfg["backtest"]
        print("Loading price data...")
        t0 = time.perf_counter()
        tickers = build_universe(cfg)
        prices = load_data(
            tickers    = tickers,
            start_date = bt["start_date"],
            end_date   = bt["end_date"],
        )
        n_days = len(prices.index.get_level_values("Date").unique())
        print(f"Loaded {n_days} trading days for {len(tickers)} tickers "
              f"in {time.perf_counter() - t0:.2f}s\n")

    # ------------------------------------------------------------------
    # Run (or load) each component
    # ------------------------------------------------------------------
    result      = backtest_runner(tickers, prices, cfg, skip=skip_bt)
    perm_result = permutation_runner(prices, cfg, skip=skip_perm)
    wfv_result  = walk_forward_runner(prices, cfg, skip=skip_wf)

    # ------------------------------------------------------------------
    # Generate outputs
    # ------------------------------------------------------------------
    # print("\nGenerating dashboard...")
    # generate_dashboard(perm_result, cfg["dashboard"]["rolling_sharpe_window"])
    print("Generating tear sheet...")
    generate_tear_sheet(result, perm_result, wfv_result, cfg)