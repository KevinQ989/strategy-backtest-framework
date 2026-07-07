from __future__ import annotations
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from strategy_backtester.core import (
    BacktestResult,
    ParamResult,
    WalkForwardFold,
    WalkForwardResult,
)
from strategy_backtester.data import PriceDataFrame
from strategy_backtester.engine import BacktestEngine
from strategy_backtester.strategies import BaseStrategy
from strategy_backtester.results import calc_sharpe_ratio
from .window_scheme import WindowScheme


_SUPPORTED_METRICS = ("sharpe",)


def _evaluate_params_worker(
    strategy_cls: type[BaseStrategy],
    params: dict,
    prices: PriceDataFrame,
    metric: str,
    initial_capital: float,
    strict: bool = False
) -> float:
    """
    Instantiate strategy_cls with params, run a backtest over prices, and
    return the requested performance metric.

    This function must be defined at module level rather than as a method or
    closure so that ProcessPoolExecutor can pickle it for dispatch to worker
    processes. Mirrors the top-level worker pattern used in permutation.py.

    Any exception raised during strategy instantiation, engine construction,
    or backtest execution is caught and returns -inf. This allows the fold's
    IS selection logic to skip broken combinations (e.g. win_in too short for
    the given lookback) without crashing the entire validation run.

    Parameters
    ----------
    strategy_cls : type[BaseStrategy]
        Strategy class to instantiate with params as constructor kwargs.
    params : dict
        Constructor keyword arguments for strategy_cls, e.g.
        {"lookback": 252, "skip": 21, "percent": 0.1, "rebalance_freq": 21}.
    prices : PriceDataFrame
        Price slice for the window being evaluated (IS or OOS).
    metric : str
        Performance metric to compute and return. Must be one of
        _SUPPORTED_METRICS.
    initial_capital : float
        Starting capital for BacktestEngine.
    strict : bool, default False
        If True, any exception raised during strategy instantiation, engine
        construction, or backtest execution is propagated to the caller rather
        than returning -inf. Used for OOS evaluation of the selected parameter.

    Returns
    -------
    float
        Performance metric value for this param combination on this price
        slice, or -inf if any exception is raised.

    Raises
    ------
    RuntimeError
        If strategy instantiation, engine construction, or backtest execution
        raises any exception.
    """
    try:
        strategy = strategy_cls(**params)
        engine = BacktestEngine(prices, strategy, initial_capital=initial_capital)
        result = engine.run_backtest()
        return _compute_metric(result, metric)
    except Exception as e:
        if strict:
            raise RuntimeError(
                f"OOS evaluation failed for params {params}: {e}"
            ) from e
        else:
            return float("-inf")


def _compute_metric(result: BacktestResult, metric: str) -> float:
    """
    Compute a named performance metric from a BacktestResult.

    Parameters
    ----------
    result : BacktestResult
        Completed backtest result.
    metric : str
        Name of the metric to compute. Currently supports "sharpe".

    Returns
    -------
    float
        Metric value.

    Raises
    ------
    ValueError
        If metric is not in _SUPPORTED_METRICS.
    """
    if metric == "sharpe":
        try:
            return calc_sharpe_ratio(result.returns)
        except ValueError:
            return float("-inf")
    raise ValueError(
        f"Unsupported metric: {metric!r}. Supported: {_SUPPORTED_METRICS}"
    )


class WalkForwardTest:
    """
    Runs walk-forward validation with full parameter grid search.

    For each fold, every parameter combination in the Cartesian product of
    param_grid is evaluated on the IS window in parallel. The combination
    with the best IS metric is selected and evaluated on the OOS window.
    Folds run sequentially with a progress line printed to stdout after each.

    The walk-forward procedure simulates realistic out-of-sample deployment:
    parameters are selected on past data only and evaluated on the immediately
    following held-out period, preventing lookahead bias. Comparing per-fold
    OOS metrics against IS metrics reveals whether parameter selection is
    exploiting in-sample noise (overfitting) or capturing a stable signal.

    Parameters
    ----------
    prices : PriceDataFrame
        Full price history. Fold windows are sliced from this.
    strategy_cls : type[BaseStrategy]
        Strategy class to instantiate per grid combination. Must accept
        the grid parameters as constructor keyword arguments.
    param_grid : dict[str, list]
        Parameter grid as {param_name: [value1, value2, ...]}. The Cartesian
        product of all lists defines the combinations evaluated per fold.
        Example::

            {
                "lookback": [126, 252, 504],
                "percent":  [0.05, 0.1, 0.2],
            }

    scheme : WindowScheme
        Instantiated window scheme that defines fold boundaries. Use
        ExpandingWindowScheme or RollingWindowScheme.
    metric : str, default "sharpe"
        Performance metric for IS parameter selection and OOS evaluation.
        Currently supports "sharpe".
    initial_capital : float, default 100_000.0
        Starting capital passed to each BacktestEngine instance. Each fold's
        IS and OOS backtests begin with this capital (cold-start — positions
        are not carried across fold boundaries).
    n_jobs : int, default 1
        Number of parallel worker processes for within-fold IS grid search.
        Set to -1 to use all available CPUs. Folds always run sequentially;
        only the grid combinations within each fold are parallelised.

    Raises
    ------
    ValueError
        If metric is not in _SUPPORTED_METRICS.
        If param_grid is empty or produces no combinations.
        If "lookback" is a key in param_grid and win_in is less than the
        maximum lookback value — the IS window would be too short for the
        strategy to generate any signal, producing meaningless IS metrics.
        If the price history is too short to produce at least one complete
        fold under the given scheme.

    Notes
    -----
    Each fold's OOS backtest is run as an independent BacktestEngine with no
    carried-over positions. The first rebalance of each OOS window incurs a
    full position-building cost rather than an incremental rebalance cost.
    This cold-start distortion is bounded (one rebalance per fold transition)
    and should be documented as a limitation when interpreting OOS metrics.
    """

    def __init__(
        self,
        prices: PriceDataFrame,
        strategy_cls: type[BaseStrategy],
        param_grid: dict[str, list],
        scheme: WindowScheme,
        metric: str = "sharpe",
        initial_capital: float = 100_000.0,
        n_jobs: int = 1,
    ) -> None:
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric!r}. Supported: {_SUPPORTED_METRICS}"
            )
        if not param_grid:
            raise ValueError("param_grid must not be empty.")

        keys, values = zip(*param_grid.items())
        self.param_combinations = [
            dict(zip(keys, combo)) for combo in itertools.product(*values)
        ]
        if not self.param_combinations:
            raise ValueError("param_grid must produce at least one combination.")

        if "lookback" in param_grid:
            max_lookback = max(param_grid["lookback"])
            if scheme.win_in <= max_lookback:
                raise ValueError(
                    f"win_in={scheme.win_in} is less than or equal to the maximum lookback in "
                    f"param_grid ({max_lookback}). The IS window must be longer than the longest lookback, "
                    f"otherwise the strategy cannot generate a signal and IS metric selection is meaningless."
                )

        unique_dates = prices.index.get_level_values("Date").unique()
        min_required = scheme.win_in + scheme.win_out
        if len(unique_dates) < min_required:
            raise ValueError(
                f"Price history has only {len(unique_dates)} trading days, which is "
                f"insufficient for one complete fold (win_in={scheme.win_in} + "
                f"win_out={scheme.win_out} = {min_required} required)."
            )

        self.prices = prices
        self.strategy_cls = strategy_cls
        self.scheme = scheme
        self.metric = metric
        self.initial_capital = initial_capital
        self.n_jobs = n_jobs if n_jobs > 0 else None


    def run(self) -> WalkForwardResult:
        """
        Execute the full walk-forward validation and return aggregated results.

        Calls scheme.split() to obtain fold boundaries, then iterates over
        folds sequentially. For each fold, _run_fold() parallelises IS grid
        evaluation and runs the OOS evaluation on the selected params.
        Prints "Fold k / N complete" to stdout after each fold.

        Returns
        -------
        WalkForwardResult
            Contains the window scheme name, metric name, and a list of
            WalkForwardFold results in chronological order.

        Raises
        ------
        ValueError
            If scheme.split() returns no folds (price history too short
            given the configured win_in and win_out).
        """
        fold_boundaries = self.scheme.split(self.prices)
        if not fold_boundaries:
            raise ValueError(
                f"No complete folds could be constructed from the price history "
                f"with win_in={self.scheme.win_in} and win_out={self.scheme.win_out}. "
                f"Use a shorter window or provide more price history."
            )

        n_folds = len(fold_boundaries)
        folds = []
        for k, (is_start, is_end, oos_start, oos_end) in enumerate(fold_boundaries, start=1):
            fold = self._run_fold(k, is_start, is_end, oos_start, oos_end)
            folds.append(fold)
            print(f"Fold {k} / {n_folds} complete")

        return WalkForwardResult(
            scheme=type(self.scheme).__name__,
            metric=self.metric,
            folds=folds,
        )


    def _slice_prices(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> PriceDataFrame:
        """
        Return the subset of self.prices whose Date index level falls within
        [start, end] inclusive.

        Slices by boolean mask on the Date level rather than .loc[start:end]
        to handle MultiIndex reliably without dependence on sort order or
        exact label matching at boundaries.

        Parameters
        ----------
        start : pd.Timestamp
            First date to include (inclusive).
        end : pd.Timestamp
            Last date to include (inclusive).

        Returns
        -------
        PriceDataFrame
            Subset of self.prices restricted to [start, end].

        Raises
        ------
        ValueError
            If the resulting slice is empty (no trading dates in range).
        """
        dates = self.prices.index.get_level_values("Date")
        mask = (dates >= start) & (dates <= end)
        sliced = self.prices.loc[mask]
        if sliced.empty:
            raise ValueError(
                f"No price data found in the range [{start}, {end}]. "
                f"Check that fold boundaries fall within the available price history."
            )
        return sliced


    def _run_fold(
        self,
        fold_idx: int,
        is_start: pd.Timestamp,
        is_end: pd.Timestamp,
        oos_start: pd.Timestamp,
        oos_end: pd.Timestamp,
    ) -> WalkForwardFold:
        """
        Run a single fold: parallel IS grid search followed by OOS evaluation.

        Submits all parameter combinations to a ProcessPoolExecutor for
        parallel IS evaluation. Once all IS results are collected, selects
        the combination with the highest IS metric and runs a single OOS
        backtest on the held-out window.

        Parameters
        ----------
        fold_idx : int
            1-based fold index, stored on the returned WalkForwardFold.
        is_start : pd.Timestamp
            First date of the IS window (inclusive).
        is_end : pd.Timestamp
            Last date of the IS window (inclusive).
        oos_start : pd.Timestamp
            First date of the OOS window (inclusive).
        oos_end : pd.Timestamp
            Last date of the OOS window (inclusive).

        Returns
        -------
        WalkForwardFold
            Contains IS grid results for all param combinations, the selected
            params, and the OOS metric for the selected params.

        Raises
        ------
        RuntimeError
            If all IS parameter combinations return -inf (every combination
            failed, e.g. IS window universally too short for all lookbacks).
            If OOS evaluation of the selected params raises an exception.
        """
        is_prices = self._slice_prices(is_start, is_end)
        oos_prices = self._slice_prices(oos_start, oos_end)

        # Run IS grid search in parallel
        param_results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(
                    _evaluate_params_worker,
                    self.strategy_cls,
                    params,
                    is_prices,
                    self.metric,
                    self.initial_capital,
                ): params
                for params in self.param_combinations
            }
            for future in as_completed(futures):
                param_results.append(
                    ParamResult(
                        params=futures[future],
                        is_metric=future.result(),
                    )
                )

        # Check for universal failure on IS window
        if all(pr.is_metric == float("-inf") for pr in param_results):
            raise RuntimeError(
                f"Fold {fold_idx}: all {len(param_results)} parameter combinations "
                f"returned -inf on the IS window [{is_start}, {is_end}]. "
                f"The IS window may be too short for all lookback values in the grid."
            )

        # Run OOS evaluation on the selected parameters
        selected = max(param_results, key=lambda pr: pr.is_metric)
        oos_metric = _evaluate_params_worker(
            self.strategy_cls,
            selected.params,
            oos_prices,
            self.metric,
            self.initial_capital,
            strict=True,
        )

        return WalkForwardFold(
            fold_idx=fold_idx,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            param_results=param_results,
            selected_params=selected.params,
            oos_metric=oos_metric,
        )