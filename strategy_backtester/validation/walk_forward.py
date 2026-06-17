from __future__ import annotations
import pandas as pd
from strategy_backtester.core import ParamResult, WalkForwardFold, WalkForwardResult
from strategy_backtester.data import PriceDataFrame
from strategy_backtester.strategies import BaseStrategy
from .window_scheme import WindowScheme


def _evaluate_params_worker(
    strategy_cls: type[BaseStrategy],
    params: dict,
    prices: PriceDataFrame,
    metric: str,
    initial_capital: float,
) -> float:
    """
    Instantiate strategy_cls with params, run BacktestEngine over prices,
    and return the requested metric.

    Must be module-level for pickle compatibility with ProcessPoolExecutor.
    Returns -inf on any exception so the fold's selection logic can skip
    this combination without crashing the validation run.

    Parameters
    ----------
    strategy_cls : type[BaseStrategy]
        Strategy class to instantiate.
    params : dict
        Constructor kwargs for strategy_cls.
    prices : PriceDataFrame
        Price slice for the window being evaluated.
    metric : str
        Metric to compute and return, e.g. "sharpe".
    initial_capital : float
        Starting capital for BacktestEngine.

    Returns
    -------
    float
        Metric value, or -inf on failure.
    """
    ...


class WalkForwardTest:
    """
    Runs walk-forward validation with full parameter grid search.

    For each fold, all parameter combinations in the grid are evaluated
    on the IS window in parallel. The combination with the best IS metric
    is selected and evaluated on the OOS window. Folds run sequentially
    with progress printed to stdout.

    Parameters
    ----------
    prices : PriceDataFrame
        Full price history.
    strategy_cls : type[BaseStrategy]
        Strategy class to instantiate per grid combination. Must accept
        the grid parameters as constructor keyword arguments.
    param_grid : dict[str, list]
        Parameter grid as {param_name: [value1, value2, ...]}. The
        Cartesian product defines all combinations evaluated per fold.
    scheme : WindowScheme
        Instantiated window scheme defining fold boundaries.
    metric : str
        Performance metric used for IS selection and OOS evaluation.
        Currently supports "sharpe".
    initial_capital : float
        Starting capital passed to each BacktestEngine. Default 100_000.
    n_workers : int | None
        Number of parallel workers for within-fold grid evaluation.
        None uses os.cpu_count().

    Raises
    ------
    ValueError
        If metric is not supported.
        If param_grid is empty or produces no combinations.
        If win_in < max(lookback) in param_grid, where lookback is a
        recognised strategy parameter — would produce empty IS signals.
        If the price history is too short for even one complete fold.
    """

    def __init__(
        self,
        prices: PriceDataFrame,
        strategy_cls: type[BaseStrategy],
        param_grid: dict[str, list],
        scheme: WindowScheme,
        metric: str = "sharpe",
        initial_capital: float = 100_000.0,
        n_workers: int | None = None,
    ) -> None:
        ...


    def run(self) -> WalkForwardResult:
        """
        Execute full walk-forward validation.

        Calls scheme.split() to get fold boundaries, then for each fold
        calls _run_fold(). Prints "Fold k / N complete" after each fold.

        Returns
        -------
        WalkForwardResult
        """
        ...


    def _slice_prices(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> PriceDataFrame:
        """
        Return the subset of self.prices whose Date index level falls
        within [start, end] inclusive.

        Parameters
        ----------
        start : pd.Timestamp
            First date to include.
        end : pd.Timestamp
            Last date to include.

        Returns
        -------
        PriceDataFrame
            Slice of self.prices. Raises ValueError if the result is empty.
        """
        ...


    def _run_fold(
        self,
        fold_idx: int,
        is_start: pd.Timestamp,
        is_end: pd.Timestamp,
        oos_start: pd.Timestamp,
        oos_end: pd.Timestamp,
    ) -> WalkForwardFold:
        """
        Run a single fold: parallel IS grid search, then OOS evaluation
        of the selected parameter combination.

        Parameters
        ----------
        fold_idx : int
            1-based fold index, stored on the returned WalkForwardFold.
        is_start, is_end : pd.Timestamp
            IS window boundaries passed to _slice_prices.
        oos_start, oos_end : pd.Timestamp
            OOS window boundaries passed to _slice_prices.

        Returns
        -------
        WalkForwardFold
        """
        ...
