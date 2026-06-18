from __future__ import annotations
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from strategy_backtester.core import BacktestResult, PermutationResult
from strategy_backtester.data import PriceDataFrame
from strategy_backtester.engine import BacktestEngine
from strategy_backtester.strategies import BaseStrategy
from strategy_backtester.results import calc_sharpe_ratio
from .permutation_strategy import PermutationStrategyWrapper


def _run_single_permutation(args: tuple) -> BacktestResult:
        """
        Helper for running a single permutation in parallel.

        Parameters
        ----------
        args : tuple
            (prices, strategy, scheme, initial_capital, seed)

        Returns
        -------
        BacktestResult
            The result of the backtest on the permuted data.
        """
        prices, base_strategy, scheme_cls, scheme_kwargs, initial_capital, seed = args
        rng = np.random.default_rng(seed)
        wrapped_strategy = scheme_cls(base_strategy, rng, **scheme_kwargs)
        engine = BacktestEngine(prices, wrapped_strategy, initial_capital=initial_capital)
        return engine.run_backtest()


class PermutationTest:
    def __init__(
        self,
        prices: PriceDataFrame,
        strategy: BaseStrategy,
        scheme_cls: type[PermutationStrategyWrapper],
        scheme_kwargs: dict,
        N: int = 1000,
        metric: str = "sharpe",
        initial_capital: float = 100000.0,
        seed: int = 42,
        n_jobs: int = 1,
    ):
        self.prices = prices
        self.strategy = strategy
        self.scheme_cls = scheme_cls
        self.scheme_kwargs = scheme_kwargs
        self.N = N
        self.metric = metric
        self.initial_capital = initial_capital
        self.seed = seed
        self.n_jobs = n_jobs
        self.rng = np.random.default_rng(seed)
        self.permutation_results = None


    def run(self) -> PermutationResult:
        #Run backtest on original data
        print("Running baseline backtest...")
        engine = BacktestEngine(self.prices, self.strategy, initial_capital=self.initial_capital)
        baseline_result = engine.run_backtest()
        baseline_metric = self._calculate_metric(baseline_result)

        # Pre-generate seeds for parallel execution
        ss = np.random.SeedSequence(self.seed)
        child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(self.N)]
        args = [
            (self.prices, self.strategy, self.scheme_cls, self.scheme_kwargs, self.initial_capital, seed)
            for seed in child_seeds
        ]

        # Run permutations in parallel
        print(f"Running {self.N} permutations with {self.n_jobs} parallel jobs...")
        null_distribution = []
        null_metrics = []
        if self.n_jobs == 1:
             for i, arg in enumerate(args):
                result = _run_single_permutation(arg)
                null_distribution.append(result)
                null_metrics.append(self._calculate_metric(result))
                if (i + 1) % 100 == 0:
                    print(f"Completed {i + 1}/{self.N} permutations")
        else:
            n_workers = self.n_jobs if self.n_jobs > 0 else None
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_run_single_permutation, arg): i for i, arg in enumerate(args)}
                for future in as_completed(futures):
                    result = future.result()
                    null_distribution.append(result)
                    null_metrics.append(self._calculate_metric(result))
                    completed = len(null_metrics)
                    if completed % 100 == 0:
                        print(f"Completed {completed}/{self.N} permutations")
        
        # Calculate one-tailed p-value
        p_value = float(np.mean(np.array(null_metrics) >= baseline_metric)) 

        self.permutation_results = PermutationResult(
            baseline = baseline_result,
            null_distribution = null_distribution,
            p_value = p_value,
            metric = self.metric,
            N = self.N,
            scheme = self.scheme_cls.__name__
        )
        return self.permutation_results
    

    def _calculate_metric(self, backtest_result: BacktestResult) -> float:
        """Calculate the chosen performance metric from backtest results."""
        if self.metric == "sharpe":
            return calc_sharpe_ratio(backtest_result.returns)
        raise ValueError(f"Unsupported metric: {self.metric}")