from __future__ import annotations
import numpy as np
from strategy_backtester.core import BacktestResult, PermutationResult
from strategy_backtester.data import PriceDataFrame
from strategy_backtester.engine import BacktestEngine
from strategy_backtester.strategies import BaseStrategy
from strategy_backtester.results import calc_sharpe_ratio
from .permutation_schemes import BasePermutationScheme


class PermutationTest:
    def __init__(
        self,
        prices: PriceDataFrame,
        strategy: BaseStrategy,
        scheme: BasePermutationScheme,
        N: int = 1000,
        metric: str = "sharpe",
        initial_capital: float = 100000.0,
        seed: int = 42
    ):
        self.prices = prices
        self.strategy = strategy
        self.scheme = scheme
        self.N = N
        self.metric = metric
        self.initial_capital = initial_capital
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.permutation_results = None


    def run(self) -> PermutationResult:
        #Run backtest on original data
        engine = BacktestEngine(self.prices, initial_capital=self.initial_capital)
        baseline_result = engine.run_backtest(self.strategy)
        baseline_metric = self._calculate_metric(baseline_result)

        #Run backtests on permuted data
        null_distribution = []
        null_metrics = []
        for i in range(self.N):
            permuted_prices = self.scheme.permute(self.prices, self.rng)
            engine = BacktestEngine(permuted_prices, initial_capital=self.initial_capital)
            result = engine.run_backtest(self.strategy)
            null_distribution.append(result)
            null_metrics.append(self._calculate_metric(result))
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{self.N} permutations")
        
        # Calculate one-tailed p-value
        p_value = float(np.mean(np.array(null_metrics) >= baseline_metric)) 

        self.permutation_results = PermutationResult(
            baseline = baseline_result,
            null_distribution = null_distribution,
            p_value = p_value,
            metric = self.metric,
            N = self.N,
            scheme = self.scheme.__class__.__name__
        )
        return self.permutation_results
    

    def _calculate_metric(self, backtest_result: BacktestResult) -> float:
        """Calculate the chosen performance metric from backtest results."""
        if self.metric == "sharpe":
            return calc_sharpe_ratio(backtest_result.returns)
        raise ValueError(f"Unsupported metric: {self.metric}")