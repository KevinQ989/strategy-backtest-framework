from __future__ import annotations
from dataclasses import dataclass
from strategy_backtester.core import BacktestResult
from strategy_backtester.data import PriceDataFrame
from strategy_backtester.engine import BacktestEngine
from strategy_backtester.strategies import BaseStrategy
from .permutation_schemes import BasePermutationScheme


@dataclass
class PermutationResult:
    baseline: BacktestResult
    null_distribution: list[BacktestResult]
    p_value: float
    metric: str
    N: int
    scheme: str


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
        ...


    def run(self) -> PermutationResult:
        ...
