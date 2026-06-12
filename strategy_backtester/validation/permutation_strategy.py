from __future__ import annotations
import numpy as np
import pandas as pd
from strategy_backtester.core import PortfolioWeights
from strategy_backtester.data import PriceDataFrame
from strategy_backtester.strategies import BaseStrategy


class PermutationStrategyWrapper(BaseStrategy):
    """
    Base class for strategies that wrap another strategy to construct a
    null-hypothesis backtest for permutation testing.

    By default, prepare(), generate(), and should_rebalance() all delegate
    unchanged to the wrapped strategy — a subclass overrides whichever
    method(s) implement its particular permutation scheme (e.g. reshuffling
    prices in prepare(), or reshuffling signal-to-ticker assignment in
    generate()).

    Parameters
    ----------
    strategy : BaseStrategy
        The strategy being tested. Its prepare/generate/should_rebalance
        behaviour is used as-is except where overridden by the subclass.
    rng : np.random.Generator
        Random number generator used for this permutation. Each permutation
        should be constructed with its own independently-seeded generator.
    """
    def __init__(self, strategy: BaseStrategy, rng: np.random.Generator):
        self.strategy = strategy
        self.rng = rng

    
    def prepare(
        self,
        prices: PriceDataFrame
    ) -> PriceDataFrame:
        return self.strategy.prepare(prices)
    

    def generate(
        self,
        prices: PriceDataFrame,
        as_of: pd.Timestamp,
        current_weights: pd.Series
    ) -> PortfolioWeights:
        return self.strategy.generate(prices, as_of, current_weights)
    

    def should_rebalance(
        self,
        date: pd.Timestamp,
        last_rebalance: pd.Timestamp,
        current_weights: pd.Series,
        prices: PriceDataFrame
    ) -> bool:
        return self.strategy.should_rebalance(date, last_rebalance, current_weights, prices)