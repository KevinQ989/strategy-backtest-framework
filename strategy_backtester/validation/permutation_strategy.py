from __future__ import annotations
import numpy as np
import pandas as pd
from strategy_backtester.core import PortfolioWeights
from strategy_backtester.data import PriceDataFrame
from strategy_backtester.strategies import BaseStrategy


class PermutationStrategyWrapper(BaseStrategy):
    """
    ...

    Parameters
    ----------
    strategy : BaseStrategy
        The original strategy to be wrapped. This strategy will be trained and
        evaluated on permuted price data.
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