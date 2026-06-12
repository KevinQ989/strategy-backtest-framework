from __future__ import annotations
import pandas as pd
from strategy_backtester.core import PortfolioWeights
from strategy_backtester.data import PriceDataFrame
from .permutation_strategy import PermutationStrategyWrapper


class RankPermutationStrategy(PermutationStrategyWrapper):
    def generate(
        self,
        prices: PriceDataFrame,
        as_of: pd.Timestamp,
        current_weights: pd.Series
    ) -> PortfolioWeights:
        # Get the original strategy's weights
        original_weights = self.strategy.generate(prices, as_of, current_weights)
        
        # Permute weights by rank independently for each ticker
        permuted_weights = original_weights.copy()
        for ticker in original_weights.index:
            ranks = original_weights[ticker].rank(method='first')
            permuted_ranks = self.rng.permutation(ranks)
            permuted_weights[ticker] = original_weights[ticker].iloc[permuted_ranks.argsort()]
        
        return permuted_weights