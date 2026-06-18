from __future__ import annotations
import numpy as np
import pandas as pd
from strategy_backtester.core import PortfolioWeights
from strategy_backtester.data import PriceDataFrame
from strategy_backtester.strategies import BaseStrategy
from .permutation_strategy import PermutationStrategyWrapper


class RankPermutationStrategy(PermutationStrategyWrapper):
    """
    Null hypothesis: the wrapped strategy's signal has no information content
    beyond its cross-sectional distribution — i.e. the *values* of the signal
    matter, but their assignment to tickers is random.

    Overrides generate() to compute the wrapped strategy's signal via
    strategy._compute_signal(...), randomly reassign the resulting signal
    values across tickers via rng.permutation, and then derive portfolio
    weights from this shuffled signal via strategy._weights_from_signal(...).
    prepare() and should_rebalance() are unchanged from the wrapped strategy.

    This tests whether the strategy's ranking of tickers carries genuine
    predictive information.

    Requires the wrapped strategy to implement _compute_signal and
    _weights_from_signal (e.g. CrossSectionalMomentumStrategy); raises
    TypeError at construction if these are not present.
    """
    def __init__(self, strategy: BaseStrategy, rng: np.random.Generator):
        if not (hasattr(strategy, "_compute_signal") and hasattr(strategy, "_weights_from_signal")):
            raise TypeError(
                f"{type(strategy).__name__} does not implement "
                f"_compute_signal/_weights_from_signal, required for "
                f"RankPermutationStrategy."
            )
        super().__init__(strategy, rng)

    def generate(
        self,
        prices: PriceDataFrame,
        as_of: pd.Timestamp,
        current_weights: pd.Series
    ) -> PortfolioWeights:
        signal = self.strategy._compute_signal(prices, as_of)
        if signal.empty:
            return self.strategy._weights_from_signal(signal, as_of)

        shuffled_values = self.rng.permutation(signal.values)
        shuffled_signal = pd.Series(shuffled_values, index=signal.index)

        return self.strategy._weights_from_signal(shuffled_signal, as_of)