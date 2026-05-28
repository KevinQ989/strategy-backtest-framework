import pandas as pd
import numpy as np
from .base import BaseStrategy
from strategy_backtester.core.types import PortfolioWeights

class RandomStrategy(BaseStrategy):
    def should_rebalance(
        self,
        date: pd.Timestamp,
        last_rebalance: pd.Timestamp,
        current_weights: pd.Series,
        prices: pd.DataFrame
    ) -> bool:
        return True #rebalance everyday for maximum chaos
    
    def generate(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        current_weights: pd.Series
    ) -> PortfolioWeights:

        #Get list of unique ticker names
        tickers = prices.index.get_level_values('Ticker').unique()

        #Generate random weights between -1 (Short) and +1 (Long), for each ticker
        random_values = np.random.uniform(-1,1,size = len(tickers))
        raw_weights = pd.Series(random_values, index = tickers)

        longs = raw_weights[raw_weights > 0]
        shorts = raw_weights[raw_weights < 0]

        #Normalise weights so total exposure is 1
        total_exposure = longs.sum() + shorts.abs().sum()

        if total_exposure > 0:
            longs = longs/total_exposure
            shorts = shorts/total_exposure

        return PortfolioWeights(
            date = as_of,
            long_weights = longs,
            short_weights = shorts
        )