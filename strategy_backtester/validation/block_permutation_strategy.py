from __future__ import annotations
import numpy as np
import pandas as pd
from strategy_backtester.data import PriceDataFrame, get_field, make_price_dataframe
from strategy_backtester.strategies import BaseStrategy
from .permutation_strategy import PermutationStrategyWrapper


class BlockPermutationStrategy(PermutationStrategyWrapper):
    def __init__(self, strategy: BaseStrategy, rng: np.random.Generator, block_size: int = 20):
        super().__init__(strategy, rng)
        self.block_size = block_size


    def _reconstruct_prices(
            self,
            original: PriceDataFrame,
            permuted_returns: pd.DataFrame
        ) -> PriceDataFrame:
        """""
        Reconstruct a valid OHLCV PriceDataFrame from permuted Close returns.

        All inputs and intermediate computations are in wide (Date × Ticker)
        format. Output is stacked back to (Date, Ticker) MultiIndex long format
        and validated via make_price_dataframe.

        O, H, L are scaled by the same daily return as C to preserve intraday
        structure. Volume is shuffled independently per ticker.

        Parameters
        ----------
        original : PriceDataFrame
            Original price data in (Date, Ticker) MultiIndex format.
        permuted_returns : pd.DataFrame
            Permuted daily returns in wide (Date × Ticker) format.
            Row 0 must be 0.0 (no return on first day — anchors start price).
        """
        orig_close = get_field(original, "Close")
        orig_open = get_field(original, "Open")
        orig_high = get_field(original, "High")
        orig_low = get_field(original, "Low")
        orig_adj_close = get_field(original, "Adj_Close")
        orig_volume = get_field(original, "Volume")

        # Reconstruct close prices from permuted returns
        start_prices = orig_close.iloc[0]
        permuted_close = (1 + permuted_returns).cumprod() * start_prices

        # Scale O, H, L by the same factor as C to preserve intraday structure
        scaling_factors = permuted_close / orig_close
        permuted_open = orig_open * scaling_factors
        permuted_high = orig_high * scaling_factors
        permuted_low = orig_low * scaling_factors
        permuted_adj_close = orig_adj_close * scaling_factors
        
        # Shuffle volume independently
        permuted_volume = orig_volume.copy()
        for col in orig_volume.columns:
            permuted_idx = self.rng.permutation(len(orig_volume))
            permuted_volume[col] = orig_volume[col].values[permuted_idx]
        
        # Combine into a new PriceDataFrame
        permuted_df = pd.concat([
            permuted_open.stack().rename("Open"),
            permuted_high.stack().rename("High"),
            permuted_low.stack().rename("Low"),
            permuted_close.stack().rename("Close"),
            permuted_adj_close.stack().rename("Adj_Close"),
            permuted_volume.stack().rename("Volume")
        ], axis=1)
        permuted_df.index.names = ["Date", "Ticker"]
        permuted_df["Volume"] = permuted_df["Volume"].astype('int64')
        return make_price_dataframe(permuted_df)
    

    def prepare(self, prices: PriceDataFrame) -> PriceDataFrame:
        close = get_field(prices, "Close")
        returns = close.pct_change().fillna(0.0)
    
        # Permute returns by blocks independently for each ticker
        shuffled_returns = returns.copy()
        n = len(returns)
        block_starts = np.arange(1, n, self.block_size)
        block_indices = [np.arange(start, min(start + self.block_size, n)) for start in block_starts]
        for ticker in returns.columns:
            block_order = self.rng.permutation(len(block_indices))
            shuffled_idx = np.concatenate([block_indices[i] for i in block_order])
            non_first_dates = returns.index[1:]
            shuffled_dates = returns.index[shuffled_idx]
            shuffled_returns.loc[non_first_dates, ticker] = returns.loc[shuffled_dates, ticker].values

        return self._reconstruct_prices(prices, shuffled_returns)