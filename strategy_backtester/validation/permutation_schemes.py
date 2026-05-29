from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from strategy_backtester.data import PriceDataFrame, get_field, make_price_dataframe


class BasePermutationScheme(ABC):
    @abstractmethod
    def permute(self, prices: PriceDataFrame, rng: np.random.Generator) -> PriceDataFrame:
        """
        Given a price dataframe, return a permuted version of the dataframe.

        The permutation should preserve the structure of the dataframe (same
        columns, same index), but shuffle the data in a way that breaks any
        temporal dependence. Examples include shuffling returns, circularly
        shifting returns, or shuffling residuals from a factor model.

        Parameters
        ----------
        prices : PriceDataFrame
            Original price dataframe to be permuted.
        rng : np.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        PriceDataFrame
            Permuted price dataframe with the same structure as the input.
        """
        ...


    def _reconstruct_prices(
            self,
            original: PriceDataFrame,
            permuted_returns: pd.DataFrame,
            rng: np.random.Generator
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
        rng : np.random.Generator
            Random number generator for volume shuffling.
        """
        orig_close = get_field(original, "Close")
        orig_open = get_field(original, "Open")
        orig_high = get_field(original, "High")
        orig_low = get_field(original, "Low")
        orig_volume = get_field(original, "Volume")

        # Reconstruct close prices from permuted returns
        start_prices = orig_close.iloc[0]
        permuted_close = (1 + permuted_returns).cumprod() * start_prices

        # Scale O, H, L by the same factor as C to preserve intraday structure
        scaling_factors = permuted_close / orig_close
        permuted_open = orig_open * scaling_factors
        permuted_high = orig_high * scaling_factors
        permuted_low = orig_low * scaling_factors
        
        # Shuffle volume independently
        permuted_volume = orig_volume.copy()
        for col in orig_volume.columns:
            permuted_idx = rng.permutation(len(orig_volume))
            permuted_volume[col] = orig_volume[col].values[permuted_idx]
        
        # Combine into a new PriceDataFrame
        permuted_df = pd.concat([
            permuted_open.stack().rename("Open"),
            permuted_high.stack().rename("High"),
            permuted_low.stack().rename("Low"),
            permuted_close.stack().rename("Close"),
            permuted_volume.stack().rename("Volume")
        ], axis=1)
        permuted_df.index.names = ["Date", "Ticker"]
        permuted_df["Volume"] = permuted_df["Volume"].astype('int64')
        return make_price_dataframe(permuted_df)


class RanksScheme(BasePermutationScheme):
    """Shuffle cross-sectional rank order per period."""
    def permute(self, prices: PriceDataFrame, rng: np.random.Generator) -> PriceDataFrame:
        close = get_field(prices, "Close")
        returns = close.pct_change().fillna(0.0)

        # Permute returns independently for each date
        permuted_matrix = np.array([
            returns.values[i, rng.permutation(returns.shape[1])] for i in range(len(returns))
        ])
        shuffled_returns = pd.DataFrame(permuted_matrix, index=returns.index, columns=returns.columns)
        shuffled_returns.iloc[0] = 0.0

        return self._reconstruct_prices(prices, shuffled_returns, rng)


class IIDScheme(BasePermutationScheme):
    """Shuffle daily returns independently across time, per ticker."""
    def permute(self, prices: PriceDataFrame, rng: np.random.Generator) -> PriceDataFrame:
        close = get_field(prices, "Close")
        returns = close.pct_change().fillna(0.0)

        # Permute returns independently for each ticker
        shuffled_returns = returns.copy()
        for ticker in returns.columns:
            shuffled_idx = rng.permutation(len(returns) - 1) + 1  # Keep first return as 0.0 to anchor start price
            shuffled_returns[ticker].iloc[1:] = returns[ticker].iloc[shuffled_idx].values

        return self._reconstruct_prices(prices, shuffled_returns, rng)
    

class BlockScheme(BasePermutationScheme):
    """Shuffle contiguous blocks of returns, preserving intra-block structure."""
    def __init__(self, block_size: int = 20):
        self.block_size = block_size


    def permute(self, prices: PriceDataFrame, rng: np.random.Generator) -> PriceDataFrame:
        close = get_field(prices, "Close")
        returns = close.pct_change().fillna(0.0)
    
        # Permute returns by blocks independently for each ticker
        shuffled_returns = returns.copy()
        n = len(returns)
        block_starts = np.arange(1, n, self.block_size)
        block_indices = [np.arange(start, min(start + self.block_size, n)) for start in block_starts]
        for ticker in returns.columns:
            block_order = rng.permutation(len(block_indices))
            shuffled_idx = np.concatenate([block_indices[i] for i in block_order])
            shuffled_returns[ticker].iloc[1:] = returns[ticker].iloc[shuffled_idx].values

        return self._reconstruct_prices(prices, shuffled_returns, rng)