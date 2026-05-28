from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from strategy_backtester.data import PriceDataFrame


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


class RanksScheme(BasePermutationScheme):
    """Shuffle cross-sectional rank order per period."""
    def permute(self, prices: PriceDataFrame, rng: np.random.Generator) -> PriceDataFrame:
        ...


class IIDScheme(BasePermutationScheme):
    """Shuffle daily returns independently across time, per ticker."""
    def permute(self, prices: PriceDataFrame, rng: np.random.Generator) -> PriceDataFrame:
        ...


class BlockScheme(BasePermutationScheme):
    """Shuffle contiguous blocks of returns, preserving intra-block structure."""
    def __init__(self, block_size: int = 20):
        self.block_size = block_size


    def permute(self, prices: PriceDataFrame, rng: np.random.Generator) -> PriceDataFrame:
        ...