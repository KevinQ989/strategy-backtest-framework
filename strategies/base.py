from core.types import PortfolioWeights
from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    @abstractmethod
    def generate(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        current_weights: pd.Series
    ) -> PortfolioWeights:
        """
        Generate target portfolio weights as of a given date.

        The engine guarantees that prices contains only data up to and
        including as_of. Implementations must not access any data beyond
        this boundary — the guarantee is structural, not a convention.

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV price history up to and including as_of.
            MultiIndex columns: (field, ticker), where field is one of
            open, high, low, close, volume, adj_close.
            Index: pd.DatetimeIndex, daily frequency, timezone-naive.
        as_of : pd.Timestamp
            Date T. Weights will be used to trade at T+1 open.
        current_weights : pd.Series
            Current portfolio weights, ticker -> float.
            Derived from PortfolioState.current_weights.
            Empty series at T=0.

        Returns
        -------
        PortfolioWeights
            Target weights for T+1. Weights are fractions of total
            portfolio value, not dollar amounts.
        """
        ...


    @abstractmethod
    def should_rebalance(
        self,
        date: pd.Timestamp,
        last_rebalance: pd.Timestamp,
        current_weights: pd.Series,
        prices: pd.DataFrame
    ) -> bool:
        """
        Determine whether the strategy should rebalance on the given date.

        Implementations may use any combination of the inputs — a simple
        calendar strategy will only use date and last_rebalance, while a
        drift-based strategy may inspect current_weights or prices.

        Parameters
        ----------
        date : pd.Timestamp
            Current simulation date.
        last_rebalance : pd.Timestamp
            Date of the most recent rebalance.
        current_weights : pd.Series
            Current portfolio weights, ticker -> float.
        prices : pd.DataFrame
            Full price history up to and including date. Same schema
            as described in generate().

        Returns
        -------
        bool
            True if the strategy should rebalance today.
        """
        ...