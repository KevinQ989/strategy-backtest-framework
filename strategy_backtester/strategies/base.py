from __future__ import annotations
from strategy_backtester.core import PortfolioWeights
from strategy_backtester.data import PriceDataFrame
from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    def prepare(
        self,
        prices: PriceDataFrame
    ) -> PriceDataFrame:
        """
        ...
        
        Parameters
        ----------
        prices : PriceDataFrame
            OHLCV price history for the entire backtest period.
        
        Returns
        -------
        PriceDataFrame
            Preprocessed OHLCV price history for the entire backtest period.
        """    
        return prices
    

    @abstractmethod
    def generate(
        self,
        prices: PriceDataFrame,
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
        prices : PriceDataFrame
            OHLCV price history up to and including as_of.
            MultiIndex (Date, Ticker) on rows.
            Columns: Open, High, Low, Close, Volume, Adj_Close.
            Use get_field(prices, "Close") to get a (Date × Ticker) matrix.
            Index is pd.DatetimeIndex, daily frequency, timezone-naive.
        as_of : pd.Timestamp
            Date T. Weights will be used to trade at T+1 open.
        current_weights : pd.Series
            Current portfolio weights at end of day T, ticker -> float.
            Positive = long, negative = short.
            Derived from PortfolioState.current_weights.
            Empty series on the first call (no positions held).

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
        prices: PriceDataFrame
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
        prices : PriceDataFrame
            Full price history up to and including date. Same schema
            as described in generate().

        Returns
        -------
        bool
            True if the strategy should rebalance today.
        """
        ...