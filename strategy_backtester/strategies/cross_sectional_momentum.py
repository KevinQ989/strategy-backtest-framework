from __future__ import annotations
import pandas as pd
from .base import BaseStrategy
from strategy_backtester.core import PortfolioWeights
from strategy_backtester.data import PriceDataFrame, get_field


class CrossSectionalMomentumStrategy(BaseStrategy):
    """
    Cross-sectional momentum strategy.

    Ranks all tickers by their 12-1 month return (12-month cumulative return
    excluding the most recent month) at each rebalance date. Goes long the top
    decile and short the bottom decile with equal weighting within each leg.
    Rebalances monthly.

    The 12-1 month return is defined as the cumulative return from 252 trading
    days ago to 21 trading days ago, deliberately excluding the most recent
    month to avoid short-term reversal contamination.

    Parameters
    ----------
    lookback : int
        Number of trading days defining the start of the momentum window.
        Default is 252 (approximately 12 months).
    skip : int
        Number of most recent trading days to exclude from the signal.
        Default is 21 (approximately 1 month).
    decile : float
        Fraction of the universe assigned to each leg. Default is 0.1 (top
        and bottom 10%). Must satisfy 2 * decile <= 1.0.
    rebalance_freq : int
        Minimum number of calendar days between rebalances. Default is 21
        (approximately monthly).
    """
    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        decile: float = 0.1,
        rebalance_freq: int = 21
    ):
        if not (0 < decile <= 0.5):
            raise ValueError("decile must be in the range (0, 0.5]")
        self.lookback = lookback
        self.skip = skip
        self.decile = decile
        self.rebalance_freq = rebalance_freq
    

    def generate(
        self,
        prices: PriceDataFrame,
        as_of: pd.Timestamp,
        current_weights: pd.Series
    ) -> PortfolioWeights:
        """
        Generate equal-weighted long/short portfolio weights based on
        12-1 month cross-sectional momentum.

        Computes the momentum signal for each ticker as the cumulative return
        over the window [T - lookback, T - skip], where T = as_of. Tickers
        are ranked by this signal. The top decile is assigned equal positive
        weights; the bottom decile is assigned equal negative weights.

        The engine guarantees that prices contains only data up to and
        including as_of. No data beyond as_of is accessed.

        Parameters
        ----------
        prices : PriceDataFrame
            OHLCV price history up to and including as_of.
            MultiIndex (Date, Ticker) on rows. Columns: Open, High,
            Low, Close, Volume (title case, flat — not a column MultiIndex).
            Use get_field(prices, "Close") to get a (Date × Ticker) matrix.
            Close prices are split and dividend adjusted (auto_adjust=True).
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
        close = get_field(prices, "Close")
        dates = close.index

        # Ensure we have enough history to compute the signal
        current_idx = dates.get_loc(as_of)
        if current_idx < self.lookback:
            return PortfolioWeights(
                date = as_of,
                long_weights = pd.Series(dtype=float),
                short_weights = pd.Series(dtype=float)
            )

        # Compute momentum signal: cumulative return from T - lookback to T - skip
        price_start = close.iloc[current_idx - self.lookback]
        price_end = close.iloc[current_idx - self.skip]
        momentum = (price_end / price_start - 1.0).dropna()
        if momentum.empty:
            return PortfolioWeights(
                date = as_of,
                long_weights = pd.Series(dtype=float),
                short_weights = pd.Series(dtype=float)
            )
        
        # Check that universe is large enough to form deciles
        q = int(len(momentum) * self.decile)
        if q < 2:
            raise ValueError(
                f"Universe too small to construct momentum deciles. "
                f"Need at least {int(2 / self.decile)} tickers with valid signals, "
                f"got {len(momentum)}."
            )

        # Rank tickers by momentum in descending order (1 = highest momentum)
        ranked = momentum.rank(method='first', ascending=False)
        n = len(momentum)

        # Assign long weights to top decile
        long_tickers = ranked[ranked <= q].index
        long_weights = pd.Series(+1.0 / q, index=long_tickers)

        # Assign short weights to bottom decile
        short_tickers = ranked[ranked > n - q].index
        short_weights = pd.Series(-1.0 / q, index=short_tickers)
        
        return PortfolioWeights(
            date = as_of,
            long_weights = long_weights,
            short_weights = short_weights
        )


    def should_rebalance(
        self,
        date: pd.Timestamp,
        last_rebalance: pd.Timestamp,
        current_weights: pd.Series,
        prices: PriceDataFrame
    ) -> bool:
        """
        Return True if at least rebalance_freq trading days have elapsed
        since the last rebalance.

        Uses a simple trading-day threshold. prices and current_weights are
        accepted for interface compatibility but are not used — this strategy
        rebalances on a fixed trading-day schedule regardless of portfolio drift
        or market conditions.

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
        dates = prices.index.get_level_values('Date').unique().sort_values()
        current_idx = dates.get_loc(date)
        last_idx = dates.get_loc(last_rebalance)
        return (current_idx - last_idx) >= self.rebalance_freq