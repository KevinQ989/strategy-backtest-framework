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
    quintile and short the bottom quintile with equal weighting within each leg.
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
    percent : float
        Fraction of the universe assigned to each leg. Default is 0.1 (top
        and bottom 10%). Must satisfy 2 * percent <= 1.0.
    rebalance_freq : int
        Minimum number of calendar days between rebalances. Default is 21
        (approximately monthly).
    """
    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        percent: float = 0.1,
        rebalance_freq: int = 21
    ):
        if not (0 < percent <= 0.5):
            raise ValueError("percent must be in the range (0, 0.5]")
        self.lookback = lookback
        self.skip = skip
        self.percent = percent
        self.rebalance_freq = rebalance_freq


    def _compute_signal(
        self,
        prices: PriceDataFrame,
        as_of: pd.Timestamp
    ) -> pd.Series:
        """
        Compute the 12-1 month momentum signal for each ticker as of a given date.

        The signal is the cumulative return over the window [T - lookback, T - skip],
        where T = as_of, computed on Adj_Close. Tickers with insufficient history
        or missing data are excluded.

        Parameters
        ----------
        prices : PriceDataFrame
            OHLCV price history up to and including as_of.
            MultiIndex (Date, Ticker) on rows.
            Columns: Open, High, Low, Close, Volume, Adj_Close.
            Index is pd.DatetimeIndex, daily frequency, timezone-naive.
        as_of : pd.Timestamp
            Date T. Signal is computed using only data up to and including T.

        Returns
        -------
        pd.Series
            Momentum signal, indexed by ticker. Empty if there is insufficient
            history (current_idx < lookback) or no tickers have a valid signal.
        """
        adj_close = get_field(prices, "Adj_Close")
        dates = adj_close.index

        # Ensure we have enough history to compute the signal
        current_idx = dates.get_loc(as_of)
        if current_idx < self.lookback:
            return pd.Series(dtype=float)

        # Compute momentum signal: cumulative return from T - lookback to T - skip
        price_start = adj_close.iloc[current_idx - self.lookback]
        price_end = adj_close.iloc[current_idx - self.skip]
        momentum = (price_end / price_start - 1.0).dropna()

        return momentum


    def _weights_from_signal(
        self,
        signal: pd.Series,
        as_of: pd.Timestamp
    ) -> PortfolioWeights:
        """
        Convert a momentum signal into equal-weighted long/short portfolio weights.

        Ranks tickers by signal in descending order. The top percent is assigned
        equal positive weights (long); the bottom percent is assigned equal
        negative weights (short). Weights are scaled by half-Kelly (0.5/q per leg).

        Parameters
        ----------
        signal : pd.Series
            Momentum signal, indexed by ticker. May be empty.
        as_of : pd.Timestamp
            Date T. Returned weights are for T+1.

        Returns
        -------
        PortfolioWeights
            Target weights for T+1. Empty long/short weights if signal is empty.

        Raises
        ------
        ValueError
            If the signal has fewer than 2 * (1 / percent) tickers, i.e. too
            few to construct non-empty long and short legs.
        """
        if signal.empty:
            return PortfolioWeights(
                date = as_of,
                long_weights = pd.Series(dtype=float),
                short_weights = pd.Series(dtype=float)
            )

        # Check that universe is large enough to form long/short legs
        q = int(len(signal) * self.percent)
        if q < 2:
            raise ValueError(
                f"Universe too small to construct momentum portfolios. "
                f"Need at least {int(2 / self.percent)} tickers with valid signals, "
                f"got {len(signal)}."
            )

        # Rank tickers by signal in descending order (1 = highest signal)
        ranked = signal.rank(method='first', ascending=False)
        n = len(signal)

        # Assign long weights to top percent
        long_tickers = ranked[ranked <= q].index
        long_weights = pd.Series(+0.5 / q, index=long_tickers)

        # Assign short weights to bottom percent
        short_tickers = ranked[ranked > n - q].index
        short_weights = pd.Series(-0.5 / q, index=short_tickers)

        return PortfolioWeights(
            date = as_of,
            long_weights = long_weights,
            short_weights = short_weights
        )


    def generate(
        self,
        prices: PriceDataFrame,
        as_of: pd.Timestamp,
        current_weights: pd.Series
    ) -> PortfolioWeights:
        """
        Generate equal-weighted long/short portfolio weights based on
        12-1 month cross-sectional momentum.

        The engine guarantees that prices contains only data up to and
        including as_of. No data beyond as_of is accessed.

        Parameters
        ----------
        prices : PriceDataFrame
            OHLCV price history up to and including as_of.
            MultiIndex (Date, Ticker) on rows.
            Columns: Open, High, Low, Close, Volume, Adj_Close.
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
        signal = self._compute_signal(prices, as_of)
        return self._weights_from_signal(signal, as_of)


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