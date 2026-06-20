from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from strategy_backtester.data import PriceDataFrame


class WindowScheme(ABC):
    """
    Abstract base class for walk-forward window splitting schemes.

    Subclasses implement split() to define how the full price history is
    divided into IS/OOS fold boundaries under a particular windowing rule.

    Parameters
    ----------
    win_in : int
        In-sample window length in trading days.
    win_out : int
        Out-of-sample window length in trading days. Also the step size
        between consecutive folds.
    """
    def __init__(self, win_in: int, win_out: int) -> None:
        if win_in <= 0:
            raise ValueError(f"win_in must be positive, got {win_in}.")
        if win_out <= 0:
            raise ValueError(f"win_out must be positive, got {win_out}.")
        self.win_in = win_in
        self.win_out = win_out


    @abstractmethod
    def split(
        self,
        prices: PriceDataFrame,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Compute fold window boundaries from the full price history.

        Boundaries are derived from the sorted unique trading dates present
        in prices, guaranteeing every boundary date exists in the data.
        A final partial OOS window (shorter than win_out trading days) is
        dropped — only complete folds are returned.

        Parameters
        ----------
        prices : PriceDataFrame
            Full price history. Only the Date index level is used.

        Returns
        -------
        list of (is_start, is_end, oos_start, oos_end) tuples
            Chronologically ordered. is_end and oos_start are adjacent
            trading days (no gap, no overlap). All four dates are present
            in the prices index.
        """
        ...