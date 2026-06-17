from __future__ import annotations
import pandas as pd
from strategy_backtester.data import PriceDataFrame
from .window_scheme import WindowScheme


class RollingWindowScheme(WindowScheme):
    """
    Fixed-length rolling in-sample window.

    Both IS and OOS windows have fixed lengths and slide forward by win_out
    at each fold. IS window always covers the win_in trading days immediately
    preceding the OOS window. Fold k (1-based) has:
        is_start  = dates[(k-1) * win_out]
        is_end    = dates[(k-1) * win_out + win_in - 1]
        oos_start = dates[(k-1) * win_out + win_in]
        oos_end   = dates[(k-1) * win_out + win_in + win_out - 1]
    """

    def split(
        self,
        prices: PriceDataFrame,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        ...