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
        dates = prices.index.get_level_values("Date").unique().sort_values()
        folds = []
        k = 1
        while True:
            is_start_idx  = (k - 1) * self.win_out
            is_end_idx    = (k - 1) * self.win_out + self.win_in - 1
            oos_start_idx = (k - 1) * self.win_out + self.win_in
            oos_end_idx   = (k - 1) * self.win_out + self.win_in + self.win_out - 1
            if oos_end_idx >= len(dates):
                break
            folds.append((
                dates[is_start_idx],
                dates[is_end_idx],
                dates[oos_start_idx],
                dates[oos_end_idx],
            ))
            k += 1
        return folds