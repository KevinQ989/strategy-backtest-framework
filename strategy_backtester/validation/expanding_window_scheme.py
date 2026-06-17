from __future__ import annotations
import pandas as pd
from strategy_backtester.data import PriceDataFrame
from .window_scheme import WindowScheme


class ExpandingWindowScheme(WindowScheme):
    """
    Anchored expanding in-sample window.

    IS window starts at a fixed origin and grows by win_out at each fold.
    OOS window steps forward by win_out. Fold k (1-based) has:
        is_start  = dates[0]
        is_end    = dates[win_in + (k-1) * win_out - 1]
        oos_start = dates[win_in + (k-1) * win_out]
        oos_end   = dates[win_in + k * win_out - 1]
    """
    def split(
        self,
        prices: PriceDataFrame,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        dates = prices.index.get_level_values("Date").unique().sort_values()
        folds = []
        k = 1
        while True:
            is_end_idx    = self.win_in + (k - 1) * self.win_out - 1
            oos_start_idx = self.win_in + (k - 1) * self.win_out
            oos_end_idx   = self.win_in + k * self.win_out - 1
            if oos_end_idx >= len(dates):
                break
            folds.append((
                dates[0],
                dates[is_end_idx],
                dates[oos_start_idx],
                dates[oos_end_idx],
            ))
            k += 1
        return folds