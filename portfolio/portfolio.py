from core.types import ExecutionResult
import pandas as pd

class PortfolioState:
    def __init__(self, date: pd.Timestamp, starting_capital: float):
        self.date = date
        self.cash = starting_capital
        self.starting_capital = starting_capital
        self.positions = pd.Series(dtype=float)    # ticker -> shares
        self._last_prices = pd.Series(dtype=float) # ticker -> price

    @property
    def total_value(self) -> float:
        if self.positions.empty:
            return self.cash
        else:
            return self.cash + (self.positions * self._last_prices).sum()
    

    @property
    def current_weights(self) -> pd.Series:
        if self.positions.empty or self.total_value == 0:
            return pd.Series(dtype=float)
        else:
            return (self.positions * self._last_prices) / self.total_value
        
    
    def update_to_market(self, prices: pd.DataFrame, date: pd.Timestamp) -> None:
        """
        Update portfolio state to reflect market changes up to the given date.

        Must be called at the start of every trading day before any
        weight generation or execution. Share counts do not change —
        only _last_prices is updated, which causes total_value and
        current_weights to reflect current market prices.

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV price history up to and including date.
            MultiIndex columns: (field, ticker), where field is one of
            open, high, low, close, volume, adj_close.
            Index: pd.DatetimeIndex, daily frequency, timezone-naive.
        date : pd.Timestamp
            Current simulation date T.
        """
        self.date = date
        self._last_prices = prices.loc[date, 'adj_close']

    
    def update_to_execution(self, result: ExecutionResult) -> None:
        """
        Update portfolio state to reflect the results of an execution.

        Must be called after mark_to_market on the same day so that
        _last_prices is current for accurate dollar fill computation.
        Costs are deducted as a fraction of pre-trade portfolio value.

        Parameters
        ----------
        result : ExecutionResult
            The result of the day's execution, including fills and costs.
        """
        pre_trade_value = self.total_value

        # Update positions based on fills        
        for ticker, shares in result.fills.items():
            if ticker in self.positions:
                self.positions[ticker] += shares
            else:
                self.positions[ticker] = shares
        self.positions = self.positions[self.positions != 0]

        # Update cash based on fills and costs
        self.cash -= (result.fills * self._last_prices.reindex(result.fills.index)).sum()
        self.cash -= result.costs * pre_trade_value

    
    def __repr__(self) -> str:
        return (
            f"PortfolioState("
            f"date={self.date.date()}, "
            f"total_value={self.total_value:,.2f}, "
            f"cash={self.cash:,.2f}, "
            f"positions={self.positions.to_dict()}"
        )