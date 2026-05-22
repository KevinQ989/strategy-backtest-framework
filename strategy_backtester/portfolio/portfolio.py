from __future__ import annotations
from strategy_backtester.data import PriceDataFrame, get_date
from strategy_backtester.core import ExecutionResult
import pandas as pd

class PortfolioState:
    """
    Single source of truth for portfolio state during simulation.
 
    Tracks cash, share positions, and the most recent price reference.
    No external component may directly modify positions or cash — all
    mutations flow through this class's methods.
 
    Positions are stored as shares (not dollar values). Dollar values
    and weights are derived on demand from shares × _last_prices.
 
    Attributes
    ----------
    date : pd.Timestamp
        Current simulation date. Advanced by update_to_market.
    cash : float
        Uninvested capital in dollars.
    starting_capital : float
        Constant reference value for CAGR and drawdown calculations.
    positions : pd.Series
        ticker -> shares held. Positive = long, negative = short.
    _last_prices : pd.Series
        ticker -> most recent adj_close price. Updated by update_to_market.
        Used to compute total_value and current_weights without requiring
        external price input on every access.
    """

    def __init__(self, date: pd.Timestamp, starting_capital: float):
        self.date = date
        self.cash = starting_capital
        self.starting_capital = starting_capital
        self.positions = pd.Series(dtype=float)    # ticker -> shares
        self._last_prices = pd.Series(dtype=float) # ticker -> price

    @property
    def total_value(self) -> float:
        """
        Total portfolio value in dollars.
 
        Computed as cash + sum(shares_i × last_price_i).
        Returns cash only when no positions are held.
        """
        if self.positions.empty:
            return self.cash
        else:
            return self.cash + (self.positions * self._last_prices).sum()
    

    @property
    def current_weights(self) -> pd.Series:
        """
        Current portfolio weights as fractions of total value.
 
        Computed as (shares_i × last_price_i) / total_value.
        Returns empty Series when no positions are held or total_value is zero.
 
        Returns
        -------
        pd.Series
            ticker -> float weight. Positive = long, negative = short.
        """
        if self.positions.empty or self.total_value == 0:
            return pd.Series(dtype=float)
        else:
            return (self.positions * self._last_prices) / self.total_value
        
    
    def update_to_market(self, prices: PriceDataFrame, date: pd.Timestamp) -> None:
        """
        Update internal price reference and advance the simulation date.
 
        Must be called at the end of every trading day before any weight
        generation or execution. Share counts do not change — only
        _last_prices is updated, which causes total_value and current_weights
        to reflect current market prices on the next access.
 
        Parameters
        ----------
        prices : PriceDataFrame
            Columns: Open, High, Low, Close, Volume.
            Close is split and dividend adjusted (auto_adjust=True).
        date : pd.Timestamp
            Current simulation date T. Must exist in prices index level 'Date'.
        """
        self.date = date
        self._last_prices = get_date(prices, date)["Close"]

    
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
        self.positions = self.positions.add(result.fills, fill_value=0.0)
        self.positions = self.positions[self.positions != 0]

        # Update cash based on fills and costs
        fill_cost = (result.fills * result.execution_prices.reindex(result.fills.index)).sum()
        self.cash -= fill_cost
        self.cash -= result.total_cost * pre_trade_value

    
    def __repr__(self) -> str:
        return (
            f"PortfolioState("
            f"date={self.date.date()}, "
            f"total_value={self.total_value:,.2f}, "
            f"cash={self.cash:,.2f}, "
            f"positions={self.positions.to_dict()}"
            f")"
        )