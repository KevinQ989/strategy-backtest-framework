from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

# Result of strategies handed to portfolio
@dataclass
class PortfolioWeights:
    date: pd.Timestamp       # Date the weights were computed
    long_weights: pd.Series  # Target weights to long with ticker labels
    short_weights: pd.Series # Target weights to short with ticker labels


    @property
    def combined(self) -> pd.Series:
        return pd.concat([self.long_weights, self.short_weights])
    

    @property
    def gross_exposure(self) -> float:
        return self.combined.abs().sum()


# Result of execution handed to backtest
@dataclass
class ExecutionResult:
    date: pd.Timestamp # Date that execution occurred
    fills: pd.Series   # Shares transacted with ticker labels (+ long, - short)
    execution_prices: pd.Series # Prices at which trades were executed
    turnover: float    # Fraction of portfolio that traded
    slippage: float    # Slippage component of costs
    commission: float  # Commission component of costs
    spread: float      # Spread component of costs


    @property
    def total_cost(self) -> float:
        return self.slippage + self.commission + self.spread


# Result of backtesting handed to validation and results
@dataclass
class BacktestResult:
    returns: pd.Series      # Daily portfolio returns, index = date
    positions: pd.DataFrame # Daily positions, index = date, cols = tickers
    costs: pd.Series        # Daily total cost, index = date
    turnover: pd.Series     # Daily turnover, index = date
    starting_capital: float # Initial capital
    metadata: dict          # Config snapshot