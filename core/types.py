from dataclasses import dataclass
import pandas as pd

# Result of strategies handed to portfolio
@dataclass
class PortfolioWeights:
    date: pd.Timestamp # Date the weights were computed
    long_weights: pd.Series # Weights to long with ticker labels
    short_weights: pd.Series # Weights to short with ticker labels


# Result of execution handed to backtest
@dataclass
class ExecutionResult:
    date: pd.Timestamp # Date that execution occurred
    fills: pd.Series # Weights after execution filled with ticker labels
    costs: float # Total cost
    turnover: float # Fraction of portfolio that traded
    slippage: float # Slippage component of costs
    commission: float # Commission component of costs


# Result of backtesting handed to validation and results
@dataclass
class BacktestResult:
    returns: pd.Series # Daily portfolio returns with date time index
    positions: pd.DataFrame # Daily positions, index = date, cols = tickers
    costs: pd.Series # Daily total cost
    turnover: pd.Series # Daily turnover
    metadata: dict # Config snapshot