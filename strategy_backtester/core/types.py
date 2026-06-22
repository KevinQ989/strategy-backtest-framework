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


# Result of permutation test handed to validation
@dataclass
class PermutationResult:
    baseline: BacktestResult                # Backtest result from original data
    null_distribution: list[BacktestResult] # List of backtest results from permuted data
    p_value: float                          # One-tailed p-value of observed performance vs null
    metric: str                             # Performance metric used for evaluation
    N: int                                  # Number of permutations
    scheme: str                             # Name of the permutation scheme used


# Result of in-sample validation handed to walk-forward fold
@dataclass
class ParamResult:
    params: dict     # Hyperparameters used for backtest
    is_metric: float # In-sample performance metric


# Result of walk-forward fold handed to walk-forward test
@dataclass
class WalkForwardFold:
    fold_idx: int                    # Index of the fold
    is_start: pd.Timestamp           # Start date of in-sample period
    is_end: pd.Timestamp             # End date of in-sample period
    oos_start: pd.Timestamp          # Start date of out-of-sample period
    oos_end: pd.Timestamp            # End date of out-of-sample period
    param_results: list[ParamResult] # List of in-sample validation results for different hyperparameters
    selected_params: dict            # Hyperparameters selected based on in-sample validation
    oos_metric: float                # Out-of-sample metric for selected hyperparameters


# Result of walk-forward test handed to validation
@dataclass
class WalkForwardResult:
    scheme: str                  # Name of the walk-forward scheme used
    metric: str                  # Performance metric used for evaluation
    folds: list[WalkForwardFold] # List of walk-forward folds