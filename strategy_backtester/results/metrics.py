import numpy as np
import pandas as pd

def calc_cumulative_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return (1+returns).prod() - 1

def calc_final_value(initial_value: float, returns: pd.Series) -> float:
    return initial_value * (1 + calc_cumulative_return(returns))

def calc_annualised_return(initial_value: float, returns: pd.Series) -> float:
    final_value = calc_final_value(initial_value, returns)
    days = len(returns)
    return (final_value/initial_value) ** (1/(days/252)) - 1

def calc_annualised_volatility(returns: pd.Series) -> float:
    sd = returns.std()
    return sd * np.sqrt(252)

#Risk free rate of 4%
def calc_sharpe_ratio(returns: pd.Series) -> float:
    return (calc_cumulative_return(returns) - 0.04)/(returns.std())

