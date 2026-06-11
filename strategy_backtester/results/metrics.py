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
    if final_value <= 0:
        raise ValueError(
            f"Portfolio value non-positive (final_value={final_value:.2f}). "
            "Simulation reached a state with no margin call / liquidation modeling. "
            "Annualised return is undefined."
        )
    days = len(returns)
    return (final_value / initial_value) ** (1 / (days / 252)) - 1


def calc_annualised_volatility(returns: pd.Series) -> float:
    sd = returns.std()
    return sd * np.sqrt(252)


#Risk free rate of 4%
def calc_sharpe_ratio(returns: pd.Series) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    daily_rf = 0.04 / 252
    excess_returns = returns - daily_rf
    return (excess_returns.mean() / returns.std() )* np.sqrt(252)
