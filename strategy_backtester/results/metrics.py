import numpy as np
import pandas as pd

def calc_holding_period_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return (1 + returns).prod() - 1


def calc_final_value(initial_value: float, returns: pd.Series) -> float:
    return initial_value * (1 + calc_holding_period_return(returns))


def calc_effective_annual_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    years = len(returns) / 252
    return (1 + calc_holding_period_return(returns)) ** (1 / years) - 1


def calc_annualised_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(252)


# Risk free rate of 4%
def calc_sharpe_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    if returns.std() == 0:
        raise ValueError(
            "Standard deviation of returns is zero. Sharpe ratio is undefined."
        )
    daily_rf = 0.04 / 252
    excess_returns = returns - daily_rf
    return (excess_returns.mean() / returns.std() ) * np.sqrt(252)


def calc_sortino_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    daily_rf = 0.04 / 252
    excess_returns = returns - daily_rf
    downside_returns = np.minimum(excess_returns.values, 0.0)
    downside_variance = (downside_returns ** 2).mean()
    if downside_variance == 0:
        return np.inf
    return (excess_returns.mean() / np.sqrt(downside_variance)) * np.sqrt(252)


def calc_calmar_ratio(returns: pd.Series) -> float:
    max_dd = calc_max_drawdown(returns)
    if max_dd == 0:
        return np.inf
    return calc_effective_annual_rate(returns) / abs(max_dd)


def calc_max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def calc_max_drawdown_duration(returns: pd.Series) -> int:
    if returns.empty:
        return 0
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak

    drawdown_duration = 0
    max_duration = 0
    for d in drawdown:
        if d < 0:
            drawdown_duration += 1
            max_duration = max(max_duration, drawdown_duration)
        else:
            drawdown_duration = 0

    return max_duration


def calc_win_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    wins = (returns > 0).sum()
    total = len(returns)
    return wins / total


def calc_rolling_drawdown(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    return (cumulative_returns - peak) / peak


def calc_cumulative_returns_series(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    return (1 + returns).cumprod() - 1


def calc_precost_returns(returns: pd.Series, costs: pd.Series, starting_capital: float) -> pd.Series:
    if returns.empty:
        return returns
    return returns + (costs / starting_capital)