import pandas as pd
import pytest
import strategy_backtester.results.metrics as metrics


@pytest.fixture
def sample_returns():
    return pd.Series([0.01, -0.02, 0.03] * 84)


@pytest.fixture
def expected_cumulative_returns():
    return ((1 + 0.01) * (1 - 0.02) * (1 + 0.03)) ** 84 - 1


def test_calc_cumulative_return(sample_returns, expected_cumulative_returns):
    assert abs(metrics.calc_cumulative_return(sample_returns) - expected_cumulative_returns) < 1e-4
    assert metrics.calc_cumulative_return(pd.Series([])) == 0.0


def test_calc_final_value(sample_returns, expected_cumulative_returns):
    initial_value = 1000
    expected_final_value = initial_value * (1 + expected_cumulative_returns)
    assert abs(metrics.calc_final_value(initial_value, sample_returns) - expected_final_value) < 1e-4


def test_calc_annualised_return(sample_returns, expected_cumulative_returns):
    initial_value = 1000
    expected_annualised_return = (1 + expected_cumulative_returns) ** (252 / len(sample_returns)) - 1
    assert abs(metrics.calc_annualised_return(initial_value, sample_returns) - expected_annualised_return) < 1e-4
    with pytest.raises(ValueError, match="Portfolio value non-positive"):
        metrics.calc_annualised_return(initial_value, pd.Series([-1.0] * 252))


def test_calc_annualised_volatility(sample_returns):
    expected_vol = sample_returns.std() * (252 ** 0.5)
    assert abs(metrics.calc_annualised_volatility(sample_returns) - expected_vol) < 1e-4


def test_calc_sharpe_ratio(sample_returns):
    daily_rf = 0.04 / 252
    excess_returns = sample_returns - daily_rf
    expected_sharpe = (excess_returns.mean() / sample_returns.std()) * (252 ** 0.5)
    assert abs(metrics.calc_sharpe_ratio(sample_returns) - expected_sharpe) < 1e-4
    assert metrics.calc_sharpe_ratio(pd.Series([])) == 0.0
    assert metrics.calc_sharpe_ratio(pd.Series([0.01] * 252)) == 0.0