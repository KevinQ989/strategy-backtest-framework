import pytest
import pandas as pd
from strategy_backtester.portfolio.portfolio import PortfolioState


@pytest.fixture
def start_portfolio_state():
    return PortfolioState(
        date=pd.Timestamp("2020-01-01"),
        starting_capital=100000
    )