import pytest
import pandas as pd
from strategy_backtester.data import PRICE_FIELDS, make_price_dataframe
from strategy_backtester.portfolio.portfolio import PortfolioState


@pytest.fixture
def mock_data_dict():
    """Unstructured mock data dictionary for testing."""
    return {
        "Date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
        "Ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "Open": [300, 305, 150, 155],
        "High": [310, 315, 160, 165],
        "Low": [295, 300, 145, 150],
        "Close": [305, 310, 155, 160],
        "Volume": [1000000, 1200000, 800000, 900000]
    }


@pytest.fixture
def mock_price_dataframe(mock_data_dict):
    """Valid PriceDataFrame fixture for testing."""
    df = pd.DataFrame(mock_data_dict)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df.set_index(['Date', 'Ticker'], inplace=True)
    for c in PRICE_FIELDS:
        df[c] = df[c].astype('int64' if c == 'Volume' else 'float64')
    pdf = make_price_dataframe(df)
    return pdf


@pytest.fixture
def start_portfolio_state():
    return PortfolioState(
        date=pd.Timestamp("2020-01-01"),
        starting_capital=100000
    )