from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from strategy_backtester.data.dataframe import PRICE_FIELDS, make_price_dataframe
from strategy_backtester.portfolio.portfolio import PortfolioState


# ---------------------------------------------------------------------------
# Date and ticker scaffolding
# ---------------------------------------------------------------------------

N_DAYS = 300        # ~14 months of trading days — covers 12-1 momentum lookback
N_TICKERS = 20      # enough for top/bottom decile construction (2 per decile)
TICKERS = [f"T{i:02d}" for i in range(N_TICKERS)]  # T00, T01, ..., T19
START_DATE = "2022-01-03"


@pytest.fixture(scope="session")
def trading_dates() -> pd.DatetimeIndex:
    """300 business days starting 2022-01-03."""
    return pd.bdate_range(start=START_DATE, periods=N_DAYS)


@pytest.fixture(scope="session")
def tickers() -> list[str]:
    return TICKERS


# ---------------------------------------------------------------------------
# Synthetic OHLCV data
# ---------------------------------------------------------------------------

def _make_ohlcv(
    dates: pd.DatetimeIndex,
    tickers: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a raw (Date, Ticker) MultiIndex DataFrame with realistic OHLCV data.

    Prices follow a GBM-like process so momentum signals are non-trivial.
    Each ticker has a distinct drift so cross-sectional spread exists.
    Volume is synthetic integer data.
    """
    rng = np.random.default_rng(seed)
    n_days = len(dates)
    n_tickers = len(tickers)

    # Distinct drift per ticker: spread from -20% to +40% annualised
    drifts = np.linspace(-0.20, 0.40, n_tickers) / 252
    vol = 0.02  # daily vol ~2%

    log_returns = rng.normal(loc=drifts, scale=vol, size=(n_days, n_tickers))
    log_price = np.cumsum(log_returns, axis=0)
    close = 100.0 * np.exp(log_price)

    daily_range = close * rng.uniform(0.005, 0.02, size=close.shape)
    open_ = close * (1 + rng.uniform(-0.01, 0.01, size=close.shape))
    high = np.maximum(close, open_) + daily_range * 0.5
    low = np.minimum(close, open_) - daily_range * 0.5
    volume = rng.integers(500_000, 5_000_000, size=close.shape)

    rows = []
    for t_idx, ticker in enumerate(tickers):
        for d_idx, date in enumerate(dates):
            rows.append({
                "Date":   date,
                "Ticker": ticker,
                "Open":   round(float(open_[d_idx, t_idx]), 4),
                "High":   round(float(high[d_idx, t_idx]), 4),
                "Low":    round(float(low[d_idx, t_idx]), 4),
                "Close":  round(float(close[d_idx, t_idx]), 4),
                "Volume": int(volume[d_idx, t_idx]),
            })

    return pd.DataFrame(rows).set_index(["Date", "Ticker"]).sort_index()


@pytest.fixture(scope="session")
def raw_ohlcv_df(trading_dates, tickers) -> pd.DataFrame:
    """
    Raw (Date, Ticker) MultiIndex DataFrame. Pre-validation.
    Session-scoped — copy before mutating in individual tests.
    """
    return _make_ohlcv(trading_dates, tickers)


@pytest.fixture(scope="session")
def price_df(raw_ohlcv_df) -> pd.DataFrame:
    """Validated PriceDataFrame built from the session-scoped synthetic data."""
    return make_price_dataframe(raw_ohlcv_df.copy())


@pytest.fixture(scope="session")
def small_price_df(tickers) -> pd.DataFrame:
    """50-day PriceDataFrame for tests that need a valid engine run but not full history."""
    dates = pd.bdate_range(start=START_DATE, periods=50)
    return make_price_dataframe(_make_ohlcv(dates, tickers[:5]))


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_data_dict():
    """Minimal unstructured dict for testing make_price_dataframe validation."""
    return {
        "Date":   ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
        "Ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "Open":   [300.0, 305.0, 150.0, 155.0],
        "High":   [310.0, 315.0, 160.0, 165.0],
        "Low":    [295.0, 300.0, 145.0, 150.0],
        "Close":  [305.0, 310.0, 155.0, 160.0],
        "Volume": [1000000, 1200000, 800000, 900000],
    }


@pytest.fixture
def mock_price_dataframe(mock_data_dict) -> pd.DataFrame:
    """Valid PriceDataFrame with correct dtypes for accessor tests."""
    df = pd.DataFrame(mock_data_dict)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df.set_index(["Date", "Ticker"], inplace=True)
    for c in PRICE_FIELDS:
        df[c] = df[c].astype("int64" if c == "Volume" else "float64")
    return make_price_dataframe(df)


def _make_yf_response(ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Produce a flat Date-indexed DataFrame mirroring yf.download output
    for a single ticker with auto_adjust=True.
    """
    df = _make_ohlcv(dates, [ticker])
    out = df.xs(ticker, level="Ticker").copy()
    out.index.name = "Date"
    return out


@pytest.fixture
def mock_yf_download(trading_dates):
    """
    Monkeypatch yfinance.download to return synthetic data without hitting
    the network. Function-scoped so each test gets a clean call count.
    """
    def _fake_download(ticker, start=None, end=None, auto_adjust=True, progress=False, **kwargs):
        dates = trading_dates
        if start is not None:
            dates = dates[dates >= pd.Timestamp(start)]
        if end is not None:
            dates = dates[dates < pd.Timestamp(end)]
        if dates.empty:
            return pd.DataFrame()
        return _make_yf_response(ticker, dates)

    with patch("strategy_backtester.data.loader.yf.download", side_effect=_fake_download) as mock:
        yield mock


@pytest.fixture
def cache_path(tmp_path) -> str:
    """Per-test temporary cache file path. Never touches the real cache."""
    return str(tmp_path / "test_cache.csv")


# ---------------------------------------------------------------------------
# Portfolio fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def start_portfolio_state():
    """PortfolioState at t=0 with no positions."""
    return PortfolioState(
        date=pd.Timestamp("2020-01-01"),
        starting_capital=100_000,
    )