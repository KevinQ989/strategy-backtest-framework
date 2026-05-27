from __future__ import annotations
import os
import pandas as pd
from unittest.mock import patch
from strategy_backtester.data.dataframe import PRICE_FIELDS
from strategy_backtester.data.loader import (
    _load_cache,
    _merge_cache,
    _save_cache,
    _find_missing_data,
    _download,
    _to_price_dataframe,
    load_data
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yf_response(raw_ohlcv_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Produce a DataFrame which mirrors yf.download() output for a single ticker."""
    df = raw_ohlcv_df.xs(ticker, level="Ticker").copy()
    df.index.name = "Date"
    return df


def _drop_dates(df: pd.DataFrame, ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Remove specific (date, ticker) rows from a cache DataFrame."""
    idx_to_drop = pd.MultiIndex.from_product([dates, [ticker]], names=["Date", "Ticker"])
    return df.drop(idx_to_drop, errors='ignore')


def _fmt(ts: pd.Timestamp) -> str:
    """Helper to format timestamps in expected error messages."""
    return ts.strftime('%Y-%m-%d')


# ---------------------------------------------------------------------------
# Test _load_cache
# ---------------------------------------------------------------------------

def test_load_cache_no_file(tmp_path):
    path = str(tmp_path / "nonexistent_cache.csv")
    df = _load_cache(path)
    assert df.empty


def test_load_cache_with_no_timezone(raw_ohlcv_df, cache_path):
    _save_cache(raw_ohlcv_df, cache_path)
    df = _load_cache(cache_path)
    assert df.index.get_level_values('Date').tz is None


def test_load_cache_with_timezone(raw_ohlcv_df, cache_path):
    df = raw_ohlcv_df.copy()
    tz_dates = df.index.get_level_values('Date').tz_localize('UTC')
    tickers_ = df.index.get_level_values('Ticker')
    df.index = pd.MultiIndex.from_arrays([tz_dates, tickers_], names=['Date', 'Ticker'])
    df.to_csv(cache_path)
    df = _load_cache(cache_path)
    assert df.index.get_level_values('Date').tz is None


def test_load_cache_has_multiindex(raw_ohlcv_df, cache_path):
    _save_cache(raw_ohlcv_df, cache_path)
    result = _load_cache(cache_path)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["Date", "Ticker"]


# ---------------------------------------------------------------------------
# Test _merge_cache
# ---------------------------------------------------------------------------

def test_merge_cache_no_existing(raw_ohlcv_df):
    result = _merge_cache(pd.DataFrame(), raw_ohlcv_df)
    assert len(result) == len(raw_ohlcv_df)
 
 
def test_merge_cache_with_itself_drops_duplicates(raw_ohlcv_df):
    result = _merge_cache(raw_ohlcv_df, raw_ohlcv_df)
    assert len(result) == len(raw_ohlcv_df)
    assert not result.index.duplicated().any()
 
 
def test_merge_cache_disjoint_sets(raw_ohlcv_df, trading_dates):
    first = raw_ohlcv_df.loc[
        raw_ohlcv_df.index.get_level_values("Date").isin(trading_dates[:100])
    ]
    second = raw_ohlcv_df.loc[
        raw_ohlcv_df.index.get_level_values("Date").isin(trading_dates[100:])
    ]
    result = _merge_cache(first, second)
    assert len(result) == len(raw_ohlcv_df)
 
 
def test_merge_cache_result_is_sorted(raw_ohlcv_df):
    shuffled = raw_ohlcv_df.sample(frac=1, random_state=0)
    result = _merge_cache(pd.DataFrame(), shuffled)
    assert result.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Test _save_cache
# ---------------------------------------------------------------------------

def test_save_cache_creates_file(raw_ohlcv_df, cache_path):
    assert not os.path.exists(cache_path)
    _save_cache(raw_ohlcv_df, cache_path)
    assert os.path.exists(cache_path)
 
 
def test_save_cache_roundtrip_row_count(raw_ohlcv_df, cache_path):
    _save_cache(raw_ohlcv_df, cache_path)
    loaded = _load_cache(cache_path)
    assert len(loaded) == len(raw_ohlcv_df)
 
 
def test_save_cache_roundtrip_tickers(raw_ohlcv_df, cache_path, tickers):
    _save_cache(raw_ohlcv_df, cache_path)
    loaded = _load_cache(cache_path)
    assert set(loaded.index.get_level_values("Ticker").unique()) == set(tickers)
 

# ---------------------------------------------------------------------------
# Test _find_missing_data
# ---------------------------------------------------------------------------


def test_find_missing_data_no_cache(tickers, trading_dates):
    result = _find_missing_data(tickers, trading_dates[0], trading_dates[-1], pd.DataFrame())
    assert set(result.keys()) == set(tickers)
 
 
def test_find_missing_data_full_cache(raw_ohlcv_df, tickers, trading_dates):
    result = _find_missing_data(tickers, trading_dates[0], trading_dates[-1], raw_ohlcv_df)
    assert result == {}
 
 
def test_find_missing_data_missing_ticker(raw_ohlcv_df, tickers, trading_dates):
    cache = raw_ohlcv_df.loc[
        raw_ohlcv_df.index.get_level_values("Ticker") != tickers[0]
    ]
    result = _find_missing_data(tickers, trading_dates[0], trading_dates[-1], cache)
    assert tickers[0] in result

 
def test_find_missing_data_leading_gap(raw_ohlcv_df, tickers, trading_dates):
    ticker = tickers[0]
    cache = _drop_dates(raw_ohlcv_df, ticker, trading_dates[:10])
    result = _find_missing_data([ticker], trading_dates[0], trading_dates[-1], cache)
    assert ticker in result
    gap_start, _ = result[ticker][0]
    assert gap_start <= trading_dates[0]
 
 
def test_find_missing_data_trailing_gap(raw_ohlcv_df, tickers, trading_dates):
    ticker = tickers[0]
    cache = _drop_dates(raw_ohlcv_df, ticker, trading_dates[-10:])
    result = _find_missing_data([ticker], trading_dates[0], trading_dates[-1], cache)
    assert ticker in result
    _, gap_end = result[ticker][-1]
    assert gap_end >= trading_dates[-10]
 
 
def test_find_missing_data_internal_gap(raw_ohlcv_df, tickers, trading_dates):
    ticker = tickers[0]
    cache = _drop_dates(raw_ohlcv_df, ticker, trading_dates[100:103])
    result = _find_missing_data([ticker], trading_dates[0], trading_dates[-1], cache)
    assert ticker in result


# ---------------------------------------------------------------------------
# Test _download
# ---------------------------------------------------------------------------

def test_download_empty_response():
    with patch("strategy_backtester.data.loader.yf.download", return_value=pd.DataFrame()):
        result = _download("FAKE", pd.Timestamp("2022-01-03"), pd.Timestamp("2022-01-07"))
    assert result.empty


def test_download_valid(raw_ohlcv_df, tickers, trading_dates):
    yf_response = _make_yf_response(raw_ohlcv_df, tickers[0])
    with patch("strategy_backtester.data.loader.yf.download", return_value=yf_response):
        result = _download(tickers[0], trading_dates[0], trading_dates[-1])
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["Date", "Ticker"]
    assert set(PRICE_FIELDS).issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Test _to_price_dataframe
# ---------------------------------------------------------------------------


def test_to_pdf_ticker_filter(raw_ohlcv_df, tickers, trading_dates):
    subset = tickers[:5]
    result = _to_price_dataframe(raw_ohlcv_df, subset, trading_dates[0], trading_dates[-1])
    returned = result.index.get_level_values("Ticker").unique()
    assert set(returned) == set(subset)
 
 
def test_to_pdf_date_filter(raw_ohlcv_df, tickers, trading_dates):
    start = trading_dates[50]
    end = trading_dates[100]
    result = _to_price_dataframe(raw_ohlcv_df, tickers, start, end)
    dates = result.index.get_level_values("Date")
    assert dates.min() >= start
    assert dates.max() <= end
 
 
def test_to_pdf_forward_fill_no_nans(raw_ohlcv_df, tickers, trading_dates):
    """Drop one date for one ticker — ffill should cover it, no NaNs in result."""
    ticker = tickers[0]
    cache = _drop_dates(raw_ohlcv_df, ticker, trading_dates[50:51])
    result = _to_price_dataframe(cache, [ticker], trading_dates[0], trading_dates[-1])
    assert not result.isna().any().any()
 
 
def test_to_pdf_drops_all_nan_dates(raw_ohlcv_df, tickers, trading_dates):
    """A date missing across all tickers should be dropped from the result."""
    drop_date = trading_dates[50]
    cache = raw_ohlcv_df.copy()
    for ticker in tickers:
        cache = _drop_dates(cache, ticker, [drop_date])
    result = _to_price_dataframe(cache, tickers, trading_dates[0], trading_dates[-1])
    assert drop_date not in result.index.get_level_values("Date")


# ---------------------------------------------------------------------------
# Test load_data
# ---------------------------------------------------------------------------

def test_load_data_cold_cache_downloads_all(mock_yf_download, cache_path, tickers, trading_dates):
    subset = tickers[:3]
    load_data(subset, _fmt(trading_dates[0]), _fmt(trading_dates[-1]), cache_path)
    assert mock_yf_download.call_count == len(subset)
 
 
def test_load_data_warm_cache_skips_download(mock_yf_download, cache_path, tickers, trading_dates):
    subset = tickers[:3]
    start, end = _fmt(trading_dates[0]), _fmt(trading_dates[-1])
    load_data(subset, start, end, cache_path)
    count = mock_yf_download.call_count
    load_data(subset, start, end, cache_path)
    assert mock_yf_download.call_count == count
 
 
def test_load_data_partial_cache_fetches_new_ticker_only(
    mock_yf_download, cache_path, tickers, trading_dates
):
    start, end = _fmt(trading_dates[0]), _fmt(trading_dates[-1])
    load_data(tickers[:2], start, end, cache_path)
    count = mock_yf_download.call_count
    load_data(tickers[:3], start, end, cache_path)
    assert mock_yf_download.call_count == count + 1