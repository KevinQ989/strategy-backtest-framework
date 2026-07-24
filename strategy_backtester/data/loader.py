from __future__ import annotations
import logging
import yfinance as yf
import pandas as pd
import os
from .dataframe import PriceDataFrame, PRICE_FIELDS, make_price_dataframe

DATA_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(DATA_DIRECTORY, "market_data_cache.csv")


def load_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_path: str = CACHE_FILE
) -> PriceDataFrame:
    """
    Fetch adjusted OHLCV data for a list of tickers and return a validated
    PriceDataFrame with (Date, Ticker) index.

    Data is cached in csv format. Subsequent calls only downloads missing data.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols to fetch e.g. ["AAPL", "INTC"].
    start_date : str
        Start date in "YYYY-MM-DD" format, inclusive.
    end_date : str
        End date in "YYYY-MM-DD" format, inclusive.
    cache_path : str, optional
        Path to cache file. Default is "market_data_cache.csv" in the same directory as this module.
    
    Returns
    -------
    PriceDataFrame
        Validated PriceDataFrame with MultiIndex (Date, Ticker).
        Columns: Open, High, Low, Close, Volume.
    """
    # Convert dates to pandas timestamps for date comparison
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    # Load existing cache if it exists
    cache_df = _load_cache(cache_path)
    
    # Check for missing data
    to_fetch = _find_missing_data(tickers, start_ts, end_ts, cache_df)

    # Fetch missing data if needed
    if to_fetch:
        new_frames = []
        failed_tickers = []
        for ticker, ranges in to_fetch.items():
            for range_start, range_end in ranges:
                data = _download(ticker, range_start, range_end)
                if not data.empty:
                    new_frames.append(data)
                else:
                    failed_tickers.append(ticker)
        if failed_tickers:
            unique_failed = list(dict.fromkeys(failed_tickers))
            print(
                f"Warning: Failed to download data for {len(unique_failed)} tickers: "
                f"{unique_failed}"
            )
        if new_frames:
            cache_df = _merge_cache(cache_df, pd.concat(new_frames))
            _save_cache(cache_df, cache_path)
    
    # Validate and return PriceDataFrame
    return _to_price_dataframe(cache_df, tickers, start_ts, end_ts)


def _load_cache(cache_path: str) -> pd.DataFrame:
    """Load cache from csv if it exists."""
    if not os.path.exists(cache_path):
        return pd.DataFrame(
    columns=PRICE_FIELDS,
    index=pd.MultiIndex.from_arrays([[], []], names=['Date', 'Ticker'])
)
    df = pd.read_csv(
        cache_path,
        index_col=['Date', 'Ticker'],
        parse_dates=['Date'],
    )
    date_level = df.index.get_level_values('Date')
    if date_level.tz is not None:
        df.index = df.index.set_levels(
            df.index.levels[0].tz_localize(None), level='Date'
        )
    return df


def _merge_cache(cache_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new data with existing cache, ensuring no duplicates."""
    if cache_df.empty:
        return new_df.sort_index()
    combined = pd.concat([cache_df, new_df])
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined.sort_index()


def _save_cache(df: pd.DataFrame, cache_path: str) -> None:
    """Save DataFrame to csv cache."""
    df.to_csv(cache_path)
    print(f"Saved cache to {cache_path}")


def _find_missing_data(
    tickers: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    cache_df: pd.DataFrame
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Determine which tickers have missing data in the cache."""
    if cache_df.empty:
        return {ticker: [(start_ts, end_ts)] for ticker in tickers}
    
    all_cached_dates = cache_df.index.get_level_values('Date').unique()
    trading_days = pd.DatetimeIndex(
        sorted(d for d in all_cached_dates if start_ts <= d <= end_ts)
    )

    # Missing all trading days, need to fetch all dates for all tickers
    if trading_days.empty:
        return {ticker: [(start_ts, end_ts)] for ticker in tickers}

    cached_tickers = cache_df.index.get_level_values('Ticker').unique()
    to_fetch = {}
    for ticker in tickers:
        # Missing ticker, need to fetch all dates for this ticker
        if ticker not in cached_tickers:
            to_fetch[ticker] = [(start_ts, end_ts)]
            continue
        ticker_dates = pd.DatetimeIndex(cache_df.xs(ticker, level='Ticker').index)
        gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        # Check for missing data before and after the cached date range
        if ticker_dates.min() > start_ts + pd.offsets.BDay(1):
            gaps.append((start_ts, ticker_dates.min() - pd.offsets.BDay(1)))
        if ticker_dates.max() < end_ts - pd.offsets.BDay(1):
            gaps.append((ticker_dates.max() + pd.offsets.BDay(1), end_ts))

        # Check for internal gaps in the cached data
        missing = trading_days.difference(ticker_dates)
        if not missing.empty:
            dates = sorted(missing)
            ranges = []
            range_start = dates[0]
            range_end = dates[0]
            for date in dates[1:]:
                if (date - range_end).days <= 4:
                    range_end = date
                else:
                    ranges.append((range_start, range_end))
                    range_start = date
                    range_end = date
            ranges.append((range_start, range_end))
            gaps.extend(ranges)
        if gaps:
            to_fetch[ticker] = gaps
    return to_fetch


def _download(
    ticker: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp
) -> pd.DataFrame:
    """Download data for the specified ticker and date range."""
    print(f"Downloading {ticker} from {start_ts.strftime('%Y-%m-%d')} to {end_ts.strftime('%Y-%m-%d')}")
    yf_log = logging.getLogger("yfinance")
    yf_log.setLevel(logging.CRITICAL)
    raw = yf.download(
        ticker,
        start=start_ts,
        end=end_ts + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False
    )
    yf_log.setLevel(logging.WARNING)
    if raw.empty:
        return pd.DataFrame()
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw.index.name = "Date"
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns={"Adj Close": "Adj_Close"})
    raw = raw[[f for f in PRICE_FIELDS if f in raw.columns]].copy()
    raw["Volume"] = raw["Volume"].astype('int64')
    raw["Ticker"] = ticker
    return raw.reset_index().set_index(["Date", "Ticker"]).sort_index()


def _to_price_dataframe(
    cache_df: pd.DataFrame,
    tickers: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp
) -> PriceDataFrame:
    """Convert raw cache DataFrame to validated PriceDataFrame with MultiIndex."""
    # Select only the requested tickers and date range from the cache
    dates = cache_df.index.get_level_values('Date')
    tickers_idx = cache_df.index.get_level_values('Ticker')
    mask = (
        (dates >= start_ts) &
        (dates <= end_ts) &
        (tickers_idx.isin(tickers))
    )
    df = cache_df.loc[mask].copy()

    # Forward fill missing data
    all_dates = df.index.get_level_values('Date').unique().sort_values()
    full_index = pd.MultiIndex.from_product([all_dates, tickers], names=['Date', 'Ticker'])
    df = df.reindex(full_index)
    df = df.groupby(level="Ticker", group_keys=False).ffill()
    df["Volume"] = df["Volume"].fillna(0).astype('int64') #OR use Int64 instead of int64

    # Drop tickers that still have NaNs after ffill
    close_wide = df["Close"].unstack(level="Ticker")
    tickers_with_nans = close_wide.columns[close_wide.isna().any()].tolist()
    if tickers_with_nans:
        print(
            f"Warning: {len(tickers_with_nans)} tickers dropped due to insufficient "
            f"history for start_date {start_ts.date()}: {tickers_with_nans}"
        )
        df = df.loc[~df.index.get_level_values("Ticker").isin(tickers_with_nans)]

    df["Volume"] = df["Volume"].astype('int64')

    # Drop dates where every ticker's Close is missing
    close_by_date = df["Close"].unstack(level="Ticker")
    valid_dates = close_by_date.dropna(how='all').index
    df = df.loc[df.index.get_level_values('Date').isin(valid_dates)]

    return make_price_dataframe(df)
