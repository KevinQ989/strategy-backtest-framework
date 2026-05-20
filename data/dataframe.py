from __future__ import annotations
from typing import NewType
import pandas as pd


PriceDataFrame = NewType("PriceDataFrame", pd.DataFrame)
PRICE_FIELDS = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]


def make_price_dataframe(df: pd.DataFrame) -> PriceDataFrame:
    """
    Validate and return a PriceDataFrame

    Expected Structure
    ------------------
    Index: pd.MultiIndex with levels (Date, Ticker)
        Date - pd.Timestamp, timezone-naive
        Ticker - str
    Columns: Open, High, Low, Close, Adj_Close, Volume
    Values: float64 for price fields, int64 for volume

    Raises
    ------
    TypeError  : Wrong index type, wrong column types, timezone present
    ValueError : Missing required columns, duplicate keys, NaN values
    """
    # Check MultiIndex structure
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError(
            f"Expected MultiIndex with levels (Date, Ticker), got {type(df.index)}",
            f"Use df.set_index(['Date', 'Ticker'], inplace=True) to set the index.",
        )
    if len(df.index.levels) != 2:
        raise TypeError(
            f"Expected MultiIndex with 2 levels (Date, Ticker), got {len(df.index.levels)} levels",
        )
    
    # Check Date level index
    date_level = df.index.get_level_values(0)
    if not isinstance(date_level, pd.DatetimeIndex):
        raise TypeError(
            f"Expected first index level to be Date (pd.Timestamp), got {type(date_level)}",
            f"Use pd.to_datetime() to convert to datetime.",
        )
    if date_level.tz is not None:
        raise TypeError(
            "Expected timezone-naive Date index, but timezone information was found.",
            "Use df.index = df.index.set_levels(df.index.levels[0].tz_localize(None), level=0) to remove timezone information.",
        )
    
    # Check Ticker level index
    ticker_level = df.index.get_level_values(1)
    if not all(isinstance(t, str) for t in ticker_level.unique()):
        raise TypeError(
            f"Expected second index level to be Ticker (str), but found non-string values.",
            f"Use df.index = df.index.set_levels(df.index.levels[1].astype(str), level=1) to convert to string.",
        )
    
    # Check required columns
    missing_cols = set(PRICE_FIELDS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}",
            f"Ensure the DataFrame contains the following columns: {PRICE_FIELDS}",
        )
    
    # Check for duplicate keys
    if df.index.duplicated().any():
        raise ValueError(
            "Duplicate keys found in MultiIndex. Each (Date, Ticker) pair must be unique.",
            "Use df.index.duplicated() to identify duplicates and resolve them.",
        )
    
    # Check for NaN values
    if df.isna().any().any():
        raise ValueError(
            "NaN values found in DataFrame. All price and volume fields must be complete.",
            "Use df.isna().sum() to identify columns with NaN values and handle them appropriately.",
        )
    
    # Enfore correct dtypes
    for col in PRICE_FIELDS:
        if col == "Volume":
            expected_dtype = "int64"
        else:
            expected_dtype = "float64"
        
        if df[col].dtype != expected_dtype:
            raise TypeError(
                f"Column '{col}' has incorrect dtype. Expected {expected_dtype}, got {df[col].dtype}.",
                f"Use df['{col}'] = df['{col}'].astype({expected_dtype}) to convert the column to the correct dtype.",
            )
    
    return PriceDataFrame(df)


def get_field(price_df: PriceDataFrame, field: str) -> pd.DataFrame:
    """
    Extract a specific price field as a DataFrame with Date index and Ticker columns.

    Parameters
    ----------
    price_df : PriceDataFrame
        The validated price DataFrame containing all fields.
    field : str
        The price field to extract (e.g., "Close", "Volume").

    Returns
    -------
    pd.DataFrame
        Index: Date (pd.Timestamp)
        Columns: Ticker (str)
        Values: The specified price field values.

    Raises
    ------
    ValueError : If the specified field is not in the DataFrame columns.
    """
    if field not in price_df.columns:
        raise ValueError(
            f"Field '{field}' not found in PriceDataFrame columns.",
            f"Available fields are: {price_df.columns.tolist()}",
        )
    
    return price_df[field].unstack("Ticker")


def get_date(price_df: PriceDataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """
    Extract all price data for a specific date.

    Parameters
    ----------
    price_df : PriceDataFrame
        The validated price DataFrame containing all fields.
    date : pd.Timestamp
        The date for which to extract price data.

    Returns
    -------
    pd.DataFrame
        Index: Ticker (str)
        Columns: Open, High, Low, Close, Adj_Close, Volume for the specified date.

    Raises
    ------
    ValueError : If the specified date is not found in the DataFrame index.
    """
    if date not in price_df.index.get_level_values(0):
        raise ValueError(
            f"Date '{date}' not found in PriceDataFrame index.",
            f"Available dates are: {price_df.index.get_level_values(0).unique().tolist()}",
        )
    
    return price_df.xs(date, level="Date")


def get_ticker(price_df: PriceDataFrame, ticker: str) -> pd.DataFrame:
    """
    Extract all price data for a specific ticker.

    Parameters
    ----------
    price_df : PriceDataFrame
        The validated price DataFrame containing all fields.
    ticker : str
        The ticker symbol for which to extract price data.

    Returns
    -------
    pd.DataFrame
        Index: Date (pd.Timestamp)
        Columns: Open, High, Low, Close, Adj_Close, Volume for the specified ticker.

    Raises
    ------
    ValueError : If the specified ticker is not found in the DataFrame index.
    """
    if ticker not in price_df.index.get_level_values(1):
        raise ValueError(
            f"Ticker '{ticker}' not found in PriceDataFrame index.",
            f"Available tickers are: {price_df.index.get_level_values(1).unique().tolist()}",
        )
    
    return price_df.xs(ticker, level="Ticker")