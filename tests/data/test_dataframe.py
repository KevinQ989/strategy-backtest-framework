import pytest
import re
import pandas as pd
from strategy_backtester.data import (
    PRICE_FIELDS,
    make_price_dataframe,
    get_field,
    get_date,
    get_ticker
)


def test_make_pdf_is_multiindex(mock_data_dict):
    df = pd.DataFrame(mock_data_dict)
    expected_message = (
        f"Expected MultiIndex with levels (Date, Ticker), got {type(df.index)}."
        f"Use df.set_index(['Date', 'Ticker'], inplace=True) to set the index."
    )
    with pytest.raises(TypeError, match=re.escape(expected_message)):
        make_price_dataframe(df)


def test_make_pdf_has_two_levels(mock_data_dict):
    df = pd.DataFrame(mock_data_dict)
    df.set_index(['Date', 'Ticker', 'Open'], inplace=True)
    expected_message = f"Expected MultiIndex with 2 levels (Date, Ticker), got {df.index.nlevels} levels."
    with pytest.raises(TypeError, match=re.escape(expected_message)):
        make_price_dataframe(df)


def test_make_pdf_date_level_is_datetime(mock_data_dict):
    df = pd.DataFrame(mock_data_dict)
    df.set_index(['Date', 'Ticker'], inplace=True)
    expected_message = (
        f"Expected first index level to be Date (pd.Timestamp), got {type(df.index.get_level_values(0))}."
        f"Use pd.to_datetime() to convert to datetime."
    )
    with pytest.raises(TypeError, match=re.escape(expected_message)):
        make_price_dataframe(df)


def test_make_pdf_date_level_is_timezone_naive(mock_data_dict):
    df = pd.DataFrame(mock_data_dict)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('UTC')
    df.set_index(['Date', 'Ticker'], inplace=True)
    expected_message = (
        "Expected timezone-naive Date index, but timezone information was found."
        "Use df.index = df.index.set_levels(df.index.levels[0].tz_localize(None), level=0) to remove timezone information."
    )
    with pytest.raises(TypeError, match=re.escape(expected_message)):
        make_price_dataframe(df)


def test_make_pdf_ticker_level_is_string(mock_data_dict):
    df = pd.DataFrame(mock_data_dict)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df['Ticker'] = 12345
    df.set_index(['Date', 'Ticker'], inplace=True)
    expected_message = (
        f"Expected second index level to be Ticker (str), but found non-string values."
        f"Use df.index = df.index.set_levels(df.index.levels[1].astype(str), level=1) to convert to string."
    )
    with pytest.raises(TypeError, match=re.escape(expected_message)):
        make_price_dataframe(df)


def test_make_pdf_missing_columns(mock_data_dict):
    df = pd.DataFrame(mock_data_dict)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df.set_index(['Date', 'Ticker'], inplace=True)
    df.drop(columns=['Open'], inplace=True)
    expected_message = (
        f"Missing required columns: {{'Open'}}"
        f"Ensure the DataFrame contains the following columns: {PRICE_FIELDS}"
    )
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        make_price_dataframe(df)


def test_make_pdf_duplicate_keys(mock_data_dict):
    df = pd.DataFrame(mock_data_dict)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df.set_index(['Date', 'Ticker'], inplace=True)
    duplicate_row = df.iloc[0]
    df = pd.concat([df, duplicate_row.to_frame().T])
    expected_message = (
        "Duplicate keys found in MultiIndex. Each (Date, Ticker) pair must be unique."
        "Use df.index.duplicated() to identify duplicates and resolve them."
    )
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        make_price_dataframe(df)


def test_make_pdf_nan_values(mock_data_dict):
    df = pd.DataFrame(mock_data_dict)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df.set_index(['Date', 'Ticker'], inplace=True)
    df.loc[df.index[0], 'Open'] = float('nan')
    expected_message = (
        "NaN values found in DataFrame. All price and volume fields must be complete."
        "Use df.isna().sum() to identify columns with NaN values and handle them appropriately."
    )
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        make_price_dataframe(df)


def test_make_pdf_valid_dtypes(mock_data_dict):
    for col in PRICE_FIELDS:
        df = pd.DataFrame(mock_data_dict)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df.set_index(['Date', 'Ticker'], inplace=True)
        for c in PRICE_FIELDS:
            df[c] = df[c].astype('int64' if c == 'Volume' else 'float64')
        if col == 'Volume':
            df[col] = df[col].astype('float64')
            expected_dtype = 'int64'
        else:
            df[col] = df[col].astype('int64')
            expected_dtype = 'float64'
        expected_message = (
            f"Column '{col}' has incorrect dtype. Expected {expected_dtype}, got {df[col].dtype}."
            f"Use df['{col}'] = df['{col}'].astype({expected_dtype}) to convert the column to the correct dtype."
        )
        with pytest.raises(TypeError, match=re.escape(expected_message)):
            make_price_dataframe(df)


def test_make_pdf_valid_dataframe(mock_price_dataframe):
    assert isinstance(mock_price_dataframe, pd.DataFrame)
    assert mock_price_dataframe.index.names == ['Date', 'Ticker']
    assert all(col in mock_price_dataframe.columns for col in PRICE_FIELDS)
    assert mock_price_dataframe.dtypes['Volume'] == 'int64'
    for col in ['Open', 'High', 'Low', 'Close']:
        assert mock_price_dataframe.dtypes[col] == 'float64'


def test_get_field_valid(mock_price_dataframe):
    close_df = get_field(mock_price_dataframe, 'Close')
    assert isinstance(close_df, pd.DataFrame)
    assert close_df.index.names == ['Date']
    assert close_df.columns.tolist() == ['AAPL', 'MSFT']
    assert close_df.dtypes['AAPL'] == 'float64'
    assert close_df.dtypes['MSFT'] == 'float64'


def test_get_field_invalid(mock_price_dataframe):
    expected_message = (
        f"Field 'Adj_Close' not found in PriceDataFrame columns."
        f"Available fields are: {mock_price_dataframe.columns.tolist()}"
    )
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        get_field(mock_price_dataframe, 'Adj_Close')


def test_get_date_valid(mock_price_dataframe):
    date = pd.Timestamp("2020-01-01")
    date_df = get_date(mock_price_dataframe, date)
    assert isinstance(date_df, pd.DataFrame)
    assert date_df.index.names == ['Ticker']
    assert date_df.columns.tolist() == ['Open', 'High', 'Low', 'Close', 'Volume']
    assert date_df.dtypes['Open'] == 'float64'
    assert date_df.dtypes['High'] == 'float64'
    assert date_df.dtypes['Low'] == 'float64'
    assert date_df.dtypes['Close'] == 'float64'
    assert date_df.dtypes['Volume'] == 'int64'


def test_get_date_invalid(mock_price_dataframe):
    date = pd.Timestamp("2020-01-03")
    expected_message = (
        f"Date '{date}' not found in PriceDataFrame index."
        f"Available dates are: {mock_price_dataframe.index.get_level_values(0).unique().tolist()}"
    )
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        get_date(mock_price_dataframe, date)


def test_get_ticker_valid(mock_price_dataframe):
    ticker = "AAPL"
    ticker_df = get_ticker(mock_price_dataframe, ticker)
    assert isinstance(ticker_df, pd.DataFrame)
    assert ticker_df.index.names == ['Date']
    assert ticker_df.columns.tolist() == ['Open', 'High', 'Low', 'Close', 'Volume']
    assert ticker_df.dtypes['Open'] == 'float64'
    assert ticker_df.dtypes['High'] == 'float64'
    assert ticker_df.dtypes['Low'] == 'float64'
    assert ticker_df.dtypes['Close'] == 'float64'
    assert ticker_df.dtypes['Volume'] == 'int64'


def test_get_ticker_invalid(mock_price_dataframe):
    ticker = "GOOG"
    expected_message = (
        f"Ticker '{ticker}' not found in PriceDataFrame index."
        f"Available tickers are: {mock_price_dataframe.index.get_level_values(1).unique().tolist()}"
    )
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        get_ticker(mock_price_dataframe, ticker)