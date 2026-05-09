#Load data from yfinance
#draws info from main.py to decide which companies to import. 
import yfinance as yf
import datetime
import pandas as pd
import os

DATA_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

CACHE_FILE = os.path.join(DATA_DIRECTORY, "market_data_cache.csv")

def get_data(tickers, start_date, end_date, cache_path = CACHE_FILE): #Input list of tickers, returns dataframe of all prices
    if os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path, index_col=0, parse_dates=True) 

        #Remove timezone to allow date comparisons
        if cache_df.index.tz is not None:
            cache_df.index = cache_df.index.tz_localise(None)
    else:
        cache_df = pd.DataFrame()

    #Convert dates to pandas timestamps for date comparison
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    #List to hold new downloaded dataframes before merging them with the cache dataframe
    new_data_frames = [] 

    #Check cache for each ticker to see if data already exists
    for ticker in tickers:
        col_name = f"{ticker}_Close"
        needs_download = False

        #Set default to entire date range
        fetch_start = start_ts
        fetch_end = end_ts

        #If we have nothing, download everything
        if cache_df.empty or col_name not in cache_df.columns:
            needs_download = True
        #Else if we have some data, find the range of dates that we have in the cache
        else:
            valid_dates = cache_df[col_name].dropna().index

            #Give a 1 day buffer for weekends
            start_tolerance = start_ts + pd.offsets.BDay(1)
            end_tolerance = end_ts - pd.offsets.BDay(1)

            if valid_dates.empty:
                needs_download = True
            
            #Download data if start_ts to end_ts contains dates that aren't currently in cache_df
            elif valid_dates.min() > start_tolerance or valid_dates.max() < end_tolerance:
                needs_download = True

                #If above is true, calculate the boundaries
                if start_ts >= valid_dates.min() and end_ts > valid_dates.max():
                    fetch_start = valid_dates.max()
                elif start_ts < valid_dates.min() and end_ts <= valid_dates.max():
                    fetch_end = valid_dates.min()
            else:
                #We are safely inside the boundaries of cache_df. Check if there are any gaps in the middle of the data longer than 10 days
                #Filter out slice we want to work with, check if there are holes inside
                requested_slice = valid_dates[(valid_dates >= start_ts) & (valid_dates <= end_ts)]

                if not requested_slice.empty:
                    #Calculate the time difference between consecutive rows
                    date_gaps = requested_slice.to_series().diff()

                    #If the largest gap is more than 10 days, patch the hole
                    if date_gaps.max() > pd.Timedelta(days = 10):
                        print(f"Internal data gap detected for {ticker}. Patching hole.")
                        needs_download = True
                        fetch_start = start_ts
                        fetch_end = end_ts
        
        #Download from yfinance only if necessary
        if needs_download:
            print(f"Downloading {ticker} from {fetch_start.strftime('%Y-%m-%d')} to {fetch_end.strftime('%Y-%m-%d')}")

            ticker_data = yf.Ticker(ticker).history(start = fetch_start, end = fetch_end + pd.Timedelta(days = 1), repair = True)

            if not ticker_data.empty:
                ticker_data.index = ticker_data.index.tz_localize(None)

                #Filter out important data. yf.Ticker().history() uses date as the index. 
                ticker_data = ticker_data[["Open", "High", "Low", "Close", "Volume"]]

                #Rename columns to include ticker, eg AAPL_Open
                ticker_data.columns = [f"{ticker}_{col}" for col in ticker_data.columns]

                new_data_frames.append(ticker_data)

        else:
            print(f"Loaded {ticker} entirely from cache.")

    #Merge new data into cache_df
    if new_data_frames: 
        new_df = pd.concat(new_data_frames, axis = 1) #axis = 1 adds columns to the right of the table instead of below

        if cache_df.empty:
            cache_df = new_df
        else:
            cache_df = cache_df.combine_first(new_df) #combine_first aligns the dates
        
        cache_df.to_csv(cache_path)

    #Extract necessary data from cache
    cols_to_keep = []
    for ticker in tickers:
        cols_to_keep.extend([f"{ticker}_Open", f"{ticker}_High", f"{ticker}_Low", f"{ticker}_Close", f"{ticker}_Volume"])

    #In case yfinance does not have all these columns for certain tickers
    valid_cols = []
    for c in cols_to_keep:
        if c in cache_df:
            valid_cols.append(c)
    
    final_df = cache_df.loc[start_ts:end_ts, valid_cols]

    #Clean data one last time just in case
    final_df = final_df.ffill().dropna()

    return final_df

    
