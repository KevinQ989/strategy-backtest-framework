#Load data from yfinance
#draws info from main.py to decide which companies to import. 
import yfinance as yf
import datetime
import pandas as pd

def get_data(tickers, start_date, end_date): #Input list of tickers, YIELDS daily snapshot of prices
    master_df = pd.DataFrame()

    for ticker in tickers: 
        ticker_data = yf.Ticker(ticker).history(start = start_date, end = end_date, repair = True) #Download data for each ticker
        
        ticker_data = ticker_data[["Open", "High", "Low", "Close", "Volume"]] #Filter out important columns

        ticker_data.columns = [f"{ticker}_{col}" for col in ticker_data.columns] #Rename columns to include ticker name, eg AAPL_Close

        if master_df.empty:
            master_df = ticker_data
        else:
            master_df = master_df.join(ticker_data, how = 'outer') #Merge into master calendar

    master_df = master_df.ffill().dropna() #If a row is missing, use the previous day's price. If price is NA (eg company did not exist before start_date), delete those rows. 

    #Generate daily snapshot to send to queue
    for date, row in master_df.iterrows(): #iterrows returns date and row data, row by row.
        daily_snapshot = {"date": date} #Initialise snapshot

        for ticker in tickers: 
            daily_snapshot[ticker] = {
                "Open": row[f"{ticker}_Open"],
                "High": row[f"{ticker}_High"],
                "Low": row[f"{ticker}_Low"],
                "Close": row[f"{ticker}_Close"],
                "Volume": row[f"{ticker}_Volume"],
            } #Create a nested dictionary for each ticker
        
        yield daily_snapshot


 