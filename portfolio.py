#keeps track of capital in the portfolio. 
#listens for SignalEvents (strategy.py) and decides how many shares to trade based on available capital. 
#outputs OrderEvents to the queue
#inputs FillEvents to update internal cash balance
#track value of portfolio based on price data