import pandas as pd
from data.loader import get_data
from portfolio.portfolio import PortfolioState
from execution.simulator import execute
from core.types import BacktestResult

class BacktestEngine:
    def __init__(
            self,
            tickers: list,
            start_date: str,
            end_date: str,
            initial_capital: float = 100000.0):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        #Download all data
        print("Downloading historical data...")
        self.historical_data = get_data(tickers,start_date,end_date)
        print("Data has been downloaded.")

        #Use python lists to store data first since they are faster than pd Series/Dataframe
        self.daily_returns = []
        self.daily_positions = []
        self.daily_costs = []
        self.daily_turnover = []

    def run_backtest(self, strategy) -> BacktestResult:
        print(f"Starting backtest for {strategy.__class__.__name__}.")
        
        unique_dates = self.historical_data.index.unique().sort_values()

        if len(unique_dates) == 0:
            raise ValueError:("No data downloaded. Check date range and tickers.")

        first_day = unique_dates[0]

        #Initialise portfolio
        portfolio = PortfolioState(date = first_day, starting_capital=self.initial_capital)
        last_rebalance = first_day

        #Step through dates day by day
        for i in range(len(unique_dates)):
            current_date = unique_dates[i]

            #Update portfolio to today's market prices
            portfolio.update_to_market(prices = self.historical_data, date = current_date)

            #Check with strategy if we should trade today
            if strategy.should_rebalance(date = current_date, 
                                         last_rebalance = last_rebalance,
                                         current_weights = portfolio.current_weights,
                                         prices = self.historical_data.loc[:current_date]):
                target_weights = strategy.generate(prices = self.historical_data.loc[:current_date],
                                  as_of = current_date,
                                  current_weights = portfolio.current_weights)
                
                #Execute trade on the next trading day. If it's the last day in the data, cannot execute trade
                if i+1 < len(unique_dates):
                    next_day = unique_dates[i+1]

                    next_day_data = self.historical_data.loc[next_day]

                    #Handle case where only 1 ticker is traded that day
                    if isinstance(next_day_data, pd.Series):
                        next_day_data = next_day_data.to_frame().T

                    #Search for price at next day open
                    open_prices = next_day_data.set_index('Ticker')['Open']

                    #Simulator executes trades and calculates slippages, commissions
                    execution_result = execute(pending = target_weights,
                                               state = portfolio,
                                               open_prices = open_prices,
                                               hist_prices = self.historical_data.loc[:current_date],
                                               date = next_day)

                    #Update portfolio
                    portfolio.update_to_execution(execution_result)

                    #Log transaction costs and turnover for T+1
                    self.daily_costs.append({'date':next_day, 'cost':execution_result.total_cost})
                    self.daily_turnover.append({'date':next_day, 'turnover':execution_result.turnover})
                    
            #End of day, track daily returns and positions
            self.daily_positions.append({'date':current_date, 'positions':portfolio.positions.to_dict()})
            self.daily_returns.append({'date':current_date, 'returns':portfolio.total_value})

        print("Backtest complete.")
        return self._generate_results()
    
    def _generate_results(self):
        #Convert lists to data format of BacktestResult class
        returns_series = pd.Series({d['date']:d['value'] for d in self.daily_returns})
        pct_returns = returns_series['value'].pct_change().fillna(0.0)
        positions_df = pd.DataFrame(self.daily_positions).set_index('date')
        costs_series = pd.Series({d['date']:d['cost'] for d in self.daily_costs}, dtype = float)
        turnover_series = pd.Series({d['date']:d['turnover'] for d in self.daily_turnover}, dtype = float)

        return BacktestResult(self,
                              returns = pct_returns,
                              positions = positions_df,
                              costs = costs_series,
                              turnover = turnover_series,
                              starting_capital = self.initial_capital,
                              metadata = {'tickers':self.tickers, 'start':self.start_date, 'end':self.end_date})