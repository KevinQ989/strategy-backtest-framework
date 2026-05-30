from __future__ import annotations
import pandas as pd
from strategy_backtester.data import PriceDataFrame
from strategy_backtester.portfolio import PortfolioState
from strategy_backtester.execution import execute
from strategy_backtester.core import ExecutionResult, BacktestResult

class BacktestEngine:
    def __init__(
            self,
            prices: PriceDataFrame,
            metadata: dict = None,
            initial_capital: float = 100000.0
        ):
        self.historical_data = prices
        self.initial_capital = initial_capital
        self.metadata = metadata

        #Use python lists to store data first since they are faster than pd Series/Dataframe
        self.daily_total_value = []
        self.daily_positions = []
        self.daily_costs = []
        self.daily_turnover = []


    def run_backtest(self, strategy) -> BacktestResult:
        self._reset_state()

        unique_dates = self.historical_data.index.get_level_values(0).unique().sort_values()
        if len(unique_dates) == 0:
            raise ValueError("No data downloaded. Check date range and tickers.")

        # Day 1
        first_day = unique_dates[0]
        portfolio = PortfolioState(date = first_day, starting_capital=self.initial_capital)
        last_rebalance = first_day

        portfolio.update_to_market(prices = self.historical_data, date = first_day)
        self._log_day(date = first_day, portfolio = portfolio, execution_result = None)

        #Step through dates day by day
        for i in range(1, len(unique_dates)):
            current_date = unique_dates[i]
            current_prices = self.historical_data.loc[
                self.historical_data.index.get_level_values('Date') <= current_date
            ]
            execution_result = None

            #Update portfolio to today's market prices
            portfolio.update_to_market(prices = self.historical_data, date = current_date)

            #Check with strategy if we should trade today
            if strategy.should_rebalance(
                date = current_date, 
                last_rebalance = last_rebalance,
                current_weights = portfolio.current_weights,
                prices = current_prices
            ):
                target_weights = strategy.generate(
                    prices = current_prices,
                    as_of = current_date,
                    current_weights = portfolio.current_weights
                )
                
                #Execute trade on the next trading day. If it's the last day in the data, cannot execute trade
                if i+1 < len(unique_dates):
                    next_day = unique_dates[i + 1]
                    next_day_open = self.historical_data.xs(next_day, level='Date')['Open']

                    #Simulator executes trades and calculates slippages, commissions
                    execution_result = execute(
                        pending = target_weights,
                        state = portfolio,
                        open_prices = next_day_open,
                        hist_prices = current_prices,
                        date = next_day
                    )

                    #Update portfolio
                    portfolio.update_to_execution(execution_result)
                    last_rebalance = current_date
                    
            # Log daily returns, positions, costs, turnover
            self._log_day(date = current_date, portfolio = portfolio, execution_result = execution_result)

        return self._generate_results()
    

    def _reset_state(self):
        self.daily_total_value = []
        self.daily_positions = []
        self.daily_costs = []
        self.daily_turnover = []


    def _log_day(
        self,
        date: pd.Timestamp,
        portfolio: PortfolioState,
        execution_result: ExecutionResult | None
    ) -> None:
        self.daily_total_value.append({
            'date': date,
            'total_value':portfolio.total_value
        })
        self.daily_positions.append({
            'date': date,
            'positions': portfolio.positions.to_dict()
        })
        self.daily_costs.append({
            'date': date,
            'cost':execution_result.total_cost if execution_result else 0.0
        })
        self.daily_turnover.append({
            'date': date,
            'turnover':execution_result.turnover if execution_result else 0.0
        })

    
    def _generate_results(self):
        daily_returns = []
        for i in range(len(self.daily_total_value)):
            if i == 0:
                daily_returns.append({
                    'date':self.daily_total_value[i]['date'],
                    'returns':0.0
                })
            else:
                prev_value = self.daily_total_value[i-1]['total_value']
                curr_value = self.daily_total_value[i]['total_value']
                daily_returns.append({
                    'date':self.daily_total_value[i]['date'],
                    'returns':(curr_value - prev_value) / prev_value
                })
        
        #Convert lists to data format of BacktestResult class
        returns_series = pd.Series({d['date']:d['returns'] for d in daily_returns})
        positions_df = pd.DataFrame(self.daily_positions).set_index('date')['positions'].apply(pd.Series).fillna(0.0)
        costs_series = pd.Series({d['date']:d['cost'] for d in self.daily_costs}, dtype = float)
        turnover_series = pd.Series({d['date']:d['turnover'] for d in self.daily_turnover}, dtype = float)

        return BacktestResult(
            returns = returns_series,
            positions = positions_df,
            costs = costs_series,
            turnover = turnover_series,
            starting_capital = self.initial_capital,
            metadata = self.metadata
        )