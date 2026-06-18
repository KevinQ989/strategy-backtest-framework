from __future__ import annotations
import pandas as pd
from strategy_backtester.data import PriceDataFrame, get_field
from strategy_backtester.portfolio import PortfolioState
from strategy_backtester.execution import execute
from strategy_backtester.core import ExecutionResult, BacktestResult
from strategy_backtester.strategies import BaseStrategy

class BacktestEngine:
    def __init__(
            self,
            prices: PriceDataFrame,
            strategy: BaseStrategy,
            metadata: dict = None,
            initial_capital: float = 100000.0
        ):
        self.historical_data = strategy.prepare(prices)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.metadata = metadata

        #Use python lists to store data first since they are faster than pd Series/Dataframe
        self.daily_total_value = []
        self.daily_positions = []
        self.daily_costs = []
        self.daily_turnover = []
        self._has_run = False


    def run_backtest(self) -> BacktestResult:
        if self._has_run:
            raise RuntimeError("Backtest has already been run. Create a new BacktestEngine instance to run another backtest.")
        self._has_run = True

        unique_dates = self.historical_data.index.get_level_values(0).unique().sort_values()
        if len(unique_dates) == 0:
            raise ValueError("No data downloaded. Check date range and tickers.")
        close_matrix = get_field(self.historical_data, "Close")

        # Day 1
        first_day = unique_dates[0]
        portfolio = PortfolioState(date = first_day, starting_capital=self.initial_capital)
        last_rebalance = first_day

        portfolio.update_to_market(close_prices = close_matrix.loc[first_day], date = first_day)
        self._log_day(date = first_day, portfolio = portfolio, execution_result = None)

        #Step through dates day by day
        for i in range(1, len(unique_dates)):
            current_date = unique_dates[i]
            current_prices = self.historical_data.loc[:current_date]
            execution_result = None

            #Update portfolio to today's market prices
            portfolio.update_to_market(close_prices = close_matrix.loc[current_date], date = current_date)

            #Check with strategy if we should trade today
            if self.strategy.should_rebalance(
                date = current_date, 
                last_rebalance = last_rebalance,
                current_weights = portfolio.current_weights,
                prices = current_prices
            ):
                target_weights = self.strategy.generate(
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
            
                print(f"Date: {portfolio.date.date()}, Num Positions: {len(portfolio.positions)}, Total Value: ${portfolio.total_value:,.2f}")
            # Log daily returns, positions, costs, turnover
            self._log_day(date = current_date, portfolio = portfolio, execution_result = execution_result)

        return self._generate_results()


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
        # Returns
        values = [d['total_value'] for d in self.daily_total_value]
        dates  = [d['date'] for d in self.daily_total_value]
        total_value_series = pd.Series(values, index=dates)
        returns_series = total_value_series.pct_change().fillna(0.0)

        # Positions
        positions_df = pd.DataFrame.from_records(
            [d['positions'] for d in self.daily_positions],
            index=pd.DatetimeIndex([d['date'] for d in self.daily_positions])
        ).fillna(0.0)
        positions_df.index.name = 'date'

        # Costs and turnover
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