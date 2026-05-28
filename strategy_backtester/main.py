from strategy_backtester.engine.backtest import BacktestEngine
from strategy_backtester.strategies.random import RandomStrategy

if __name__ == "__main__":
    # Define backtest parameters
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    initial_capital = 100000.0

    # Initialize backtest engine
    engine = BacktestEngine(tickers, start_date, end_date, initial_capital)

    # Initialize strategy
    strategy = RandomStrategy()

    # Run backtest
    result = engine.run_backtest(strategy)

    # Print results
    print("Backtest completed.")
    print(f"Final portfolio value: ${result.final_value:.2f}")
    print(f"Cumulative return: {result.cumulative_return:.2%}")
    print(f"Annualized return: {result.annualized_return:.2%}")
    print(f"Annualized volatility: {result.annualized_volatility:.2%}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")