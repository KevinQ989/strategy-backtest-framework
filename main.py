import numpy as np
from strategy_backtester.engine import BacktestEngine
from strategy_backtester.strategies import RandomStrategy
from strategy_backtester.data import load_data
from strategy_backtester.validation import PermutationTest, IIDScheme, RanksScheme, BlockScheme

if __name__ == "__main__":
    # Define backtest parameters
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    initial_capital = 100000.0

    # Load price data
    prices = load_data(tickers, start_date, end_date)
    print("Data loaded successfully.")

    # Run backtest
    print("\n\nRunning backtest...")
    metadata = {
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital
    }
    engine = BacktestEngine(prices, metadata=metadata, initial_capital=initial_capital)
    result = engine.run_backtest(RandomStrategy())
    print("Backtest completed successfully.")

    # Print results
    print(f"Final portfolio value: ${result.final_value:.2f}")
    print(f"Cumulative return: {result.cumulative_return:.2%}")
    print(f"Annualised return: {result.annualised_return:.2%}")
    print(f"Annualised volatility: {result.annualised_volatility:.2%}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")

    # Run permutation test
    print("\n\nRunning permutation test...")
    perm_test = PermutationTest(
        prices = prices,
        strategy = RandomStrategy(),
        scheme = BlockScheme(block_size=5),
        N = 1000,
        metric = "sharpe",
        initial_capital = initial_capital,
        seed = 42
    )
    perm_results = perm_test.run()
    print("Permutation test completed successfully.")
    print(f"Permutation test p-value: {perm_results.p_value:.4f}")
    print(f"Baseline Sharpe ratio: {perm_results.baseline.sharpe_ratio:.2f}")
    print(f"Mean null Sharpe ratio: {np.mean([res.sharpe_ratio for res in perm_results.null_distribution]):.2f}")