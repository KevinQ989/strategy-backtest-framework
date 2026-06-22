import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from strategy_backtester.core import PermutationResult, BacktestResult

def generate_dashboard(perm_result: PermutationResult, rolling_window: int = 126):
    """
    Generates a 2x2 dashboard to contain all plots. 

    Parameters:

    perm_result : PermutationResult
        The result object from PermutationTest.run()
    rolling_window : int
        Trading days for the rolling Sharpe ratio window (default is 126 days or approximately 6 months)
    """
    sns.set_theme(style = 'whitegrid')
    fig, axes = plt.subplots(2,2, figsize = (16,22))
    fig.suptitle("Strategy Validation Dashboard", fontsize = 16, fontweight = 'bold')

    baseline_result = perm_result.baseline

    _plot_permutation_envelope(axes[0,0], perm_result)
    _plot_underwater_drawdown(axes[0,1], baseline_result)
    _plot_rolling_sharpe(axes[1,0], baseline_result, rolling_window)
    _plot_daily_returns_histogram(axes[1,1], baseline_result)

    plt.tight_layout(h_pad = 3.0, w_pad = 2.0)
    plt.show()

def _plot_permutation_envelope(ax, perm_result: PermutationResult):
    baseline = perm_result.baseline
    starting_cap = baseline.starting_capital

    #Calculate baseline curve
    baseline_equity = (1+baseline.returns).cumprod() * starting_cap

    #Calculate permutation curves
    null_equities = []
    for null_res in perm_result.null_distribution:
        null_eq = (1+null_res.returns).cumprod() * starting_cap
        null_equities.append(null_eq)

    #Create df with rows for each date, and a column for each permutation
    null_df = pd.concat(null_equities, axis = 1)

    #Find envelope
    p05 = null_df.quantile(0.05, axis = 1)
    p50 = null_df.quantile(0.5, axis = 1)
    p95 = null_df.quantile(0.95, axis = 1)

    #Plot some permutations with low opacity
    sample_size = min(100, perm_result.N)
    sample_paths = null_df.sample(n = sample_size, axis = 1, random_state = 0)
    for col in sample_paths.columns: 
        ax.plot(sample_paths.index, sample_paths[col], color = 'gray', alpha = 0.2, linewidth = 1)

    #Highlight area between 5th and 95th percentile
    ax.fill_between(null_df.index, p05, p95, color = 'gray', alpha = 0.3, label = '5th-95th Envelope')

    #Plot median line and baseline 
    ax.plot(p50.index, p50, color = 'black', linestyle = '--', alpha = 0.6, label = 'Median Percentile')
    ax.plot(baseline_equity.index, baseline_equity, color = 'red', linewidth = 2, label = 'Baseline Strategy')

    #Formatting
    ax.set_title(f"Permutation  ({perm_result.N} Runs, p-value = {perm_result.p_value:.3f})", fontweight = 'bold')
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc = 'upper left')

    #Log scale to preserve percentage changes
    ax.set_yscale('log')

def _plot_underwater_drawdown(ax, result: BacktestResult):
    equity = (1+result.returns).cumprod() * result.starting_capital

    #Calculate high. Only remembers highest value up till that date.
    high = equity.cummax()

    #Calculate daily drawdown from high
    drawdown = (equity/high) - 1

    #Plot red underwater area
    ax.fill_between(drawdown.index, drawdown, 0, color = 'red', alpha = 0.3)
    ax.plot(drawdown.index, drawdown, color = 'red', linewidth = 1)
    ax.axhline(0, color = 'black', linewidth = 1.5) #axhline is axis horizontal line

    #Mark out worst day
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    ax.scatter(max_dd_date, max_dd, color = 'darkred', zorder = 5) #zorder brings to front of plot
    ax.annotate(f"Max DD: {max_dd:.2%}",
                xy = (max_dd_date, max_dd),
                xytext = (10,-10),
                textcoords = 'offset points', 
                color = 'darkred',
                fontweight = 'bold')

    #Formatting
    ax.set_title("Underwater Drawdown", fontweight = "bold")
    ax.set_ylabel("Drawdown Percentage")

    #Show percentages as -10% instead of -0.1
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

def _plot_rolling_sharpe(ax, result: BacktestResult, window: int):
    returns = result.returns

    #Calculate moving average and moving volatility
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    #Calculate Sharpe Ratio
    rolling_sharpe = np.where(rolling_std != 0, (rolling_mean/rolling_std) * np.sqrt(252), 0)

    #Convert back to pandas Series
    rolling_sharpe = pd.Series(rolling_sharpe, index = returns.index).dropna()

    #Generate plot
    ax.plot(rolling_sharpe.index, rolling_sharpe, color = 'black', linewidth = 1.5)
    ax.axhline(0, color = 'black', linewidth = 1.5)

    #Plot green for above 0 and red for below 0
    ax.fill_between(rolling_sharpe.index, rolling_sharpe, 0, where = (rolling_sharpe>= 0), color = 'green', alpha = 0.2)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe, 0, where = (rolling_sharpe < 0), color = 'red', alpha = 0.2)

    #Formatting
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontweight = 'bold')
    ax.set_ylabel("Annualised Sharpe Ratio")

def _plot_daily_returns_histogram(ax, result: BacktestResult):
    #Filter out days where return was 0
    returns = result.returns[result.returns != 0]

    #Plot histogram. kde draws smooth curve.
    sns.histplot(returns, bins = 50, kde = True, ax = ax, color = 'steelblue', edgecolor = 'black', alpha = 0.6)

    #Calculate key metrics
    mean_returns = returns.mean()
    p05 = returns.quantile(0.05)
    p95 = returns.quantile(0.95)

    #Draw vertical lines for metrics
    ax.axvline(0, color = 'black', linewidth = 1.5, linestyle = '-')
    ax.axvline(mean_returns, color = 'blue', linewidth = 2, linestyle = '--', label = f"Mean: {mean_returns:.2f}")
    ax.axvline(p05,color = 'red', linewidth = 1.5, linestyle = ':', label = f"5th Pctl: {p05:.2f}")
    ax.axvline(p95,color = 'green', linewidth = 1.5, linestyle = ':', label = f"95th Pctl: {p95:.2f}")

    #Formatting
    ax.set_title("Histogram of Daily Returns", fontweight = 'bold')
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")

    import matplotlib.ticker as mtick
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()