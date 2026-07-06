import calendar
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import seaborn as sns
from strategy_backtester.core import (
    BacktestResult,
    PermutationResult,
    WalkForwardResult,
)

# ------------------------------------------------------------------
# Design tokens — single source of truth for plots.py and tearsheet.py
# ------------------------------------------------------------------
_C_BG       = "#F7F8FC"
_C_SURFACE  = "#FFFFFF"
_C_BORDER   = "#DDE3EE"
_C_TEXT     = "#1C2333"
_C_MUTED    = "#64748B"
_C_ACCENT   = "#1B3A6B"
_C_POSITIVE = "#1A6B45"
_C_NEGATIVE = "#B91C1C"
_C_CHART_1  = "#1B3A6B"
_C_CHART_2  = "#94A3B8"
_FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
_MONO_STACK = "'SF Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace"


# ------------------------------------------------------------------
# Partner charts
# ------------------------------------------------------------------

def chart_permutation_envelope(perm_result: PermutationResult) -> plt.Figure:
    """
    Equity curves of baseline strategy vs. permutation null distribution.
    Shows 5th–95th percentile envelope, median null, and baseline.
    Log scale preserves proportional changes across the full period.
    """
    baseline = perm_result.baseline
    starting_cap = baseline.starting_capital
    baseline_equity = (1 + baseline.returns).cumprod() * starting_cap

    null_equities = [
        (1 + r.returns).cumprod() * starting_cap
        for r in perm_result.null_distribution
    ]
    null_df = pd.concat(null_equities, axis=1)
    p05 = null_df.quantile(0.05, axis=1)
    p50 = null_df.quantile(0.50, axis=1)
    p95 = null_df.quantile(0.95, axis=1)

    fig, ax = plt.subplots(figsize=(12, 4), facecolor=_C_SURFACE)
    ax.set_facecolor(_C_SURFACE)

    sample_paths = null_df.sample(n=min(100, perm_result.N), axis=1, random_state=0)
    for col in sample_paths.columns:
        ax.plot(sample_paths.index, sample_paths[col],
                color="gray", alpha=0.15, linewidth=0.8)

    ax.fill_between(null_df.index, p05, p95,
                    color=_C_CHART_2, alpha=0.3, label="5th–95th Envelope")
    ax.plot(p50.index, p50, color=_C_MUTED, linestyle="--",
            linewidth=1.2, alpha=0.8, label="Median Null")
    ax.plot(baseline_equity.index, baseline_equity,
            color=_C_ACCENT, linewidth=2, label="Baseline Strategy")

    ax.set_yscale("log")
    ax.set_title(
        f"Permutation Envelope  ({perm_result.N} runs, p = {perm_result.p_value:.3f})",
        fontsize=10, color=_C_TEXT, fontweight="600", pad=8,
    )
    ax.set_ylabel("Portfolio Value ($)", fontsize=8, color=_C_MUTED)
    ax.legend(fontsize=7, framealpha=0)
    ax.tick_params(labelsize=7, colors=_C_MUTED)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(_C_BORDER)

    fig.tight_layout()
    return fig


def chart_underwater_drawdown(result: BacktestResult) -> plt.Figure:
    """Rolling drawdown from the running peak."""
    equity   = (1 + result.returns).cumprod() * result.starting_capital
    drawdown = (equity / equity.cummax()) - 1

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=_C_SURFACE)
    ax.set_facecolor(_C_SURFACE)

    ax.fill_between(drawdown.index, drawdown, 0, color=_C_NEGATIVE, alpha=0.25)
    ax.plot(drawdown.index, drawdown, color=_C_NEGATIVE, linewidth=1)
    ax.axhline(0, color=_C_BORDER, linewidth=1.2)

    max_dd      = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax.scatter(max_dd_date, max_dd, color=_C_NEGATIVE, zorder=5, s=30)
    ax.annotate(
        f"Max DD: {max_dd:.2%}",
        xy=(max_dd_date, max_dd),
        xytext=(10, -10),
        textcoords="offset points",
        color=_C_NEGATIVE,
        fontsize=7.5,
        fontweight="bold",
    )

    ax.set_title("Underwater Drawdown", fontsize=10, color=_C_TEXT,
                 fontweight="600", pad=8)
    ax.set_ylabel("Drawdown", fontsize=8, color=_C_MUTED)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(labelsize=7, colors=_C_MUTED)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(_C_BORDER)

    fig.tight_layout()
    return fig


def chart_rolling_sharpe(result: BacktestResult, window: int) -> plt.Figure:
    """Rolling annualised Sharpe ratio. Zero-std windows are plotted as NaN."""
    returns      = result.returns
    rolling_mean = returns.rolling(window).mean()
    rolling_std  = returns.rolling(window).std()
    rolling_sharpe = pd.Series(
        np.where(rolling_std != 0, (rolling_mean / rolling_std) * np.sqrt(252), np.nan),
        index=returns.index,
    ).dropna()

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=_C_SURFACE)
    ax.set_facecolor(_C_SURFACE)

    ax.plot(rolling_sharpe.index, rolling_sharpe, color=_C_TEXT, linewidth=1.2)
    ax.axhline(0, color=_C_BORDER, linewidth=1.2)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe, 0,
                    where=(rolling_sharpe >= 0), color=_C_POSITIVE, alpha=0.2)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe, 0,
                    where=(rolling_sharpe < 0), color=_C_NEGATIVE, alpha=0.2)

    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontsize=10,
                 color=_C_TEXT, fontweight="600", pad=8)
    ax.set_ylabel("Annualised Sharpe", fontsize=8, color=_C_MUTED)
    ax.tick_params(labelsize=7, colors=_C_MUTED)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(_C_BORDER)

    fig.tight_layout()
    return fig


def chart_returns_histogram_kde(result: BacktestResult) -> plt.Figure:
    """
    KDE-smoothed histogram of non-zero daily returns.
    Marks mean, 5th, and 95th percentiles.
    """
    returns = result.returns[result.returns != 0]
    mean_r  = returns.mean()
    p05     = returns.quantile(0.05)
    p95     = returns.quantile(0.95)

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(5.5, 3.5), facecolor=_C_SURFACE)
        sns.histplot(returns, bins=50, kde=True, ax=ax,
                     color="steelblue", edgecolor="none", alpha=0.6)

    ax.axvline(0,      color=_C_TEXT,     linewidth=1.2)
    ax.axvline(mean_r, color=_C_ACCENT,   linewidth=1.5, linestyle="--",
               label=f"Mean: {mean_r:.3%}")
    ax.axvline(p05,    color=_C_NEGATIVE, linewidth=1.2, linestyle=":",
               label=f"5th Pctl: {p05:.3%}")
    ax.axvline(p95,    color=_C_POSITIVE, linewidth=1.2, linestyle=":",
               label=f"95th Pctl: {p95:.3%}")

    ax.set_title("Returns — KDE", fontsize=10, color=_C_TEXT,
                 fontweight="600", pad=8)
    ax.set_xlabel("Daily Return", fontsize=8, color=_C_MUTED)
    ax.set_ylabel("Frequency",   fontsize=8, color=_C_MUTED)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(labelsize=7, colors=_C_MUTED)
    ax.legend(fontsize=7, framealpha=0)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Tearsheet charts
# ------------------------------------------------------------------

def chart_monthly_returns(returns: pd.Series) -> plt.Figure:
    """Monthly returns heatmap (years × months)."""
    monthly = (1 + returns).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M")

    years = sorted(monthly.index.year.unique())
    data  = np.full((len(years), 12), np.nan)
    for period, val in monthly.items():
        data[years.index(period.year), period.month - 1] = val

    fig, ax = plt.subplots(figsize=(12, max(3, len(years) * 0.55)),
                           facecolor=_C_SURFACE)
    ax.set_facecolor(_C_SURFACE)

    vmax = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 0.05
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", [_C_NEGATIVE, _C_SURFACE, _C_POSITIVE]
    )
    im = ax.imshow(
        data,
        cmap=cmap,
        norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        aspect="auto",
    )

    ax.set_xticks(range(12))
    ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)],
                       fontsize=8, color=_C_MUTED)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=8, color=_C_MUTED)

    for r in range(len(years)):
        for c in range(12):
            v = data[r, c]
            if not np.isnan(v):
                colour = _C_SURFACE if abs(v) > vmax * 0.5 else _C_TEXT
                ax.text(c, r, f"{v:.1%}", ha="center", va="center",
                        fontsize=6.5, color=colour, fontfamily=_MONO_STACK)

    ax.set_title("Monthly Returns", fontsize=10, color=_C_TEXT,
                 fontweight="600", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
    cbar.ax.tick_params(labelsize=7, colors=_C_MUTED)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    return fig


def chart_return_distribution(returns: pd.Series) -> plt.Figure:
    """Histogram of daily returns with normal distribution overlay."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5), facecolor=_C_SURFACE)
    ax.set_facecolor(_C_SURFACE)

    ax.hist(returns, bins=80, color=_C_CHART_1, alpha=0.75,
            edgecolor="none", density=True, label="Daily returns")

    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 300)
    normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, normal, color=_C_NEGATIVE, linewidth=1.5,
            linestyle="--", label="Normal fit", zorder=3)
    ax.axvline(mu, color=_C_ACCENT, linewidth=1.2, alpha=0.8)

    ax.text(0.97, 0.95,
            f"Skew: {returns.skew():.2f}\nKurt: {returns.kurtosis():.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=_C_MUTED, fontfamily=_MONO_STACK)

    ax.set_title("Returns — Normal Fit", fontsize=10, color=_C_TEXT,
                 fontweight="600", pad=8)
    ax.set_xlabel("Daily return", fontsize=8, color=_C_MUTED)
    ax.set_ylabel("Density",      fontsize=8, color=_C_MUTED)
    ax.tick_params(labelsize=7, colors=_C_MUTED)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend(fontsize=7, framealpha=0)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(_C_BORDER)

    fig.tight_layout()
    return fig


def chart_permutation_null_sharpes(
    baseline_metric: float,
    null_metrics: np.ndarray,
    p_value: float,
) -> plt.Figure:
    """Histogram of permutation test null metric values with baseline marked."""
    clean = null_metrics[~np.isnan(null_metrics)]

    fig, ax = plt.subplots(figsize=(5.5, 3.5), facecolor=_C_SURFACE)
    ax.set_facecolor(_C_SURFACE)

    ax.hist(clean, bins=50, color=_C_CHART_2, alpha=0.80,
            edgecolor="none", density=True, label="Null distribution")
    ax.axvline(baseline_metric, color=_C_ACCENT, linewidth=2,
               label=f"Baseline ({baseline_metric:.3f})", zorder=3)

    ax.text(0.97, 0.95, f"p = {p_value:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=_C_MUTED, fontfamily=_MONO_STACK)

    ax.set_title("Null Metric Distribution", fontsize=10, color=_C_TEXT,
                 fontweight="600", pad=8)
    ax.set_xlabel("Metric",  fontsize=8, color=_C_MUTED)
    ax.set_ylabel("Density", fontsize=8, color=_C_MUTED)
    ax.tick_params(labelsize=7, colors=_C_MUTED)
    ax.legend(fontsize=7, framealpha=0)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(_C_BORDER)

    fig.tight_layout()
    return fig


def chart_wf_oos_sharpes(result: WalkForwardResult) -> plt.Figure:
    """Bar chart of OOS metric value per walk-forward fold."""
    folds   = [f.fold_idx for f in result.folds]
    metrics = [f.oos_metric for f in result.folds]
    colours = [_C_POSITIVE if m > 0 else _C_NEGATIVE for m in metrics]

    fig, ax = plt.subplots(figsize=(max(5, len(folds) * 1.2), 3.2),
                           facecolor=_C_SURFACE)
    ax.set_facecolor(_C_SURFACE)

    bars = ax.bar(folds, metrics, color=colours, alpha=0.85, width=0.6)
    ax.axhline(0, color=_C_BORDER, linewidth=1)
    ax.axhline(np.mean(metrics), color=_C_ACCENT, linewidth=1.2,
               linestyle="--", alpha=0.7, label=f"Mean ({np.mean(metrics):.3f})")

    for bar, val in zip(bars, metrics):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.02 if val >= 0 else -0.06),
            f"{val:.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=7.5, color=_C_TEXT, fontfamily=_MONO_STACK,
        )

    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {f}" for f in folds], fontsize=8, color=_C_MUTED)
    ax.set_ylabel("OOS Sharpe", fontsize=8, color=_C_MUTED)
    ax.set_title("Walk-Forward OOS Sharpe by Fold", fontsize=10,
                 color=_C_TEXT, fontweight="600", pad=8)
    ax.tick_params(labelsize=7, colors=_C_MUTED)
    ax.legend(fontsize=7, framealpha=0)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(_C_BORDER)

    fig.tight_layout()
    return fig