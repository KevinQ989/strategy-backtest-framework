from __future__ import annotations
import io
import base64
import calendar
import os
from datetime import date as _date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from strategy_backtester.core import (
    BacktestResult,
    PermutationResult,
    WalkForwardResult,
)
from .metrics import (
    calc_cumulative_return,
    calc_final_value,
    calc_annualised_return,
    calc_annualised_volatility,
    calc_sharpe_ratio,
    calc_sortino_ratio,
    calc_calmar_ratio,
    calc_max_drawdown,
    calc_max_drawdown_duration,
    calc_win_rate,
)

# ------------------------------------------------------------------
# Design tokens
# ------------------------------------------------------------------
_C_BG        = "#F7F8FC"
_C_SURFACE   = "#FFFFFF"
_C_BORDER    = "#DDE3EE"
_C_TEXT      = "#1C2333"
_C_MUTED     = "#64748B"
_C_ACCENT    = "#1B3A6B"   # deep navy — header bar, section titles
_C_POSITIVE  = "#1A6B45"   # forest green
_C_NEGATIVE  = "#B91C1C"   # deep red
_C_CHART_1   = "#1B3A6B"   # primary chart series
_C_CHART_2   = "#94A3B8"   # secondary / null distribution
_FONT_STACK  = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
_MONO_STACK  = "'SF Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace"


# ------------------------------------------------------------------
# Helpers — chart encoding
# ------------------------------------------------------------------

def _fig_to_img_tag(fig: plt.Figure, width: str = "100%") -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=_C_SURFACE)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{encoded}" style="width:{width};display:block;" />'


def _safe_sharpe(returns: pd.Series) -> float:
    try:
        return calc_sharpe_ratio(returns)
    except ValueError:
        return float("nan")


# ------------------------------------------------------------------
# Helpers — HTML primitives
# ------------------------------------------------------------------

def _fmt_ratio(val: float, fmt: str = ".4f") -> str:
    if np.isinf(val):  return "∞"
    if np.isnan(val):  return "N/A"
    return format(val, fmt)


def _card(title: str, body: str, full_width: bool = True) -> str:
    w = "100%" if full_width else "auto"
    return f"""
    <div class="card" style="width:{w}">
      <h2 class="card-title">{title}</h2>
      {body}
    </div>"""


def _kv_table(rows: list[tuple[str, str]]) -> str:
    trs = "".join(
        f'<tr><td class="kv-key">{k}</td><td class="kv-val">{v}</td></tr>'
        for k, v in rows
    )
    return f'<table class="kv-table"><tbody>{trs}</tbody></table>'


def _data_table(headers: list[str], rows: list[list[str]]) -> str:
    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"""
    <div class="table-scroll">
      <table class="data-table">
        <thead><tr>{ths}</tr></thead>
        <tbody>{trs}</tbody>
      </table>
    </div>"""


def _metric_grid(metrics: list[tuple[str, str, str | None]]) -> str:
    """
    Render a grid of metric tiles.
    metrics: list of (label, value, colour_class | None)
    colour_class: 'pos', 'neg', or None for neutral
    """
    tiles = []
    for label, value, colour in metrics:
        cls = f" val-{colour}" if colour else ""
        tiles.append(
            f'<div class="metric-tile">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value{cls}">{value}</div>'
            f'</div>'
        )
    return '<div class="metric-grid">' + "".join(tiles) + "</div>"


# ------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------

def _chart_monthly_returns(returns: pd.Series) -> str:
    """Heatmap of monthly returns."""
    monthly = (1 + returns).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M")

    years  = sorted(monthly.index.year.unique())
    months = list(range(1, 13))
    data   = np.full((len(years), 12), np.nan)
    for period, val in monthly.items():
        r = years.index(period.year)
        c = period.month - 1
        data[r, c] = val

    fig, ax = plt.subplots(figsize=(12, max(3, len(years) * 0.55)),
                           facecolor=_C_SURFACE)
    ax.set_facecolor(_C_SURFACE)

    vmax = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 0.05
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", [_C_NEGATIVE, _C_SURFACE, _C_POSITIVE]
    )
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im   = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(12))
    ax.set_xticklabels([calendar.month_abbr[m] for m in months],
                       fontsize=8, color=_C_MUTED)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=8, color=_C_MUTED)

    for r in range(len(years)):
        for c in range(12):
            v = data[r, c]
            if not np.isnan(v):
                colour = _C_SURFACE if abs(v) > vmax * 0.5 else _C_TEXT
                ax.text(c, r, f"{v:.1%}", ha="center", va="center",
                        fontsize=6.5, color=colour,
                        fontfamily=_MONO_STACK)

    ax.set_title("Monthly Returns", fontsize=10, color=_C_TEXT,
                 fontweight="600", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
    cbar.ax.tick_params(labelsize=7, colors=_C_MUTED)
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    return _fig_to_img_tag(fig)


def _chart_return_distribution(returns: pd.Series) -> str:
    """Histogram of daily returns."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5), facecolor=_C_SURFACE)
    ax.set_facecolor(_C_SURFACE)

    ax.hist(returns, bins=80, color=_C_CHART_1, alpha=0.75,
            edgecolor="none", density=True, label="Daily returns")

    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 300)
    normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, normal, color=_C_NEGATIVE, linewidth=1.5,
            linestyle="--", label="Normal fit", zorder=3)

    ax.axvline(mu, color=_C_ACCENT, linewidth=1.2, linestyle="-", alpha=0.8)
    ax.set_title("Return Distribution", fontsize=10, color=_C_TEXT,
                 fontweight="600", pad=8)
    ax.set_xlabel("Daily return", fontsize=8, color=_C_MUTED)
    ax.set_ylabel("Density", fontsize=8, color=_C_MUTED)
    ax.tick_params(labelsize=7, colors=_C_MUTED)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend(fontsize=7, framealpha=0)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(_C_BORDER)

    skew = returns.skew()
    kurt = returns.kurtosis()
    ax.text(0.97, 0.95,
            f"Skew: {skew:.2f}\nKurt: {kurt:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=_C_MUTED, fontfamily=_MONO_STACK)
    fig.tight_layout()
    return _fig_to_img_tag(fig)


def _chart_permutation_sharpes(
    baseline_metric: float,
    null_metrics: np.ndarray,
    p_value: float,
) -> str:
    """Histogram of permutation test metric values."""
    clean = null_metrics[~np.isnan(null_metrics)]

    fig, ax = plt.subplots(figsize=(5.5, 3.5), facecolor=_C_SURFACE)
    ax.set_facecolor(_C_SURFACE)

    ax.hist(clean, bins=50, color=_C_CHART_2, alpha=0.80,
            edgecolor="none", density=True, label="Null distribution")
    ax.axvline(baseline_metric, color=_C_ACCENT, linewidth=2,
               label=f"Baseline ({baseline_metric:.3f})", zorder=3)

    ax.set_title("Permutation Test — Null Metric Distribution",
                 fontsize=10, color=_C_TEXT, fontweight="600", pad=8)
    ax.set_xlabel("Metric", fontsize=8, color=_C_MUTED)
    ax.set_ylabel("Density", fontsize=8, color=_C_MUTED)
    ax.tick_params(labelsize=7, colors=_C_MUTED)
    ax.legend(fontsize=7, framealpha=0)

    ax.text(0.97, 0.95, f"p = {p_value:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=_C_MUTED, fontfamily=_MONO_STACK)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(_C_BORDER)

    fig.tight_layout()
    return _fig_to_img_tag(fig)


def _chart_wf_oos_sharpes(result: WalkForwardResult) -> str:
    """Bar chart of OOS metric by fold."""
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
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.02 if val >= 0 else -0.06),
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=7.5, color=_C_TEXT, fontfamily=_MONO_STACK)

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
    return _fig_to_img_tag(fig)


# ------------------------------------------------------------------
# Section builders
# ------------------------------------------------------------------

def _build_overview(cfg: dict) -> str:
    bt  = cfg["backtest"]
    ex  = cfg.get("execution", {})
    universe = bt.get("universe", "") or "custom"
    rows = [
        ("Strategy",        bt["strategy"].replace("_", " ").title()),
        ("Universe",        "S&P 500" if universe == "sp500" else universe),
        ("Period",          f"{bt['start_date']} → {bt['end_date']}"),
        ("Initial Capital", f"${bt['initial_capital']:>,.2f}"),
    ]
    if ex:
        rows += [
            ("Commission", f"{ex['commission_bps']} bps"),
            ("Spread",     f"{ex['spread_bps']} bps"),
            ("Slippage k", str(ex["slippage_k"])),
            ("ADV Window", f"{ex['adv_window']} days"),
        ]
    return _card("Overview", _kv_table(rows))


def _build_strategy_params(cfg: dict) -> str:
    name   = cfg["backtest"]["strategy"]
    params = cfg.get("strategies", {}).get(name, {})
    if not params:
        return ""
    rows = [(k.replace("_", " ").title(), str(v)) for k, v in params.items()]
    return _card("Strategy Parameters", _kv_table(rows))


def _build_performance_summary(result: BacktestResult) -> str:
    r, c = result.returns, result.starting_capital
    max_dd = calc_max_drawdown(r)

    def _sign(val: float) -> str | None:
        if np.isnan(val) or np.isinf(val): return None
        return "pos" if val > 0 else "neg" if val < 0 else None

    sharpe  = _safe_sharpe(r)
    sortino = calc_sortino_ratio(r)
    calmar  = calc_calmar_ratio(c, r)
    ann_ret = calc_annualised_return(c, r)

    metrics = [
        ("Final Portfolio Value",  f"${calc_final_value(c, r):>,.2f}",          _sign(calc_cumulative_return(r))),
        ("Cumulative Return",      f"{calc_cumulative_return(r):.2%}",           _sign(calc_cumulative_return(r))),
        ("Annualised Return",      f"{ann_ret:.2%}",                             _sign(ann_ret)),
        ("Annualised Volatility",  f"{calc_annualised_volatility(r):.2%}",       None),
        ("Sharpe Ratio",           _fmt_ratio(sharpe),                           _sign(sharpe)),
        ("Sortino Ratio",          _fmt_ratio(sortino),                          _sign(sortino)),
        ("Calmar Ratio",           _fmt_ratio(calmar, ".2f"),                    _sign(calmar)),
        ("Max Drawdown",           f"{abs(max_dd):.2%}",                         "neg" if max_dd < 0 else None),
        ("Max Drawdown Duration",  f"{calc_max_drawdown_duration(r)} trading days", None),
        ("Win Rate",               f"{calc_win_rate(r):.2%}",                   None),
        ("Return Skew",            _fmt_ratio(r.skew()),                         _sign(r.skew())),
        ("Excess Kurtosis",        _fmt_ratio(r.kurtosis()),                     None),
    ]
    grid    = _metric_grid(metrics)
    dist_chart = _chart_return_distribution(r)
    body = f"""
    <div class="perf-layout">
      <div class="perf-metrics">{grid}</div>
      <div class="perf-chart">{dist_chart}</div>
    </div>"""
    return _card("Performance Summary", body)


def _build_costs_summary(result: BacktestResult) -> str:
    rows = [
        ("Total Transaction Costs", f"${result.costs.sum():>,.2f}"),
        ("Avg Daily Costs",         f"${result.costs.mean():>,.4f}"),
        ("Avg Daily Turnover",      f"{result.turnover.mean():.4%}"),
    ]
    return _card("Transaction Costs", _kv_table(rows))


def _build_permutation_section(result: PermutationResult) -> str:
    baseline_sharpe = _safe_sharpe(result.baseline.returns)
    null_sharpes    = np.array([_safe_sharpe(r.returns) for r in result.null_distribution])
    clean           = null_sharpes[~np.isnan(null_sharpes)]

    if result.p_value < 0.05:
        interp = "Statistically significant at the 5% level."
        interp_cls = "pos"
    elif result.p_value < 0.10:
        interp = "Marginal significance at the 10% level."
        interp_cls = "neg"
    else:
        interp = "Not statistically significant — cannot reject the null hypothesis of no predictive power."
        interp_cls = "neg"

    rows = [
        ("Scheme",               result.scheme),
        ("N Permutations",       str(result.N)),
        ("Baseline Sharpe",      _fmt_ratio(baseline_sharpe)),
        ("Mean Null Sharpe",     _fmt_ratio(clean.mean() if len(clean) else float("nan"))),
        ("Null Sharpe Std",      _fmt_ratio(clean.std()  if len(clean) else float("nan"))),
        ("p-value (one-tailed)", f"{result.p_value:.4f}"),
        ("Interpretation",       f'<span class="val-{interp_cls}">{interp}</span>'),
    ]
    chart = _chart_permutation_sharpes(baseline_sharpe, null_sharpes, result.p_value)
    body  = f"""
    <div class="side-layout">
      <div>{_kv_table(rows)}</div>
      <div>{chart}</div>
    </div>"""
    return _card("Permutation Test", body)


def _build_walk_forward_section(result: WalkForwardResult) -> str:
    oos    = [f.oos_metric for f in result.folds]
    metric = result.metric.capitalize()
    n_pos  = sum(1 for m in oos if m > 0)
    summary_rows = [
        ("Scheme",                  result.scheme),
        ("Metric",                  metric),
        ("Folds",                   str(len(result.folds))),
        (f"Mean OOS {metric}",      _fmt_ratio(np.mean(oos))),
        (f"Std OOS {metric}",       _fmt_ratio(np.std(oos))),
        ("Folds with Positive OOS", f"{n_pos} / {len(result.folds)}"),
    ]
    headers = ["Fold", "IS Start", "IS End", "OOS Start", "OOS End",
               f"Best IS {metric}", f"OOS {metric}", "Selected Params"]
    fold_rows = []
    for fold in result.folds:
        best_is    = max(fold.param_results, key=lambda x: x.is_metric).is_metric
        params_str = ", ".join(f"{k}={v}" for k, v in fold.selected_params.items())
        oos_val    = fold.oos_metric
        colour     = _C_POSITIVE if oos_val > 0 else _C_NEGATIVE
        fold_rows.append([
            str(fold.fold_idx),
            fold.is_start.strftime("%Y-%m-%d"),
            fold.is_end.strftime("%Y-%m-%d"),
            fold.oos_start.strftime("%Y-%m-%d"),
            fold.oos_end.strftime("%Y-%m-%d"),
            _fmt_ratio(best_is),
            f'<span style="color:{colour};font-weight:600">{_fmt_ratio(oos_val)}</span>',
            f'<code>{params_str}</code>',
        ])
    bar_chart = _chart_wf_oos_sharpes(result)
    body = f"""
    {bar_chart}
    <div class="side-layout" style="margin-top:1.5rem">
      <div>{_kv_table(summary_rows)}</div>
      <div style="flex:2">{_data_table(headers, fold_rows)}</div>
    </div>"""
    return _card("Walk-Forward Validation", body)


# ------------------------------------------------------------------
# CSS
# ------------------------------------------------------------------

_CSS = f"""
  :root {{
    --bg:       {_C_BG};
    --surface:  {_C_SURFACE};
    --border:   {_C_BORDER};
    --text:     {_C_TEXT};
    --muted:    {_C_MUTED};
    --accent:   {_C_ACCENT};
    --pos:      {_C_POSITIVE};
    --neg:      {_C_NEGATIVE};
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: {_FONT_STACK};
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.6;
  }}
  .container {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
  }}
  /* Header */
  .ts-header {{
    background: var(--accent);
    color: #fff;
    padding: 2rem 2.5rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
  }}
  .ts-header h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }}
  .ts-header .subtitle {{
    font-size: 0.8rem;
    opacity: 0.65;
    margin-top: 0.25rem;
    font-family: {_MONO_STACK};
  }}
  /* Cards */
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .card-title {{
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--accent);
    padding-bottom: 0.75rem;
    border-bottom: 2px solid var(--accent);
    margin-bottom: 1.1rem;
  }}
  /* KV table */
  .kv-table {{
    width: 100%;
    border-collapse: collapse;
  }}
  .kv-table tr:not(:last-child) td {{
    border-bottom: 1px solid var(--border);
  }}
  .kv-key {{
    padding: 0.45rem 0.75rem 0.45rem 0;
    color: var(--muted);
    font-size: 0.82rem;
    white-space: nowrap;
    width: 40%;
  }}
  .kv-val {{
    padding: 0.45rem 0;
    font-size: 0.88rem;
    font-family: {_MONO_STACK};
    font-weight: 500;
  }}
  /* Metric tiles */
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 0.75rem;
  }}
  .metric-tile {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 0.75rem 0.875rem;
  }}
  .metric-label {{
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
  }}
  .metric-value {{
    font-size: 1.1rem;
    font-weight: 700;
    font-family: {_MONO_STACK};
    color: var(--text);
  }}
  .val-pos {{ color: var(--pos); }}
  .val-neg {{ color: var(--neg); }}
  /* Data table */
  .table-scroll {{ overflow-x: auto; }}
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }}
  .data-table th {{
    background: var(--bg);
    color: var(--muted);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.55rem 0.75rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  .data-table td {{
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
    font-family: {_MONO_STACK};
    white-space: nowrap;
  }}
  .data-table tr:last-child td {{ border-bottom: none; }}
  .data-table tr:hover td {{ background: var(--bg); }}
  code {{
    font-family: {_MONO_STACK};
    font-size: 0.78rem;
    color: var(--muted);
  }}
  /* Layouts */
  .perf-layout {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    align-items: start;
  }}
  .side-layout {{
    display: flex;
    gap: 1.5rem;
    align-items: start;
    flex-wrap: wrap;
  }}
  .side-layout > div {{ flex: 1; min-width: 260px; }}
  /* Monthly heatmap is always full width */
  .heatmap-card {{ width: 100%; }}
  @media (max-width: 780px) {{
    .perf-layout {{ grid-template-columns: 1fr; }}
    .side-layout {{ flex-direction: column; }}
  }}
"""

# ------------------------------------------------------------------
# Public interface
# ------------------------------------------------------------------

def generate_tear_sheet(
    backtest_result: BacktestResult,
    permutation_result: PermutationResult,
    wf_result: WalkForwardResult,
    cfg: dict,
    output_path: str = "tearsheet.html",
) -> None:
    """
    Write a self-contained HTML tear sheet summarising backtest,
    permutation test, and walk-forward validation results.

    Parameters
    ----------
    backtest_result : BacktestResult
    permutation_result : PermutationResult
    wf_result : WalkForwardResult
    cfg : dict
        Full config dict.
    output_path : str
        Destination path. Defaults to tearsheet.html in the CWD.
    """
    generated = _date.today().isoformat()
    strategy  = cfg["backtest"]["strategy"].replace("_", " ").title()

    monthly_heatmap = _chart_monthly_returns(backtest_result.returns)

    sections = [
        _build_overview(cfg),
        _build_strategy_params(cfg),
        _build_performance_summary(backtest_result),
        _card("Monthly Returns", monthly_heatmap),
        _build_costs_summary(backtest_result),
        _build_permutation_section(permutation_result),
        _build_walk_forward_section(wf_result),
    ]
    body = "\n".join(s for s in sections if s)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Tear Sheet — {strategy}</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="container">
    <header class="ts-header">
      <h1>Strategy Tear Sheet</h1>
      <p class="subtitle">Generated {generated} · {strategy}</p>
    </header>
    {body}
  </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Tear sheet written to {output_path}")