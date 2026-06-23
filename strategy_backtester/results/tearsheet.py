from __future__ import annotations
import os
import numpy as np
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
# Markdown helpers
# ------------------------------------------------------------------

def _md_section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n"


def _md_kv_table(rows: list[tuple[str, str]]) -> str:
    """Two-column key/value markdown table."""
    lines = ["| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines) + "\n"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """General markdown table."""
    header_row    = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows     = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row] + data_rows) + "\n"


# ------------------------------------------------------------------
# Section builders
# ------------------------------------------------------------------

def _build_overview(cfg: dict) -> str:
    bt = cfg["backtest"]
    universe = bt.get("universe", "") or "custom"
    universe_label = "S&P 500" if universe == "sp500" else universe

    rows = [
        ("Strategy",        bt["strategy"].replace("_", " ").title()),
        ("Universe",        universe_label),
        ("Period",          f"{bt['start_date']} → {bt['end_date']}"),
        ("Initial Capital", f"${bt['initial_capital']:>,.2f}"),
    ]
    return _md_section("Overview", _md_kv_table(rows))


def _build_strategy_params(cfg: dict) -> str:
    strategy_name = cfg["backtest"]["strategy"]
    params = cfg.get("strategies", {}).get(strategy_name, {})
    if not params:
        return ""
    rows = [(k.replace("_", " ").title(), str(v)) for k, v in params.items()]
    return _md_section("Strategy Parameters", _md_kv_table(rows))


def _build_execution_params(cfg: dict) -> str:
    ex = cfg.get("execution", {})
    if not ex:
        return ""
    rows = [
        ("Commission",  f"{ex['commission_bps']} bps"),
        ("Spread",      f"{ex['spread_bps']} bps"),
        ("Slippage k",  str(ex['slippage_k'])),
        ("ADV Window",  str(ex['adv_window'])),
    ]
    return _md_section("Execution Parameters", _md_kv_table(rows))


def _build_performance_summary(result: BacktestResult) -> str:
    r = result.returns
    c = result.starting_capital

    rows = [
        ("Final Portfolio Value",   f"${calc_final_value(c, r):>,.2f}"),
        ("Cumulative Return",       f"{calc_cumulative_return(r):.2%}"),
        ("Annualised Return",       f"{calc_annualised_return(c, r):.2%}"),
        ("Annualised Volatility",   f"{calc_annualised_volatility(r):.2%}"),
        ("Sharpe Ratio",            f"{calc_sharpe_ratio(r):.4f}"),
        ("Sortino Ratio",           f"{calc_sortino_ratio(r):.4f}"),
        ("Calmar Ratio",            f"{calc_calmar_ratio(c, r):.2f}"),
        ("Max Drawdown",            f"{calc_max_drawdown(r):.2%}"),
        ("Max Drawdown Duration",   f"{calc_max_drawdown_duration(r)}"),
        ("Win Rate",                f"{calc_win_rate(r):.2%}"),
    ]
    return _md_section("Performance Summary", _md_kv_table(rows))


def _build_costs_summary(result: BacktestResult) -> str:
    total_cost   = result.costs.sum()
    avg_daily    = result.costs.mean()
    avg_turnover = result.turnover.mean()

    rows = [
        ("Total Transaction Costs", f"${total_cost:>,.2f}"),
        ("Avg Daily Costs",         f"${avg_daily:>,.4f}"),
        ("Avg Daily Turnover",      f"{avg_turnover:.4%}"),
    ]
    return _md_section("Transaction Costs", _md_kv_table(rows))


def _build_permutation_summary(result: PermutationResult) -> str:
    baseline_sharpe = calc_sharpe_ratio(result.baseline.returns)
    null_sharpes = np.array([
        calc_sharpe_ratio(r.returns) for r in result.null_distribution
    ])

    if result.p_value < 0.05:
        interpretation = "Statistically significant at the 5% level."
    elif result.p_value < 0.10:
        interpretation = "Marginal significance at the 10% level."
    else:
        interpretation = (
            "Not statistically significant. "
            "Cannot reject the null hypothesis of no predictive power."
        )

    rows = [
        ("Scheme",               result.scheme),
        ("N Permutations",       str(result.N)),
        ("Baseline Sharpe",      f"{baseline_sharpe:.4f}"),
        ("Mean Null Sharpe",     f"{null_sharpes.mean():.4f}"),
        ("Null Sharpe Std",      f"{null_sharpes.std():.4f}"),
        ("p-value (one-tailed)", f"{result.p_value:.4f}"),
        ("Interpretation",       interpretation),
    ]
    return _md_section("Permutation Test", _md_kv_table(rows))


def _build_walk_forward_summary(result: WalkForwardResult) -> str:
    oos_metrics = [f.oos_metric for f in result.folds]
    positive    = sum(1 for m in oos_metrics if m > 0)

    summary_rows = [
        ("Scheme",                  result.scheme),
        ("Metric",                  result.metric),
        ("Folds",                   str(len(result.folds))),
        ("Mean OOS Sharpe",         f"{np.mean(oos_metrics):.4f}"),
        ("Std OOS Sharpe",          f"{np.std(oos_metrics):.4f}"),
        ("Folds with Positive OOS", f"{positive} / {len(result.folds)}"),
    ]
    summary = _md_kv_table(summary_rows)

    # Per-fold detail table
    headers = ["Fold", "IS Start", "IS End", "OOS Start", "OOS End",
               "Best IS Sharpe", "OOS Sharpe", "Selected Params"]
    fold_rows = []
    for fold in result.folds:
        best_is     = max(fold.param_results, key=lambda x: x.is_metric).is_metric
        params_str  = ", ".join(f"{k}={v}" for k, v in fold.selected_params.items())
        fold_rows.append([
            str(fold.fold_idx),
            fold.is_start.strftime("%Y-%m-%d"),
            fold.is_end.strftime("%Y-%m-%d"),
            fold.oos_start.strftime("%Y-%m-%d"),
            fold.oos_end.strftime("%Y-%m-%d"),
            f"{best_is:.4f}",
            f"{fold.oos_metric:.4f}",
            params_str,
        ])
    fold_table = _md_table(headers, fold_rows)

    return _md_section("Walk-Forward Validation", summary + "\n" + fold_table)


# ------------------------------------------------------------------
# Public interface
# ------------------------------------------------------------------

def generate_tear_sheet(
    backtest_result: BacktestResult,
    permutation_result: PermutationResult,
    wf_result: WalkForwardResult,
    cfg: dict,
    output_path: str = "tearsheet.md",
) -> None:
    """
    Write a markdown tear sheet summarising backtest, permutation test,
    and walk-forward validation results.

    Parameters
    ----------
    backtest_result : BacktestResult
    permutation_result : PermutationResult
    wf_result : WalkForwardResult
    cfg : dict
        Full config dict (used for header metadata).
    output_path : str
        Destination path for the markdown file. Defaults to tearsheet.md
        in the current working directory.
    """
    sections = [
        "# Strategy Tear Sheet\n",
        _build_overview(cfg),
        _build_strategy_params(cfg),
        _build_execution_params(cfg),
        _build_performance_summary(backtest_result),
        _build_costs_summary(backtest_result),
        _build_permutation_summary(permutation_result),
        _build_walk_forward_summary(wf_result),
    ]
    content = "\n".join(s for s in sections if s)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Tear sheet written to {output_path}")