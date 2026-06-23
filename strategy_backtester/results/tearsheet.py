from __future__ import annotations
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
)


# ------------------------------------------------------------------
# Print helpers
# ------------------------------------------------------------------


def _build_backtest_result_md(result: BacktestResult) -> str:
    return (
        f"--- Backtest Results ---\n"
        f"  Final portfolio value:  ${calc_final_value(result.starting_capital, result.returns):<12,.2f}\n"
        f"  Cumulative return:      {calc_cumulative_return(result.returns):<12.2%}\n"
        f"  Annualised return:      {calc_annualised_return(result.starting_capital, result.returns):<12.2%}\n"
        f"  Annualised volatility:  {calc_annualised_volatility(result.returns):<12.2%}\n"
        f"  Sharpe ratio:           {calc_sharpe_ratio(result.returns):<12.2f}\n"
    )


def _build_permutation_result_md(result: PermutationResult) -> str:
    null_sharpes = [calc_sharpe_ratio(r.returns) for r in result.null_distribution]
    return (
        f"--- Permutation Test Results ---\n"
        f"  Scheme:                 {result.scheme}\n"
        f"  N permutations:         {result.N:<8d}\n"
        f"  Baseline Sharpe:        {calc_sharpe_ratio(result.baseline.returns):<8.2f}\n"
        f"  Mean null Sharpe:       {np.mean(null_sharpes):<8.2f}\n"
        f"  Null Sharpe std:        {np.std(null_sharpes):<8.2f}\n"
        f"  p-value (one-tailed):   {result.p_value:<8.4f}\n"
    )


def _build_wf_result_md(result: WalkForwardResult) -> str:
    return (
        f"--- Walk-Forward Validation Results ---\n"
        f"  Scheme:  {result.scheme}\n"
        f"  Folds:   {len(result.folds):<8d}\n"
        f"  Metric:  {result.metric:<8s}\n\n"
        f"  {'Fold':<6} {'IS Start':<12} {'IS End':<12} {'OOS Start':<12} {'OOS End':<12} {'IS Sharpe':<12} {'OOS Sharpe':<12} Selected Params\n"
        + "\n".join(
            f"  {fold.fold_idx:<6d} "
            f"{fold.is_start.date().strftime('%Y-%m-%d'):<12} "
            f"{fold.is_end.date().strftime('%Y-%m-%d'):<12} "
            f"{fold.oos_start.date().strftime('%Y-%m-%d'):<12} "
            f"{fold.oos_end.date().strftime('%Y-%m-%d'):<12} "
            f"{max(fold.param_results, key=lambda x: x.is_metric).is_metric:<12.2f} "
            f"{fold.oos_metric:<12.2f} "
            + ", ".join(f"{k}={v}" for k, v in fold.selected_params.items())
            for fold in result.folds
        )
        + "\n\n"
        f"  Mean OOS {result.metric}: {np.mean([f.oos_metric for f in result.folds]):.2f}\n"
        f"  Std OOS {result.metric}:  {np.std([f.oos_metric for f in result.folds]):.2f}\n"
        f"  Folds with positive OOS metric: {sum(1 for m in [f.oos_metric for f in result.folds] if m > 0)} / {len(result.folds)}\n"
    )


def generate_tear_sheet(
    backtest_results: BacktestResult,
    permutation_result: PermutationResult,
    wf_result: WalkForwardResult,
) -> None:
    """
    Generates a structured summary report (tear sheet) for the backtest,
    permutation test, and walk-forward validation.

    Parameters
    ----------
    backtest_results : BacktestResult
        The result of the backtest on the original data.
    permutation_result : PermutationResult
        The result of the permutation test.
    wf_result : WalkForwardResult
        The result of the walk-forward validation.
    """
    backtest_md = _build_backtest_result_md(backtest_results)
    permutation_md = _build_permutation_result_md(permutation_result)
    wf_md = _build_wf_result_md(wf_result)

    markdown_content = f"""# Tear Sheet

{backtest_md}

{permutation_md}

{wf_md}
    """
    with open("tearsheet.md", "w") as f:
        f.write(markdown_content)