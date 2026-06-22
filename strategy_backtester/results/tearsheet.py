from __future__ import annotations
from strategy_backtester.core import (
    BacktestResult,
    PermutationResult,
    WalkForwardResult,
)


# ------------------------------------------------------------------
# Print helpers
# ------------------------------------------------------------------


def _to_label(name: str) -> str:
    return name.replace("_", " ").title()


def _print_backtest_result(label: str, result: BacktestResult) -> None:
    print(f"\n--- {label} ---")
    print(f"  Final portfolio value:  ${result.final_value:<12,.2f}")
    print(f"  Cumulative return:      {result.cumulative_return:<12.2%}")
    print(f"  Annualised return:      {result.annualised_return:<12.2%}")
    print(f"  Annualised volatility:  {result.annualised_volatility:<12.2%}")
    print(f"  Sharpe ratio:           {result.sharpe_ratio:<12.2f}")


def _print_permutation_result(result: PermutationResult) -> None:
    null_sharpes = [r.sharpe_ratio for r in result.null_distribution]
    print("\n--- Permutation Test Results ---")
    print(f"  Scheme:                 {result.scheme}")
    print(f"  N permutations:         {result.N:<8d}")
    print(f"  Baseline Sharpe:        {result.baseline.sharpe_ratio:<8.2f}")
    print(f"  Mean null Sharpe:       {np.mean(null_sharpes):<8.2f}")
    print(f"  Null Sharpe std:        {np.std(null_sharpes):<8.2f}")
    print(f"  p-value (one-tailed):   {result.p_value:<8.4f}")
    if result.p_value < 0.05:
        print("  Interpretation: Statistically significant at the 5% level.")
    elif result.p_value < 0.10:
        print("  Interpretation: Marginal significance at the 10% level.")
    else:
        print("  Interpretation: Not statistically significant. Cannot reject the null hypothesis of no predictive power.")


def _print_wf_result(result: WalkForwardResult) -> None:
    print("\n--- Walk-Forward Validation Results ---")
    print(f"  Scheme:  {result.scheme}")
    print(f"  Folds:   {len(result.folds):<8d}")
    print(f"  Metric:  {result.metric:<8s}")
    print()
    header = f"  {'Fold':<6} {'IS Start':<12} {'IS End':<12} {'OOS Start':<12} {'OOS End':<12} {'IS Sharpe':<12} {'OOS Sharpe':<12} Selected Params"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for fold in result.folds:
        best_is = max(fold.param_results, key=lambda x: x.is_metric)
        params_str = ", ".join(f"{k}={v}" for k, v in fold.selected_params.items())
        print(
            f"  {fold.fold_idx:<6d} "
            f"{fold.is_start.date().strftime("%Y-%m-%d"):<12} "
            f"{fold.is_end.date().strftime("%Y-%m-%d"):<12} "
            f"{fold.oos_start.date().strftime("%Y-%m-%d"):<12} "
            f"{fold.oos_end.date().strftime("%Y-%m-%d"):<12} "
            f"{best_is.is_metric:<12.2f} "
            f"{fold.oos_metric:<12.2f} "
            f"{params_str}"
        )
    oos_metrics = [f.oos_metric for f in result.folds]
    print()
    print(f"  Mean OOS {result.metric}: {np.mean(oos_metrics):.2f}")
    print(f"  Std OOS {result.metric}:  {np.std(oos_metrics):.2f}")
    positive_folds = sum(1 for m in oos_metrics if m > 0)
    print(f"  Folds with positive OOS metric: {positive_folds} / {len(result.folds)}")


def generate_tear_sheet(
    backtest_results: BacktestResult,
    permutation_result: PermutationResult,
    wf_result: WalkForwardResult,
    cfg: dict,
    output_dir: str = "tearsheet",
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
    cfg : dict
        Configuration dictionary containing parameters for the tear sheet.
    output_dir : str
        Directory where the tear sheet will be saved (default is "tearsheet").
    """
    _print_backtest_result(_to_label(cfg["backtest"]["strategy"]), backtest_results)
    _print_permutation_result(permutation_result)
    _print_wf_result(wf_result)