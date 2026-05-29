from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from strategy_backtester.core import BacktestResult, PermutationResult
from strategy_backtester.validation.permutation import PermutationTest
from strategy_backtester.validation.permutation_schemes import IIDScheme, BlockScheme, RanksScheme
from strategy_backtester.strategies import RandomStrategy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_PERMUTATIONS = 5
SEED = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy():
    return RandomStrategy()


@pytest.fixture
def iid_scheme():
    return IIDScheme()


@pytest.fixture
def perm_test(small_price_df, strategy, iid_scheme):
    return PermutationTest(
        prices=small_price_df,
        strategy=strategy,
        scheme=iid_scheme,
        N=N_PERMUTATIONS,
        metric="sharpe",
        initial_capital=100_000.0,
        seed=SEED,
    )


@pytest.fixture
def perm_result(perm_test):
    """Run the permutation test once and cache the result."""
    return perm_test.run()


# ---------------------------------------------------------------------------
# Test __init__
# ---------------------------------------------------------------------------

def test_init(small_price_df, strategy, iid_scheme):
    pt = PermutationTest(small_price_df, strategy, iid_scheme, N=N_PERMUTATIONS)
    assert pt.prices is small_price_df
    assert pt.strategy is strategy
    assert pt.scheme is iid_scheme
    assert pt.rng is not None
    assert pt.permutation_results is None


# ---------------------------------------------------------------------------
# Test result structure
# ---------------------------------------------------------------------------

def test_run_returns_permutation_result(perm_result):
    assert isinstance(perm_result, PermutationResult)
    assert isinstance(perm_result.baseline, BacktestResult)
    assert len(perm_result.null_distribution) == N_PERMUTATIONS
    assert all(isinstance(r, BacktestResult) for r in perm_result.null_distribution)
    assert perm_result.scheme == "IIDScheme"
    assert perm_result.metric == "sharpe"


def test_run_stores_result_on_instance(perm_test, perm_result):
    assert perm_test.permutation_results is perm_result


# ---------------------------------------------------------------------------
# Test p-value
# ---------------------------------------------------------------------------

def test_run_p_value_is_valid(perm_result):
    assert isinstance(perm_result.p_value, float)
    assert 0.0 <= perm_result.p_value <= 1.0


def test_run_p_value_one_tailed_definition(small_price_df, strategy, iid_scheme):
    """
    p-value = fraction of null metrics >= baseline metric.
    Verify by constructing a controlled case: mock _calculate_metric so the
    baseline is lower than all null metrics → p-value should be 1.0.
    """
    pt = PermutationTest(
        prices=small_price_df,
        strategy=strategy,
        scheme=iid_scheme,
        N=N_PERMUTATIONS,
        seed=SEED,
    )
    call_count = 0
    def mock_metric(result):
        nonlocal call_count
        call_count += 1
        # First call is baseline (return low value), rest are null (return high value)
        return 0.0 if call_count == 1 else 1.0

    pt._calculate_metric = mock_metric
    result = pt.run()
    assert result.p_value == pytest.approx(1.0)


def test_run_p_value_zero_when_baseline_beats_all_null(small_price_df, strategy, iid_scheme):
    """Baseline higher than all null metrics → p-value should be 0.0."""
    pt = PermutationTest(
        prices=small_price_df,
        strategy=strategy,
        scheme=iid_scheme,
        N=N_PERMUTATIONS,
        seed=SEED,
    )
    call_count = 0
    def mock_metric(result):
        nonlocal call_count
        call_count += 1
        return 1.0 if call_count == 1 else 0.0

    pt._calculate_metric = mock_metric
    result = pt.run()
    assert result.p_value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test null distribution properties
# ---------------------------------------------------------------------------

def test_null_distribution_returns_are_valid_series(perm_result):
    baseline_len = len(perm_result.baseline.returns)
    for result in perm_result.null_distribution:
        assert isinstance(result.returns, pd.Series)
        assert len(result.returns) == baseline_len


def test_null_distribution_not_all_identical(perm_result):
    """Null results should differ from each other — permutations must vary."""
    returns_sums = [r.returns.sum() for r in perm_result.null_distribution]
    assert len(set(returns_sums)) > 1


# ---------------------------------------------------------------------------
# Test _calculate_metric
# ---------------------------------------------------------------------------

def test_calculate_metric_sharpe_returns_float(perm_test, perm_result):
    metric = perm_test._calculate_metric(perm_result.baseline)
    assert isinstance(metric, float)


def test_calculate_metric_unsupported_raises(small_price_df, strategy, iid_scheme):
    pt = PermutationTest(
        prices=small_price_df,
        strategy=strategy,
        scheme=iid_scheme,
        N=N_PERMUTATIONS,
        metric="unsupported_metric",
    )
    dummy_result = MagicMock(spec=BacktestResult)
    dummy_result.returns = pd.Series([0.01, -0.005, 0.002])
    with pytest.raises(ValueError, match="Unsupported metric"):
        pt._calculate_metric(dummy_result)


# ---------------------------------------------------------------------------
# Test all schemes run without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme_cls,kwargs", [
    (IIDScheme, {}),
    (BlockScheme, {"block_size": 5}),
    (RanksScheme, {}),
])
def test_all_schemes_complete_without_error(small_price_df, strategy, scheme_cls, kwargs):
    pt = PermutationTest(
        prices=small_price_df,
        strategy=strategy,
        scheme=scheme_cls(**kwargs),
        N=N_PERMUTATIONS,
        seed=SEED,
    )
    result = pt.run()
    assert isinstance(result, PermutationResult)
    assert 0.0 <= result.p_value <= 1.0