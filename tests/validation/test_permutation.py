from __future__ import annotations
import pandas as pd
import pytest
from unittest.mock import MagicMock
from strategy_backtester.core import BacktestResult, PermutationResult
from strategy_backtester.validation.permutation import PermutationTest
from strategy_backtester.validation import (
    IIDPermutationStrategy,
    BlockPermutationStrategy,
    RankPermutationStrategy,
)
from strategy_backtester.strategies import RandomStrategy, CrossSectionalMomentumStrategy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_PERMUTATIONS = 5
SEED = 42

# Small momentum strategy whose lookback fits within small_price_df (50 days)
SMALL_LOOKBACK, SMALL_SKIP, SMALL_PERCENT, SMALL_REBALANCE = 30, 5, 0.4, 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_strategy():
    return RandomStrategy()


@pytest.fixture
def small_momentum_strategy():
    return CrossSectionalMomentumStrategy(
        lookback=SMALL_LOOKBACK, skip=SMALL_SKIP, percent=SMALL_PERCENT, rebalance_freq=SMALL_REBALANCE
    )


@pytest.fixture
def perm_test(small_price_df, random_strategy):
    return PermutationTest(
        prices=small_price_df,
        strategy=random_strategy,
        scheme_cls=IIDPermutationStrategy,
        scheme_kwargs={},
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

def test_init(small_price_df, random_strategy):
    pt = PermutationTest(
        prices=small_price_df,
        strategy=random_strategy,
        scheme_cls=IIDPermutationStrategy,
        scheme_kwargs={},
        N=N_PERMUTATIONS,
    )
    assert pt.prices is small_price_df
    assert pt.strategy is random_strategy
    assert pt.scheme_cls is IIDPermutationStrategy
    assert pt.scheme_kwargs == {}
    assert pt.rng is not None
    assert pt.permutation_results is None


def test_init_default_values(small_price_df, random_strategy):
    pt = PermutationTest(
        prices=small_price_df,
        strategy=random_strategy,
        scheme_cls=IIDPermutationStrategy,
        scheme_kwargs={},
    )
    assert pt.N == 1000
    assert pt.metric == "sharpe"
    assert pt.initial_capital == 100_000.0
    assert pt.seed == 42
    assert pt.n_jobs == 1


# ---------------------------------------------------------------------------
# Test result structure
# ---------------------------------------------------------------------------

def test_run_returns_permutation_result(perm_result):
    assert isinstance(perm_result, PermutationResult)
    assert isinstance(perm_result.baseline, BacktestResult)
    assert len(perm_result.null_distribution) == N_PERMUTATIONS
    assert all(isinstance(r, BacktestResult) for r in perm_result.null_distribution)
    assert perm_result.scheme == "IIDPermutationStrategy"
    assert perm_result.metric == "sharpe"
    assert perm_result.N == N_PERMUTATIONS


def test_run_stores_result_on_instance(perm_test, perm_result):
    assert perm_test.permutation_results is perm_result


# ---------------------------------------------------------------------------
# Test scheme name reflects scheme_cls, not a hardcoded string
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme_cls,kwargs", [
    (IIDPermutationStrategy, {}),
    (BlockPermutationStrategy, {"block_size": 5}),
])
def test_scheme_name_matches_scheme_cls(small_price_df, random_strategy, scheme_cls, kwargs):
    pt = PermutationTest(
        prices=small_price_df,
        strategy=random_strategy,
        scheme_cls=scheme_cls,
        scheme_kwargs=kwargs,
        N=N_PERMUTATIONS,
        seed=SEED,
    )
    result = pt.run()
    assert result.scheme == scheme_cls.__name__


# ---------------------------------------------------------------------------
# Test p-value
# ---------------------------------------------------------------------------

def test_run_p_value_is_valid(perm_result):
    assert isinstance(perm_result.p_value, float)
    assert 0.0 <= perm_result.p_value <= 1.0


def test_run_p_value_one_tailed_definition(small_price_df, random_strategy):
    """
    p-value = fraction of null metrics >= baseline metric.
    Verify by mocking _calculate_metric so the baseline is lower than all
    null metrics -> p-value should be 1.0.
    """
    pt = PermutationTest(
        prices=small_price_df,
        strategy=random_strategy,
        scheme_cls=IIDPermutationStrategy,
        scheme_kwargs={},
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


def test_run_p_value_zero_when_baseline_beats_all_null(small_price_df, random_strategy):
    """Baseline higher than all null metrics -> p-value should be 0.0."""
    pt = PermutationTest(
        prices=small_price_df,
        strategy=random_strategy,
        scheme_cls=IIDPermutationStrategy,
        scheme_kwargs={},
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


def test_calculate_metric_unsupported_raises(small_price_df, random_strategy):
    pt = PermutationTest(
        prices=small_price_df,
        strategy=random_strategy,
        scheme_cls=IIDPermutationStrategy,
        scheme_kwargs={},
        N=N_PERMUTATIONS,
        metric="unsupported_metric",
    )
    dummy_result = MagicMock(spec=BacktestResult)
    dummy_result.returns = pd.Series([0.01, -0.005, 0.002])
    with pytest.raises(ValueError, match="Unsupported metric"):
        pt._calculate_metric(dummy_result)


# ---------------------------------------------------------------------------
# Test all schemes run end-to-end without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme_cls,kwargs", [
    (IIDPermutationStrategy, {}),
    (BlockPermutationStrategy, {"block_size": 5}),
])
def test_iid_and_block_schemes_complete_without_error(small_price_df, random_strategy, scheme_cls, kwargs):
    pt = PermutationTest(
        prices=small_price_df,
        strategy=random_strategy,
        scheme_cls=scheme_cls,
        scheme_kwargs=kwargs,
        N=N_PERMUTATIONS,
        seed=SEED,
    )
    result = pt.run()
    assert isinstance(result, PermutationResult)
    assert 0.0 <= result.p_value <= 1.0


def test_rank_scheme_completes_without_error(small_price_df, small_momentum_strategy):
    """RankPermutationStrategy requires a strategy with _compute_signal/_weights_from_signal."""
    pt = PermutationTest(
        prices=small_price_df,
        strategy=small_momentum_strategy,
        scheme_cls=RankPermutationStrategy,
        scheme_kwargs={},
        N=N_PERMUTATIONS,
        seed=SEED,
    )
    result = pt.run()
    assert isinstance(result, PermutationResult)
    assert 0.0 <= result.p_value <= 1.0
    assert result.scheme == "RankPermutationStrategy"


def test_rank_scheme_rejects_incompatible_strategy(small_price_df, random_strategy):
    """
    A strategy without _compute_signal/_weights_from_signal should raise
    TypeError as soon as a permutation is constructed (i.e. when .run() is
    called), not silently produce a result.
    """
    pt = PermutationTest(
        prices=small_price_df,
        strategy=random_strategy,
        scheme_cls=RankPermutationStrategy,
        scheme_kwargs={},
        N=1,
        seed=SEED,
    )
    with pytest.raises(TypeError, match="_compute_signal"):
        pt.run()


# ---------------------------------------------------------------------------
# Test scheme_kwargs are actually passed through to the wrapper
# ---------------------------------------------------------------------------

def test_scheme_kwargs_passed_to_wrapper(small_price_df, random_strategy):
    """A custom block_size in scheme_kwargs must affect the constructed wrapper."""
    pt = PermutationTest(
        prices=small_price_df,
        strategy=random_strategy,
        scheme_cls=BlockPermutationStrategy,
        scheme_kwargs={"block_size": 7},
        N=1,
        seed=SEED,
    )
    result = pt.run()
    assert result.scheme == "BlockPermutationStrategy"
    # Indirect check: result completes without error using the custom block_size
    assert len(result.null_distribution) == 1


# ---------------------------------------------------------------------------
# Test n_jobs parallel path produces equivalent structure to serial path
# ---------------------------------------------------------------------------

def test_n_jobs_parallel_matches_serial_structure(small_price_df, random_strategy):
    pt_serial = PermutationTest(
        prices=small_price_df,
        strategy=random_strategy,
        scheme_cls=IIDPermutationStrategy,
        scheme_kwargs={},
        N=3,
        seed=SEED,
        n_jobs=1,
    )
    pt_parallel = PermutationTest(
        prices=small_price_df,
        strategy=random_strategy,
        scheme_cls=IIDPermutationStrategy,
        scheme_kwargs={},
        N=3,
        seed=SEED,
        n_jobs=2,
    )
    result_serial = pt_serial.run()
    result_parallel = pt_parallel.run()

    assert len(result_serial.null_distribution) == len(result_parallel.null_distribution) == 3
    assert result_serial.scheme == result_parallel.scheme == "IIDPermutationStrategy"
    assert result_serial.metric == result_parallel.metric == "sharpe"
    assert result_serial.N == result_parallel.N == 3
    assert all(isinstance(r, BacktestResult) for r in result_serial.null_distribution)
    assert all(isinstance(r, BacktestResult) for r in result_parallel.null_distribution)