from __future__ import annotations
import numpy as np
import pandas as pd
from strategy_backtester.data import get_field
from strategy_backtester.strategies import RandomStrategy
from strategy_backtester.validation import BlockPermutationStrategy


SEED = 42


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _close_returns(price_df, ticker_subset=None) -> pd.DataFrame:
    close = get_field(price_df, "Close")
    if ticker_subset:
        close = close[ticker_subset]
    return close.pct_change().fillna(0.0)


def _wrapper(seed: int = SEED, block_size: int = 20) -> BlockPermutationStrategy:
    return BlockPermutationStrategy(RandomStrategy(), _rng(seed), block_size=block_size)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_block_size():
    wrapper = BlockPermutationStrategy(RandomStrategy(), _rng())
    assert wrapper.block_size == 20


def test_custom_block_size():
    wrapper = BlockPermutationStrategy(RandomStrategy(), _rng(), block_size=5)
    assert wrapper.block_size == 5


def test_init_stores_strategy_and_rng():
    strategy = RandomStrategy()
    rng = _rng()
    wrapper = BlockPermutationStrategy(strategy, rng, block_size=10)
    assert wrapper.strategy is strategy
    assert wrapper.rng is rng


# ---------------------------------------------------------------------------
# prepare() output is a valid PriceDataFrame
# ---------------------------------------------------------------------------

def test_prepare_preserves_index(price_df):
    result = _wrapper().prepare(price_df)
    assert result.index.equals(price_df.index)


def test_prepare_preserves_columns(price_df):
    result = _wrapper().prepare(price_df)
    assert set(result.columns) == set(price_df.columns)


def test_prepare_returns_valid_multiindex(price_df):
    result = _wrapper().prepare(price_df)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["Date", "Ticker"]


def test_prepare_no_nans(price_df):
    result = _wrapper().prepare(price_df)
    assert not result.isna().any().any()


def test_prepare_volume_is_int64(price_df):
    result = _wrapper().prepare(price_df)
    assert result["Volume"].dtype == "int64"


def test_prepare_high_geq_close(price_df):
    result = _wrapper().prepare(price_df)
    close = get_field(result, "Close")
    high = get_field(result, "High")
    assert (high >= close - 1e-8).all().all()


def test_prepare_low_leq_close(price_df):
    result = _wrapper().prepare(price_df)
    close = get_field(result, "Close")
    low = get_field(result, "Low")
    assert (low <= close + 1e-8).all().all()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_prepare_reproducible_with_same_seed(price_df):
    result1 = _wrapper(SEED).prepare(price_df)
    result2 = _wrapper(SEED).prepare(price_df)
    pd.testing.assert_frame_equal(result1, result2)


def test_prepare_different_with_different_seed(price_df):
    result1 = _wrapper(SEED).prepare(price_df)
    result2 = _wrapper(SEED + 1).prepare(price_df)
    close1 = get_field(result1, "Close")
    close2 = get_field(result2, "Close")
    assert not close1.equals(close2)


# ---------------------------------------------------------------------------
# Anchoring: first day's Close unchanged
# ---------------------------------------------------------------------------

def test_prepare_start_prices_unchanged(price_df):
    result = _wrapper().prepare(price_df)
    orig_close = get_field(price_df, "Close")
    perm_close = get_field(result, "Close")
    first_date = orig_close.index[0]
    pd.testing.assert_series_equal(
        orig_close.loc[first_date],
        perm_close.loc[first_date],
        check_names=False,
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# Block-specific behaviour
# ---------------------------------------------------------------------------

def test_block_returns_are_permutation_of_original_per_ticker(price_df, tickers):
    """
    The multiset of returns for each ticker (days 1..n-1) must be unchanged —
    block-shuffling only reorders them.
    """
    orig_returns = _close_returns(price_df)
    result = _wrapper().prepare(price_df)
    perm_returns = _close_returns(result)

    for ticker in tickers:
        orig_sorted = np.sort(orig_returns[ticker].values[1:])
        perm_sorted = np.sort(perm_returns[ticker].values[1:])
        np.testing.assert_allclose(orig_sorted, perm_sorted, rtol=1e-4)


def test_block_preserves_intra_block_order(price_df, tickers):
    """
    Within a single block, the sequence of returns must be preserved
    (only the order of whole blocks is permuted, not their internal order).
    Verify by checking that at least one contiguous run of returns from the
    original series appears unchanged (as a contiguous subsequence) in the
    permuted series for some ticker.
    """
    block_size = 20
    orig_returns = _close_returns(price_df)
    wrapper = _wrapper(SEED, block_size=block_size)
    result = wrapper.prepare(price_df)
    perm_returns = _close_returns(result)

    n = len(orig_returns)
    found_intact_block = False
    for ticker in tickers:
        orig_vals = orig_returns[ticker].values
        perm_vals = perm_returns[ticker].values
        # First block (indices 1..block_size) is the only block guaranteed
        # to start at a fixed offset regardless of permutation, but its
        # *position* may move. Search for it as a contiguous subsequence.
        first_block = orig_vals[1:1 + block_size]
        for start in range(1, n - block_size + 1):
            if np.allclose(perm_vals[start:start + block_size], first_block, rtol=1e-4):
                found_intact_block = True
                break
        if found_intact_block:
            break

    assert found_intact_block


def test_block_handles_indivisible_length(price_df):
    """Block size that doesn't divide evenly into series length must not crash."""
    wrapper = _wrapper(SEED, block_size=7)
    result = wrapper.prepare(price_df)
    assert not result.isna().any().any()


def test_block_size_one_equivalent_to_full_shuffle_structure(price_df):
    """
    block_size=1 should still produce a valid, fully-permuted price series
    (degenerates to per-day shuffling).
    """
    wrapper = _wrapper(SEED, block_size=1)
    result = wrapper.prepare(price_df)
    assert not result.isna().any().any()
    assert result.index.equals(price_df.index)


def test_block_larger_than_series_does_not_crash(price_df):
    """block_size >= series length should produce a single block (no-op shuffle order)."""
    n = len(price_df.index.get_level_values("Date").unique())
    wrapper = _wrapper(SEED, block_size=n + 10)
    result = wrapper.prepare(price_df)
    assert not result.isna().any().any()


# ---------------------------------------------------------------------------
# Block vs IID: block preserves more short-run structure
# ---------------------------------------------------------------------------

def test_block_shuffle_differs_from_iid_shuffle(price_df):
    """Sanity check: with the same seed, block-shuffle output differs from a
    fully independent per-day shuffle (different reconstructed price paths)."""
    from strategy_backtester.validation import IIDPermutationStrategy

    block_result = BlockPermutationStrategy(RandomStrategy(), _rng(SEED), block_size=20).prepare(price_df)
    iid_result = IIDPermutationStrategy(RandomStrategy(), _rng(SEED)).prepare(price_df)

    block_close = get_field(block_result, "Close")
    iid_close = get_field(iid_result, "Close")
    assert not block_close.equals(iid_close)


# ---------------------------------------------------------------------------
# generate() / should_rebalance() unchanged
# ---------------------------------------------------------------------------

def test_generate_and_should_rebalance_delegate(price_df):
    strategy = RandomStrategy()
    wrapper = _wrapper()

    dates = price_df.index.get_level_values("Date").unique().sort_values()
    as_of = dates[5]
    current_prices = price_df.loc[:as_of]

    assert wrapper.should_rebalance(as_of, dates[0], pd.Series(dtype=float), current_prices) is True
    weights = wrapper.generate(current_prices, as_of, pd.Series(dtype=float))
    assert weights.date == as_of