from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from strategy_backtester.data import get_field
from strategy_backtester.validation import (
    RanksScheme,
    IIDScheme,
    BlockScheme
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
SEED = 42
 
 
def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)
 
 
def _close_returns(price_df, ticker_subset=None) -> pd.DataFrame:
    """Wide (Date x Ticker) returns matrix from a PriceDataFrame."""
    close = get_field(price_df, "Close")
    if ticker_subset:
        close = close[ticker_subset]
    return close.pct_change().fillna(0.0)


# ---------------------------------------------------------------------------
# Test BasePermutationScheme
# ---------------------------------------------------------------------------

@pytest.fixture(params=["iid", "block", "ranks"])
def scheme(request):
    if request.param == "iid":
        return IIDScheme()
    elif request.param == "block":
        return BlockScheme(block_size=20)
    elif request.param == "ranks":
        return RanksScheme()


def test_permute_preserves_index(scheme, price_df):
    result = scheme.permute(price_df, _rng())
    assert result.index.equals(price_df.index)


def test_permute_preserves_columns(scheme, price_df):
    result = scheme.permute(price_df, _rng())
    assert set(result.columns) == set(price_df.columns)


def test_permute_returns_valid_price_dataframe(scheme, price_df):
    """Output must pass make_price_dataframe validation without raising."""
    result = scheme.permute(price_df, _rng())
    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["Date", "Ticker"]


def test_permute_no_nans(scheme, price_df):
    result = scheme.permute(price_df, _rng())
    assert not result.isna().any().any()


def test_permute_volume_is_int64(scheme, price_df):
    result = scheme.permute(price_df, _rng())
    assert result["Volume"].dtype == "int64"


def test_permute_high_geq_close(scheme, price_df):
    """High must always be >= Close after reconstruction."""
    result = scheme.permute(price_df, _rng())
    close = get_field(result, "Close")
    high = get_field(result, "High")
    assert (high >= close - 1e-8).all().all()


def test_permute_low_leq_close(scheme, price_df):
    """Low must always be <= Close after reconstruction."""
    result = scheme.permute(price_df, _rng())
    close = get_field(result, "Close")
    low = get_field(result, "Low")
    assert (low <= close + 1e-8).all().all()

 
def test_permute_reproducible_with_same_seed(scheme, price_df):
    result1 = scheme.permute(price_df, _rng(SEED))
    result2 = scheme.permute(price_df, _rng(SEED))
    pd.testing.assert_frame_equal(result1, result2)


def test_permute_different_with_different_seed(scheme, price_df):
    result1 = scheme.permute(price_df, _rng(SEED))
    result2 = scheme.permute(price_df, _rng(SEED + 1))
    close1 = get_field(result1, "Close")
    close2 = get_field(result2, "Close")
    assert not close1.equals(close2)


def test_permute_start_prices_unchanged(scheme, price_df):
    """Day one Close must match original — reconstruction is anchored to it."""
    result = scheme.permute(price_df, _rng())
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
# Test RanksScheme
# ---------------------------------------------------------------------------

def test_ranks_cross_sectional_returns_same_set(price_df):
    """Every date must have the same set of return values as the original."""
    scheme = RanksScheme()
    orig_returns = _close_returns(price_df)
    result = scheme.permute(price_df, _rng())
    perm_returns = _close_returns(result)
    for date in orig_returns.index[1:]:
        orig_sorted = np.sort(orig_returns.loc[date].values)
        perm_sorted = np.sort(perm_returns.loc[date].values)
        np.testing.assert_allclose(orig_sorted, perm_sorted, rtol=1e-4)


def test_ranks_ticker_return_series_differs_from_original(price_df, tickers):
    """Each ticker's return series should differ from original after permutation."""
    scheme = RanksScheme()
    orig_returns = _close_returns(price_df)
    result = scheme.permute(price_df, _rng())
    perm_returns = _close_returns(result)
    n_different = sum(
        not orig_returns[t].equals(perm_returns[t]) for t in tickers
    )
    assert n_different > len(tickers) // 2


# ---------------------------------------------------------------------------
# Test IIDScheme
# ---------------------------------------------------------------------------

def test_iid_breaks_temporal_autocorrelation_iid(price_df, tickers):
    """
    Shuffling should reduce lag-1 autocorrelation of returns toward zero
    on average across many permutations.
    """
    scheme = IIDScheme()
    orig_returns = _close_returns(price_df)
    orig_autocorr = orig_returns[tickers[0]].autocorr(lag=1)
 
    n_trials = 20
    perm_autocorrs = []
    rng = _rng()
    for _ in range(n_trials):
        result = scheme.permute(price_df, rng)
        perm_returns = _close_returns(result)
        perm_autocorrs.append(perm_returns[tickers[0]].autocorr(lag=1))
 
    mean_perm_autocorr = np.mean(perm_autocorrs)
    assert abs(mean_perm_autocorr) < abs(orig_autocorr) + 0.1


def test_iid_shuffles_independently_per_ticker(price_df, tickers):
    """Two tickers should receive different shufflings — not the same permutation."""
    scheme = IIDScheme()
    rng = _rng()
    result = scheme.permute(price_df, rng)
    perm_returns = _close_returns(result)
    assert not perm_returns[tickers[0]].equals(perm_returns[tickers[1]])


# ---------------------------------------------------------------------------
# Test BlockScheme
# ---------------------------------------------------------------------------

def test_block_default_block_size():
    scheme = BlockScheme()
    assert scheme.block_size == 20


def test_block_custom_block_size():
    scheme = BlockScheme(block_size=5)
    assert scheme.block_size == 5


def test_block_handles_indivisible_length(price_df):
    """Block size that doesn't divide evenly into series length must not crash."""
    scheme = BlockScheme(block_size=7)
    result = scheme.permute(price_df, _rng())
    assert not result.isna().any().any()
