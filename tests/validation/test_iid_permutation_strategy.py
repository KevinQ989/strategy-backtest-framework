from __future__ import annotations
import numpy as np
import pandas as pd
from strategy_backtester.data import get_field
from strategy_backtester.strategies import RandomStrategy
from strategy_backtester.validation.iid_permutation_strategy import IIDPermutationStrategy


SEED = 42


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _close_returns(price_df, ticker_subset=None) -> pd.DataFrame:
    close = get_field(price_df, "Close")
    if ticker_subset:
        close = close[ticker_subset]
    return close.pct_change().fillna(0.0)


def _wrapper(seed: int = SEED) -> IIDPermutationStrategy:
    return IIDPermutationStrategy(RandomStrategy(), _rng(seed))


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
# IID-specific behaviour
# ---------------------------------------------------------------------------

def test_iid_breaks_temporal_autocorrelation(price_df, tickers):
    """
    Shuffling should reduce lag-1 autocorrelation of returns toward zero
    on average across many permutations.
    """
    orig_returns = _close_returns(price_df)
    orig_autocorr = orig_returns[tickers[0]].autocorr(lag=1)

    n_trials = 20
    perm_autocorrs = []
    rng = _rng()
    for _ in range(n_trials):
        wrapper = IIDPermutationStrategy(RandomStrategy(), rng)
        result = wrapper.prepare(price_df)
        perm_returns = _close_returns(result)
        perm_autocorrs.append(perm_returns[tickers[0]].autocorr(lag=1))

    mean_perm_autocorr = np.mean(perm_autocorrs)
    assert abs(mean_perm_autocorr) < abs(orig_autocorr) + 0.1


def test_iid_shuffles_independently_per_ticker(price_df, tickers):
    """Two tickers should receive different shufflings — not the same permutation."""
    result = _wrapper().prepare(price_df)
    perm_returns = _close_returns(result)
    assert not perm_returns[tickers[0]].equals(perm_returns[tickers[1]])


def test_iid_returns_are_a_permutation_of_original_per_ticker(price_df, tickers):
    """
    The multiset of returns for each ticker (days 1..n-1) must be unchanged —
    only their order is shuffled.
    """
    orig_returns = _close_returns(price_df)
    result = _wrapper().prepare(price_df)
    perm_returns = _close_returns(result)

    for ticker in tickers:
        orig_sorted = np.sort(orig_returns[ticker].values[1:])
        perm_sorted = np.sort(perm_returns[ticker].values[1:])
        np.testing.assert_allclose(orig_sorted, perm_sorted, rtol=1e-4)


# ---------------------------------------------------------------------------
# Volume is shuffled independently of price
# ---------------------------------------------------------------------------

def test_volume_is_permutation_of_original_per_ticker(price_df, tickers):
    """Volume values for each ticker are shuffled, not regenerated."""
    orig_volume = get_field(price_df, "Volume")
    result = _wrapper().prepare(price_df)
    perm_volume = get_field(result, "Volume")

    for ticker in tickers:
        orig_sorted = np.sort(orig_volume[ticker].values)
        perm_sorted = np.sort(perm_volume[ticker].values)
        np.testing.assert_array_equal(orig_sorted, perm_sorted)


# ---------------------------------------------------------------------------
# generate() / should_rebalance() unchanged
# ---------------------------------------------------------------------------

def test_generate_and_should_rebalance_delegate(price_df):
    strategy = RandomStrategy()
    rng = _rng()
    wrapper = IIDPermutationStrategy(strategy, rng)

    dates = price_df.index.get_level_values("Date").unique().sort_values()
    as_of = dates[5]
    current_prices = price_df.loc[:as_of]

    assert wrapper.should_rebalance(as_of, dates[0], pd.Series(dtype=float), current_prices) is True
    # generate() delegates — RandomStrategy.generate uses global np.random,
    # so just confirm it returns a valid PortfolioWeights without error
    weights = wrapper.generate(current_prices, as_of, pd.Series(dtype=float))
    assert weights.date == as_of