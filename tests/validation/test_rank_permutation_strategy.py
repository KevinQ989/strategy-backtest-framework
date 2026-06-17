from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from strategy_backtester.strategies import RandomStrategy, CrossSectionalMomentumStrategy
from strategy_backtester.validation.rank_permutation_strategy import RankPermutationStrategy


LOOKBACK, SKIP, PERCENT, REBALANCE_FREQ = 252, 21, 0.1, 21


def _momentum_strategy():
    return CrossSectionalMomentumStrategy(
        lookback=LOOKBACK, skip=SKIP, percent=PERCENT, rebalance_freq=REBALANCE_FREQ
    )


# ---------------------------------------------------------------------------
# Construction / fail-fast contract
# ---------------------------------------------------------------------------

def test_init_accepts_strategy_with_signal_protocol():
    strategy = _momentum_strategy()
    rng = np.random.default_rng(0)
    # Should not raise
    RankPermutationStrategy(strategy, rng)


def test_init_raises_typeerror_for_strategy_without_signal_protocol():
    strategy = RandomStrategy()
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="_compute_signal"):
        RankPermutationStrategy(strategy, rng)


def test_init_error_message_names_strategy_class():
    strategy = RandomStrategy()
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="RandomStrategy"):
        RankPermutationStrategy(strategy, rng)


# ---------------------------------------------------------------------------
# prepare() / should_rebalance() unchanged
# ---------------------------------------------------------------------------

def test_prepare_is_passthrough(price_df):
    strategy = _momentum_strategy()
    rng = np.random.default_rng(0)
    wrapper = RankPermutationStrategy(strategy, rng)
    assert wrapper.prepare(price_df) is price_df


def test_should_rebalance_delegates(price_df):
    strategy = _momentum_strategy()
    rng = np.random.default_rng(0)
    wrapper = RankPermutationStrategy(strategy, rng)

    dates = price_df.index.get_level_values("Date").unique().sort_values()
    last_rebalance = dates[260]
    date = dates[260 + 21]

    direct = strategy.should_rebalance(date, last_rebalance, pd.Series(dtype=float), price_df.loc[:date])
    via_wrapper = wrapper.should_rebalance(date, last_rebalance, pd.Series(dtype=float), price_df.loc[:date])
    assert direct == via_wrapper


# ---------------------------------------------------------------------------
# generate(): empty signal short-circuit
# ---------------------------------------------------------------------------

def test_generate_empty_signal_returns_empty_weights(price_df):
    """
    Before lookback days of history are available, _compute_signal returns
    an empty Series. generate() must short-circuit and return empty weights
    without calling rng.permutation.
    """
    strategy = _momentum_strategy()
    rng = np.random.default_rng(0)
    wrapper = RankPermutationStrategy(strategy, rng)

    dates = price_df.index.get_level_values("Date").unique().sort_values()
    as_of = dates[LOOKBACK - 1]  # current_idx < lookback -> empty signal
    current_prices = price_df.loc[:as_of]

    state_before = rng.bit_generator.state
    weights = wrapper.generate(current_prices, as_of, pd.Series(dtype=float))
    state_after = rng.bit_generator.state

    assert weights.long_weights.empty
    assert weights.short_weights.empty
    assert weights.date == as_of
    # rng must not have been consumed on the empty-signal path
    assert state_before == state_after


# ---------------------------------------------------------------------------
# generate(): structural invariants of shuffled weights
# ---------------------------------------------------------------------------

@pytest.fixture
def as_of_with_signal(price_df):
    dates = price_df.index.get_level_values("Date").unique().sort_values()
    return dates[LOOKBACK + 5], price_df.loc[:dates[LOOKBACK + 5]]


def test_generate_shuffled_weights_preserve_structure(price_df, as_of_with_signal):
    as_of, current_prices = as_of_with_signal
    strategy = _momentum_strategy()

    signal = strategy._compute_signal(current_prices, as_of)
    assert not signal.empty
    q_expected = int(len(signal) * PERCENT)

    rng = np.random.default_rng(7)
    wrapper = RankPermutationStrategy(strategy, rng)
    weights = wrapper.generate(current_prices, as_of, pd.Series(dtype=float))

    assert len(weights.long_weights) == q_expected
    assert len(weights.short_weights) == q_expected
    assert set(weights.long_weights.unique()) == {0.5 / q_expected}
    assert set(weights.short_weights.unique()) == {-0.5 / q_expected}
    assert weights.gross_exposure == pytest.approx(1.0)

    overlap = set(weights.long_weights.index) & set(weights.short_weights.index)
    assert overlap == set()


def test_generate_shuffled_weights_match_unpermuted_exposure(price_df, as_of_with_signal):
    """
    Shuffling preserves the exposure structure exactly (same q, same per-leg
    weight magnitude, same gross exposure) — only ticker assignment changes.
    """
    as_of, current_prices = as_of_with_signal
    strategy = _momentum_strategy()

    direct = strategy.generate(current_prices, as_of, pd.Series(dtype=float))

    rng = np.random.default_rng(7)
    wrapper = RankPermutationStrategy(strategy, rng)
    shuffled = wrapper.generate(current_prices, as_of, pd.Series(dtype=float))

    assert len(direct.long_weights) == len(shuffled.long_weights)
    assert len(direct.short_weights) == len(shuffled.short_weights)
    assert direct.gross_exposure == pytest.approx(shuffled.gross_exposure)
    assert set(direct.long_weights.unique()) == set(shuffled.long_weights.unique())
    assert set(direct.short_weights.unique()) == set(shuffled.short_weights.unique())


# ---------------------------------------------------------------------------
# generate(): the shuffle actually shuffles
# ---------------------------------------------------------------------------

def test_generate_shuffle_changes_ticker_assignment(price_df, as_of_with_signal):
    """
    With a non-trivial universe (20 tickers, q=2), a shuffled signal should
    select different long/short tickers than the unshuffled signal with very
    high probability.
    """
    as_of, current_prices = as_of_with_signal
    strategy = _momentum_strategy()

    direct = strategy.generate(current_prices, as_of, pd.Series(dtype=float))

    rng = np.random.default_rng(7)
    wrapper = RankPermutationStrategy(strategy, rng)
    shuffled = wrapper.generate(current_prices, as_of, pd.Series(dtype=float))

    assert set(direct.long_weights.index) != set(shuffled.long_weights.index) or \
        set(direct.short_weights.index) != set(shuffled.short_weights.index)


def test_generate_different_seeds_produce_different_assignments(price_df, as_of_with_signal):
    as_of, current_prices = as_of_with_signal
    strategy = _momentum_strategy()

    results = []
    for seed in (1, 2, 3, 4, 5):
        rng = np.random.default_rng(seed)
        wrapper = RankPermutationStrategy(strategy, rng)
        w = wrapper.generate(current_prices, as_of, pd.Series(dtype=float))
        results.append(frozenset(w.long_weights.index) | frozenset(w.short_weights.index))

    assert len(set(results)) > 1


def test_generate_same_seed_reproducible(price_df, as_of_with_signal):
    as_of, current_prices = as_of_with_signal
    strategy = _momentum_strategy()

    rng1 = np.random.default_rng(99)
    wrapper1 = RankPermutationStrategy(strategy, rng1)
    w1 = wrapper1.generate(current_prices, as_of, pd.Series(dtype=float))

    rng2 = np.random.default_rng(99)
    wrapper2 = RankPermutationStrategy(strategy, rng2)
    w2 = wrapper2.generate(current_prices, as_of, pd.Series(dtype=float))

    pd.testing.assert_series_equal(w1.long_weights.sort_index(), w2.long_weights.sort_index())
    pd.testing.assert_series_equal(w1.short_weights.sort_index(), w2.short_weights.sort_index())


# ---------------------------------------------------------------------------
# generate(): does not mutate input prices
# ---------------------------------------------------------------------------

def test_generate_does_not_mutate_prices(price_df, as_of_with_signal):
    as_of, current_prices = as_of_with_signal
    strategy = _momentum_strategy()
    rng = np.random.default_rng(0)
    wrapper = RankPermutationStrategy(strategy, rng)

    before = current_prices.copy(deep=True)
    wrapper.generate(current_prices, as_of, pd.Series(dtype=float))
    pd.testing.assert_frame_equal(current_prices, before)