from __future__ import annotations
import numpy as np
import pandas as pd
from strategy_backtester.strategies import RandomStrategy, CrossSectionalMomentumStrategy
from strategy_backtester.validation.permutation_strategy import PermutationStrategyWrapper


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_init_stores_strategy_and_rng():
    strategy = RandomStrategy()
    rng = np.random.default_rng(0)
    wrapper = PermutationStrategyWrapper(strategy, rng)
    assert wrapper.strategy is strategy
    assert wrapper.rng is rng


# ---------------------------------------------------------------------------
# prepare() delegation
# ---------------------------------------------------------------------------

def test_prepare_delegates_to_wrapped_strategy(price_df):
    strategy = RandomStrategy()
    rng = np.random.default_rng(0)
    wrapper = PermutationStrategyWrapper(strategy, rng)

    # BaseStrategy.prepare is a no-op passthrough, so both should return
    # the same object (identity), confirming delegation rather than a copy.
    result = wrapper.prepare(price_df)
    expected = strategy.prepare(price_df)
    assert result is expected
    assert result is price_df


# ---------------------------------------------------------------------------
# generate() delegation
# ---------------------------------------------------------------------------

def test_generate_delegates_to_wrapped_strategy(price_df, tickers):
    """
    generate() with no override must produce the exact same PortfolioWeights
    as calling the wrapped strategy's generate() directly, for the same inputs.
    """
    lookback, skip, percent, rebalance_freq = 252, 21, 0.1, 21
    strategy = CrossSectionalMomentumStrategy(
        lookback=lookback, skip=skip, percent=percent, rebalance_freq=rebalance_freq
    )
    rng = np.random.default_rng(0)
    wrapper = PermutationStrategyWrapper(strategy, rng)

    dates = price_df.index.get_level_values("Date").unique().sort_values()
    as_of = dates[lookback + 5]  # ensure enough history for a non-empty signal
    current_weights = pd.Series(dtype=float)
    current_prices = price_df.loc[:as_of]

    direct = strategy.generate(current_prices, as_of, current_weights)
    via_wrapper = wrapper.generate(current_prices, as_of, current_weights)

    pd.testing.assert_series_equal(direct.long_weights, via_wrapper.long_weights)
    pd.testing.assert_series_equal(direct.short_weights, via_wrapper.short_weights)
    assert direct.date == via_wrapper.date


def test_generate_unchanged_does_not_consume_rng(price_df, tickers):
    """
    The base wrapper's generate() is a pure passthrough — it must not draw
    from rng. Confirm rng state is unchanged after a call.
    """
    strategy = CrossSectionalMomentumStrategy(lookback=252, skip=21, percent=0.1, rebalance_freq=21)
    rng = np.random.default_rng(0)
    wrapper = PermutationStrategyWrapper(strategy, rng)

    dates = price_df.index.get_level_values("Date").unique().sort_values()
    as_of = dates[260]
    current_prices = price_df.loc[:as_of]

    state_before = rng.bit_generator.state
    wrapper.generate(current_prices, as_of, pd.Series(dtype=float))
    state_after = rng.bit_generator.state

    assert state_before == state_after


# ---------------------------------------------------------------------------
# should_rebalance() delegation
# ---------------------------------------------------------------------------

def test_should_rebalance_delegates_to_wrapped_strategy(price_df):
    strategy = CrossSectionalMomentumStrategy(lookback=252, skip=21, percent=0.1, rebalance_freq=21)
    rng = np.random.default_rng(0)
    wrapper = PermutationStrategyWrapper(strategy, rng)

    dates = price_df.index.get_level_values("Date").unique().sort_values()
    last_rebalance = dates[260]
    current_prices = price_df.loc[:dates[260 + 21]]

    for date in (dates[260 + 5], dates[260 + 21], dates[260 + 22]):
        direct = strategy.should_rebalance(date, last_rebalance, pd.Series(dtype=float), price_df.loc[:date])
        via_wrapper = wrapper.should_rebalance(date, last_rebalance, pd.Series(dtype=float), price_df.loc[:date])
        assert direct == via_wrapper


# ---------------------------------------------------------------------------
# Subclassing / overriding
# ---------------------------------------------------------------------------

def test_subclass_can_override_single_method(price_df):
    """
    A subclass overriding only generate() must still delegate prepare()
    and should_rebalance() to the wrapped strategy unchanged.
    """
    class OnlyGenerateOverride(PermutationStrategyWrapper):
        def generate(self, prices, as_of, current_weights):
            # Return a fixed, recognisable result instead of delegating
            return self.strategy.generate(prices, as_of, current_weights)

    strategy = RandomStrategy()
    rng = np.random.default_rng(0)
    wrapper = OnlyGenerateOverride(strategy, rng)

    # prepare() still passes through unchanged
    assert wrapper.prepare(price_df) is price_df

    dates = price_df.index.get_level_values("Date").unique().sort_values()
    # should_rebalance still delegates (RandomStrategy always returns True)
    assert wrapper.should_rebalance(dates[1], dates[0], pd.Series(dtype=float), price_df.loc[:dates[1]]) is True