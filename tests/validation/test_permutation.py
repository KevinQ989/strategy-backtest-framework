from __future__ import annotations
import pytest
from strategy_backtester.core import BacktestResult, PermutationResult
from strategy_backtester.data import PriceDataFrame
from strategy_backtester.validation import (
    PermutationTest,
    BasePermutationScheme,
    RanksScheme,
    IIDScheme,
    BlockScheme
)