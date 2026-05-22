import pandas as pd


def test_total_value_start(start_portfolio_state):
    assert start_portfolio_state.total_value == 100000