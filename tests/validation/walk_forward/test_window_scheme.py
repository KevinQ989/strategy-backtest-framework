from __future__ import annotations
import pytest
from strategy_backtester.validation import ExpandingWindowScheme, RollingWindowScheme, WindowScheme


# ---------------------------------------------------------------------------
# Test __init__ validation (shared by both concrete subclasses)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme_cls", [ExpandingWindowScheme, RollingWindowScheme])
def test_init_stores_win_in_and_win_out(scheme_cls):
    scheme = scheme_cls(win_in=100, win_out=50)
    assert scheme.win_in == 100
    assert scheme.win_out == 50


@pytest.mark.parametrize("scheme_cls", [ExpandingWindowScheme, RollingWindowScheme])
def test_init_zero_win_in_raises(scheme_cls):
    with pytest.raises(ValueError, match="win_in"):
        scheme_cls(win_in=0, win_out=50)


@pytest.mark.parametrize("scheme_cls", [ExpandingWindowScheme, RollingWindowScheme])
def test_init_negative_win_in_raises(scheme_cls):
    with pytest.raises(ValueError, match="win_in"):
        scheme_cls(win_in=-10, win_out=50)


@pytest.mark.parametrize("scheme_cls", [ExpandingWindowScheme, RollingWindowScheme])
def test_init_zero_win_out_raises(scheme_cls):
    with pytest.raises(ValueError, match="win_out"):
        scheme_cls(win_in=100, win_out=0)


@pytest.mark.parametrize("scheme_cls", [ExpandingWindowScheme, RollingWindowScheme])
def test_init_negative_win_out_raises(scheme_cls):
    with pytest.raises(ValueError, match="win_out"):
        scheme_cls(win_in=100, win_out=-5)


# ---------------------------------------------------------------------------
# Test WindowScheme is abstract — cannot be instantiated directly
# ---------------------------------------------------------------------------

def test_window_scheme_is_abstract():
    with pytest.raises(TypeError):
        WindowScheme(win_in=100, win_out=50)  # type: ignore