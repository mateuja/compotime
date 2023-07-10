"""Unit tests for preprocess functions."""
import pandas as pd
import pytest
from hypothesis import given
from numpy import testing as npt

from compotime import preprocess

from ._utils import compositional_ts


@given(compositional_ts())
def test_treat_small_should_return_compositional_ts(time_series: pd.DataFrame):
    """Test that ``treat_small`` always returns a compositional time series."""
    res = preprocess.treat_small(time_series, 0.05)

    npt.assert_allclose(res.to_numpy().sum(axis=1), 1)


@given(compositional_ts())
def test_treat_small_should_raise_error_if_minimum_is_too_large(time_series: pd.DataFrame):
    """Test that ``treat_small`` raises a ``ValueError`` when the ``minimum`` argument is too large.

    At most, the value should be equal to 1.0 / time_series.shape[1].
    """
    error_msg = "It is possible to satisfy the sum one constraint with the given ``minimum`` value."
    with pytest.raises(ValueError, match=error_msg):
        preprocess.treat_small(time_series, 1.0 / time_series.shape[1] + 0.01)
