"""Unit tests for state space models."""
import datetime
import itertools
from typing import Optional, Union

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import DataObject, DrawFn

from compotime import LocalLevelForecaster, LocalTrendForecaster, models


@st.composite
def compositional_ts_array(
    draw: DrawFn,
    shape: Optional[tuple[int, int]] = None,
):
    """Strategy to generate a numpy array of compositional time series."""
    if not shape:
        shape = hnp.array_shapes(min_dims=2, max_dims=2, min_side=2)

    array = draw(
        hnp.arrays(
            np.dtype(float),
            shape=shape,
            elements=hnp.from_dtype(
                np.dtype(float),
                min_value=0.1,
                max_value=0.6,
                allow_nan=False,
            ),
        ),
    )
    # Add random jitter
    array = array + np.abs(np.random.default_rng(0).normal(0.0, 0.1, array.shape))
    return array / array.sum(axis=1)[:, None]


@st.composite
def compositional_ts(
    draw: DrawFn,
    index_type: Union[
        type[pd.RangeIndex],
        type[pd.PeriodIndex],
        type[pd.DatetimeIndex],
    ] = pd.DatetimeIndex,
    shape: Optional[tuple[int, int]] = None,
):
    """Strategy to generate a dataframe of compositional time series."""
    array = draw(compositional_ts_array(shape=shape))

    if index_type == pd.DatetimeIndex:
        ts_index = pd.date_range("2020-01-01", periods=len(array), freq="D")
    elif index_type == pd.PeriodIndex:
        ts_index = pd.period_range("2020-01", periods=len(array), freq="M")
    elif index_type == pd.RangeIndex:
        ts_index = pd.RangeIndex.from_range(range(len(array)))
    else:
        raise NotImplementedError

    return pd.DataFrame(array, index=ts_index)


@settings(max_examples=10, deadline=datetime.timedelta(seconds=20))
@given(compositional_ts())
def test_local_level_forecaster_should_predict_constant_values(time_series: pd.DataFrame):
    """Test that the forecasts of the ``LocalLevelForecaster`` are constant."""
    model = LocalLevelForecaster()
    model.fit(time_series)
    predictions = model.predict(horizon=10).to_numpy()
    assert (predictions[0, :] == predictions).all()


@pytest.mark.parametrize(
    ("model", "index_type", "shape"),
    list(
        itertools.product(
            [LocalLevelForecaster, LocalTrendForecaster],
            [pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex],
            [(12, 3), (15, 2), (10, 5)],
        ),
    ),
)
@settings(max_examples=5, deadline=datetime.timedelta(seconds=30))
@given(data=st.data())
def test_models_should_work_with_different_types_of_indexes(
    model: Union[type[LocalLevelForecaster], type[LocalTrendForecaster]],
    index_type: Union[type[pd.RangeIndex], type[pd.DatetimeIndex], type[pd.PeriodIndex]],
    shape: tuple[int, int],
    data: DataObject,
):
    """Test that models work with different types of pandas indexes."""
    time_series = data.draw(compositional_ts(index_type=index_type, shape=shape))
    model().fit(time_series).predict(5)


@given(st.lists(hnp.from_dtype(np.dtype(float)), min_size=1))
def test_unflatten_is_inverse_of_flatten(params: list[np.ndarray]):
    """Test that the ``_unflatten_params`` function is the inverse of ``_flatten_params``."""
    flat_params, shapes = models._flatten_params(params)
    unflattened_params = models._unflatten_params(flat_params, shapes)

    for original, unflattened in zip(params, unflattened_params):
        assert np.array_equal(original, unflattened, equal_nan=True)


@given(compositional_ts_array())
def test_inv_log_ratio_is_inverse_of_log_ratio(array: np.ndarray):
    """Test that the ``_inv_log_ratio`` function is the inverse of the ``_log_ratio``function.

    This condition should hold as long as the sum of the time series at a given timestamp always
    equals one.
    """
    assert np.allclose(array, models._inv_log_ratio(models._log_ratio(array)))
