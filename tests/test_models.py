"""Unit tests for state space models."""
import datetime
import itertools
from typing import Union

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import DataObject

from compotime import LocalLevelForecaster, LocalTrendForecaster, models
from compotime.errors import FreqInferenceError, InvalidIndexError, LogRatioTransformError

from ._utils import compositional_ts, compositional_ts_array


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
            [pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex, pd.Index],
            [(12, 3), (15, 2), (10, 5)],
        ),
    ),
)
@settings(max_examples=10, deadline=datetime.timedelta(seconds=30))
@given(data=st.data())
def test_models_should_work_with_different_types_of_indexes(
    model: type[Union[LocalLevelForecaster, LocalTrendForecaster]],
    index_type: type[Union[pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]],
    shape: tuple[int, int],
    data: DataObject,
):
    """Test that models work with different types of pandas indexes."""
    time_series = data.draw(compositional_ts(index_type=index_type, shape=shape))
    fcsts = model().fit(time_series).predict(5)

    assert fcsts.index.name == time_series.index.name == "date"


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
    transformed_array, base_col_idx = models._log_ratio(array)
    assert np.allclose(array, models._inv_log_ratio(transformed_array, base_col_idx))


@pytest.mark.parametrize("model", [(LocalLevelForecaster), (LocalTrendForecaster)])
def test_forecasts_should_be_invariant_to_choice_of_base_ts(
    model: type[Union[LocalLevelForecaster, LocalTrendForecaster]],
):
    """Test that the forecasts of the models are invariant to the choice of the base time series."""
    time_series = pd.DataFrame(
        {
            "date": pd.period_range("2020-01", "2020-12", freq="M"),
            "a": [0.5] * 12,
            "b": [0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46],
            "c": [0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04],
        },
    ).set_index("date")

    forecasts = model().fit(time_series).predict(10)
    for i in range(1, time_series.shape[1]):
        colnames = list(time_series.columns)
        colnames[0], colnames[i] = colnames[i], colnames[0]
        forecaster = model().fit(time_series[colnames])
        assert np.allclose(forecasts, forecaster.predict(10)[time_series.columns], atol=1e-2)


@pytest.mark.parametrize("model", [(LocalLevelForecaster), (LocalTrendForecaster)])
@given(time_series=compositional_ts())
def test_log_ratio_raises_error_when_all_columns_have_nan(
    model: type[Union[LocalLevelForecaster, LocalTrendForecaster]],
    time_series: pd.DataFrame,
):
    """Test that a ``LogRatioTransformError`` is raised when all time series have nans."""
    time_series.iloc[0, :] = np.nan

    error_msg = (
        "It is not possible to apply the log-ratio transform on the given time series. At "
        "least one of them should not contain any missing values."
    )
    with pytest.raises(LogRatioTransformError, match=error_msg):
        model().fit(time_series)


@pytest.mark.parametrize("model", [(LocalLevelForecaster), (LocalTrendForecaster)])
def test_freq_inference_error(model: type[Union[LocalLevelForecaster, LocalTrendForecaster]]):
    """Test that a ``FreqInferenceError`` is raised when the index frequency cannot be inferred."""
    index = pd.DatetimeIndex([pd.to_datetime("2020-01-01"), pd.to_datetime("2020-01-02")])
    time_series = pd.DataFrame({"t1": [0.5, 0.3], "t2": [0.5, 0.7]}, index)

    error_msg = "Cannot infer the frequency of the given time series"
    with pytest.raises(FreqInferenceError, match=error_msg):
        model().fit(time_series)


@pytest.mark.parametrize("model", [(LocalLevelForecaster), (LocalTrendForecaster)])
@given(time_series=compositional_ts(shape=(10, 3)))
def test_invalid_index_error(
    model: type[Union[LocalLevelForecaster, LocalTrendForecaster]],
    time_series: pd.DataFrame,
):
    """Test that an ``InvalidIndexError`` is raised when the index values are not equally spaced."""
    time_series = time_series.drop(index=time_series.index[1])

    error_msg = "The index of the time series should have equally spaced values"
    with pytest.raises(InvalidIndexError, match=error_msg):
        model().fit(time_series)
