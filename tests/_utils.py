"""Utils for tests."""
from typing import Optional, Union

import numpy as np
import pandas as pd
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import DrawFn


@st.composite
def compositional_ts_array(
    draw: DrawFn,
    shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Strategy to generate a numpy array of compositional time series."""
    if not shape:
        shape = hnp.array_shapes(min_dims=2, max_dims=2, min_side=2)

    array = draw(
        hnp.arrays(
            np.dtype(float),
            shape=shape,
            elements=hnp.from_dtype(
                np.dtype(float),
                min_value=0.0,
                max_value=1e20,
                allow_nan=False,
            ),
        ),
    )
    # Add random jitter
    array = np.abs(array + np.random.default_rng(0).standard_normal(array.shape))
    return array / array.sum(axis=1)[:, None]


@st.composite
def compositional_ts(
    draw: DrawFn,
    index_type: type[Union[pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex]] = pd.DatetimeIndex,
    shape: Optional[tuple[int, int]] = None,
) -> pd.DataFrame:
    """Strategy to generate a dataframe of compositional time series."""
    array = draw(compositional_ts_array(shape=shape))

    if index_type == pd.DatetimeIndex:
        ts_index = pd.date_range("2020-01-01", periods=len(array), freq="D")
    elif index_type == pd.PeriodIndex:
        ts_index = pd.period_range("2020-01", periods=len(array), freq="M")
    elif index_type == pd.RangeIndex:
        ts_index = pd.RangeIndex.from_range(range(len(array)))
    else:
        ts_index = pd.Index(list(range(len(array))))

    ts_index.name = "date"

    return pd.DataFrame(array, index=ts_index)
