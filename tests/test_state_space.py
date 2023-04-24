import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from hypothesis import given

from compotime import _models as models


@given(st.lists(hnp.from_dtype(np.dtype(float)), min_size=1))
def test_unflatten_is_inverse_of_flatten(params):
    flat_params, shapes = models._flatten_params(params)
    unflattened_params = models._unflatten_params(flat_params, shapes)

    for original, unflattened in zip(params, unflattened_params):
        assert np.array_equal(original, unflattened, equal_nan=True)


@given(
    hnp.arrays(
        np.dtype(float),
        shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=2),
        elements=hnp.from_dtype(np.dtype(float), min_value=0.1, max_value=1e20, allow_nan=False),
    ).map(lambda x: x / x.sum(axis=1)[:, None])
)
def test_inv_log_ratio_is_inverse_of_log_ratio(array):
    """Test that the ``inv_log_ratio`` function is an inverse of the ``log_ratio``function when the
    sum of the time series at a given timestamp always equals one."""
    assert np.allclose(array, models._inv_log_ratio(models._log_ratio(array)))
