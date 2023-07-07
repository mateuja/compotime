"""Preprocess funcions for compositional time series."""
import numpy as np
import pandas as pd


def treat_small(y: pd.DataFrame, minimum: float) -> pd.DataFrame:
    """Adjust the compositional time series so that no value is smaller than ``minimum``.

    Parameters
    ----------
        y: Time series data, where each column represents a particular time series.
        minimum: Minimum value to appear in the time series.

    Returns
    -------
       Adjusted compositional time series.

    Notes
    -----
    TODO!
    
    References
    ----------

    .. [*] Snyder, R.D. et al. 2017
       Forecasting compositional time series: A state space approach
       International Journal of Forecasting.
    """
    y = y.copy()
    for idx, y_t in y.iterrows():
        is_below_min = y_t < minimum
        n_below_min = is_below_min.sum()
        if n_below_min == 0:
            continue

        y.loc[idx] = np.where(
            is_below_min,
            minimum,
            (1 - minimum * n_below_min) * y_t / y_t[~is_below_min].sum(),
        )
    return y

