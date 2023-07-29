"""Compositional time series state-space models.

References
----------
.. [*] Hyndman, R.J., Koehler, A.B., Ord, J.K. & Snyder, R.D. 2008.
    Forecasting with exponential smoothing.
    Berlin: Springer.

.. [*] Snyder, R.D. et al. 2017
    Forecasting compositional time series: A state space approach
    International Journal of Forecasting.

.. [*] Olive, D. 2023
    Prediction and Statistical Learning.
"""
import abc
import logging
from abc import ABC
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats
from scipy.optimize import Bounds, LinearConstraint
from typing_extensions import Self

from .errors import FreqInferenceError, InvalidIndexError, LogRatioTransformError

INITIAL_ALPHA = 0.1
INITIAL_BETA = 0.01

ALPHA_BOUNDS = (0.0, 2.0)

CONDITION_NUMBER = 50


class Params(ABC):
    """Parameters abstract class."""

    @classmethod
    @abc.abstractmethod
    def init(cls, time_series: np.ndarray) -> Self:
        """Initialize parameters based on the observed time series.

        Parameters
        ----------
        time_series
            Observed time series.

        Returns
        -------
        Self
            Initialized parameters.
        """

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over the different parameters.

        Yields
        ------
        Iterator[np.ndarray]
            Parameter.
        """
        yield from vars(self).values()


@dataclass(frozen=True)
class LocalLevelParams(Params):
    """Parameters for the local level model.

    Notes
    -----
    The seed state matrix (`X_zero`) repesents the value for the transition equation that describes
    how the state vectors evolve over time. The persistence vector (`g`) determines the extend of
    the innovation on the state. These are the only parameters that need to be estimated.

    Parameters
    ----------
    X_zero
        Seed state matrix.
    g
        Persistence vector.

    Attributes
    ----------
    X_zero
        Seed state matrix.
    g
        Persistence vector.
    """

    X_zero: np.ndarray
    g: np.ndarray

    @classmethod
    def init(cls, time_series: np.ndarray) -> Self:
        """Initialize parameters.

        Parameters
        ----------
        time_series
            Observed time series.

        Returns
        -------
        Self
            Initialized parameters.
        """
        X_zero = _initialize_X_zero(time_series, no_trend=True)
        g = np.array([INITIAL_ALPHA])
        return cls(X_zero, g)

    @property
    def bounds(self) -> Bounds:
        """Get the bounds for the parameters of the local level model.

        Notes
        -----
        In the local level model, `g` values must be within the range between 0 and 2, both
        included.

        Returns
        -------
        Bounds
            Bounds for the parameters of the local level model.
        """
        lower, upper = zip(*([(-np.inf, np.inf)] * self.X_zero.size + [ALPHA_BOUNDS]))
        return Bounds(lower, upper)


@dataclass(frozen=True)
class LocalTrendParams(Params):
    """Parameters for the local level model.

    Notes
    -----
    The seed state matrix (`X_zero`) repesents the value for the transition equation that describes
    how the state vectors evolve over time. The persistence vector (`g`) determines the extend of
    the innovation on the state. These are the only parameters that need to be estimated.

    Parameters
    ----------
    X_zero
        Seed state matrix.
    g
        Persistence vector.

    Attributes
    ----------
    X_zero
        Seed state matrix.
    g
        Persistence vector.
    """

    X_zero: np.ndarray
    g: np.ndarray

    @classmethod
    def init(cls, time_series: np.ndarray) -> Self:
        """Initialize parameters.

        Parameters
        ----------
        time_series
            Observed time series.

        Returns
        -------
        Self
            Initialized parameters.
        """
        X_zero = _initialize_X_zero(time_series, no_trend=False)
        g = np.array([[INITIAL_ALPHA], [INITIAL_BETA]])
        return cls(X_zero, g)

    @property
    def constraints(self) -> list[LinearConstraint]:
        r"""Get the linear constraints for the parameters of the local trend model.

        Notes
        -----
        In the local trend model, `g` can be decomposed into :math:`\alpha` and :math:`\beta`
        parameters, which must be greater than or equal to zero and satisfy the following
        linear constraint:

        .. math::
            2 \alpha + \beta \le 4.

        Returns
        -------
        list[LinearConstraint]
            Linear constraints for the parameters of the local trend model.

        """
        constraint_matrix = linalg.block_diag(np.eye(self.X_zero.size), np.array([[2, 1], [0, 1]]))
        lb = np.array([-1e12] * self.X_zero.size + [0.0] * 2)
        ub = np.array([1e12] * self.X_zero.size + [4.0] * 2)
        return [LinearConstraint(constraint_matrix, lb=lb, ub=ub)]


class LocalLevelForecaster:
    r"""Local level state-space forecaster.

    Notes
    -----
    The local level model is described by the following equations:

    .. math::

        \boldsymbol y_t &= \boldsymbol l_{t-1} +        \boldsymbol \epsilon_t \\
        \boldsymbol l_t &= \boldsymbol l_{t-1} + \alpha \boldsymbol \epsilon_t

    where :math:`\boldsymbol y_t` represents the unbounded time series observations at timestep
    :math:`t` that result from applying the log-ratio transform, and :math:`\boldsymbol l_t`
    represents the local level.

    Equivalently, to express it in the same terms as the ``LocalTrendForecaster``, it is possible to
    use

    .. math::

        \boldsymbol y_t' &= \boldsymbol w \boldsymbol x_{t-1} + \boldsymbol \epsilon_t' \\
        \boldsymbol x_t  &= \boldsymbol F \boldsymbol x_{t-1}  + \boldsymbol g \boldsymbol
            \epsilon_t'

    where

    :math:`\boldsymbol x_t = \boldsymbol l^{'}_{t}`,
    :math:`\boldsymbol w = 1`,
    :math:`\boldsymbol F = 1`, and
    :math:`\boldsymbol g = \alpha`.
    """

    optim_params_: LocalLevelParams
    X_: list[np.ndarray]
    fitted_curve_: pd.DataFrame
    colnames_: pd.Index
    time_idx_: pd.Index
    idx_freq_: Optional[str]
    base_col_idx_: int

    def fit(self, y: pd.DataFrame) -> Self:
        """Fit the model.

        Parameters
        ----------
        y
            Time series dataframe, where rows represent the timestamps and columns the different
            shares series.

        Returns
        -------
        Self
            Fitted instance of the model.
        """
        self.colnames_ = y.columns
        self.time_idx_ = y.index

        _validate_idx(self.time_idx_)

        self.idx_freq_ = _get_idx_freq(self.time_idx_)

        log_y, self.base_col_idx_ = _log_ratio(y.values)

        self.optim_params_ = _fit_local_level(log_y)
        self.X_, fitted_curve, _ = _forward(
            self.optim_params_.X_zero,
            self.optim_params_.g,
            log_y,
        )

        self.fitted_curve_ = pd.DataFrame(
            _inv_log_ratio(fitted_curve, self.base_col_idx_),
            y.index,
            y.columns,
        )

        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Predict future values of the time series.

        Parameters
        ----------
        horizon
            Number of steps into the future to be predicted.

        Returns
        -------
        pd.DataFrame
            Predicted time series.
        """
        return pd.DataFrame(
            _inv_log_ratio(np.tile(self.X_[-1], (horizon, 1)), self.base_col_idx_),
            _get_preds_idx(horizon, self.time_idx_, self.idx_freq_),
            self.colnames_,
        )


class LocalTrendForecaster:
    r"""Local trend state-space forecaster.

    Notes
    -----
    The local trend model is described by the following equations:

    .. math::

        \boldsymbol y_t &= \boldsymbol l_{t-1} + \boldsymbol b_{t-1} +        \boldsymbol
            \epsilon_t \\
        \boldsymbol l_t &= \boldsymbol l_{t-1} + \boldsymbol b_{t-1} + \alpha \boldsymbol
            \epsilon_t \\
        \boldsymbol b_t &= \boldsymbol b_{t-1} + \beta                        \boldsymbol
            \epsilon_t

    where :math:`\boldsymbol y_t` represents the unbounded time series observations at timestep
    :math:`t` that result from applying the log-ratio transform. :math:`\boldsymbol l_t` and
    :math:`\boldsymbol b_t` represent the level and the trend, respectively.

    An equivalent expression is as follows

    .. math::
        \boldsymbol y_t' &= \boldsymbol w' \boldsymbol X_{t-1} + \boldsymbol \epsilon_t' \\
        \boldsymbol X_t  &= \boldsymbol F \boldsymbol X_{t-1}  + \boldsymbol g \boldsymbol
            \epsilon_t'

    where

    :math:`\boldsymbol X_t = \begin{bmatrix} \boldsymbol l^{'}_{t} \\ \boldsymbol b^{'}_{t} \end{bmatrix}`,
    :math:`\boldsymbol w = \begin{bmatrix} 1 \\ 1 \end{bmatrix}`,
    :math:`\boldsymbol F = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}` and
    :math:`\boldsymbol g = \begin{bmatrix} \alpha \\ \beta \end{bmatrix}`.
    """  # noqa: E501

    optim_params_: LocalTrendParams
    X_: list[np.ndarray]
    fitted_curve_: pd.DataFrame
    colnames_: pd.Index
    time_idx_: pd.Index
    idx_freq_: Optional[str]
    base_col_idx_: int

    def fit(self, y: pd.DataFrame) -> Self:
        """Fit the model.

        Parameters
        ----------
        y
            Time series dataframe, where rows represent the timestamps and columns the different
            shares series.

        Returns
        -------
        Self
            Fitted instance of the model.
        """
        self.colnames_ = y.columns
        self.time_idx_ = y.index

        _validate_idx(self.time_idx_)

        self.idx_freq_ = _get_idx_freq(self.time_idx_)

        log_y, self.base_col_idx_ = _log_ratio(y.values)

        self.optim_params_ = _fit_local_trend(log_y)

        self.X_, fitted_curve, _ = _forward(self.optim_params_.X_zero, self.optim_params_.g, log_y)

        self.fitted_curve_ = pd.DataFrame(
            _inv_log_ratio(fitted_curve, self.base_col_idx_),
            y.index,
            y.columns,
        )

        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Predict future values of the time series.

        Parameters
        ----------
        horizon
            Number of steps into the future to be predicted.

        Returns
        -------
        pd.DataFrame
            Predicted time series.
        """
        return pd.DataFrame(
            _inv_log_ratio(_predict_local_trend(horizon, self.X_[-1]), self.base_col_idx_),
            _get_preds_idx(horizon, self.time_idx_, self.idx_freq_),
            self.colnames_,
        )


def _log_ratio(array: np.ndarray) -> tuple[np.ndarray, int]:
    """Apply log ratio transform to the given time series.

    Parameters
    ----------
    array
        Multivariate time series array, where the rows represent the different time steps and the
        columns represent each of the individual series.

    Returns
    -------
    tuple[np.ndarray, int]
        Unbounded time series and index of the base column in the original array.

    Raises
    ------
    LogRatioTransformError
        If there is no column without any missing values.
    """
    not_nan_cols = np.flatnonzero(~np.isnan(array).any(axis=0))
    if not_nan_cols.size == 0:
        raise LogRatioTransformError(
            "It is not possible to apply the log-ratio transform on the given time series. At "
            "least one of them should not contain any missing values.",
        )

    base_col = not_nan_cols[0]
    return np.log(np.delete(array, base_col, 1) / array[:, [base_col]]), base_col


def _inv_log_ratio(array: np.ndarray, base_col_idx: int) -> np.ndarray:
    """Apply the inverse function of the log ratio transform to the given time series.

    Parameters
    ----------
    array
        Multivariate time series array, where the rows represent the different time steps and
        the columns represent each of the individual series.
    base_col_idx
        Base column index.

    Returns
    -------
    np.ndarray
        Time series that add up to zero at each time subscript t.
    """
    divisor = 1 + np.exp(array).sum(axis=1)
    array = np.exp(array) / divisor[:, None]
    return np.insert(array, base_col_idx, 1 - array.sum(axis=1), axis=1)


def _flatten_params(params: Params) -> tuple[np.ndarray, tuple[int]]:
    """Flatten the given parameters into a single unidimensional array.

    Parameters
    ----------
    params
        Parameters.

    Returns
    -------
    tuple[np.ndarray, tuple[int]]
        Flattened parameters and their original shapes.
    """
    params, shapes = tuple(zip(*((np.ravel(x), x.shape) for x in params)))
    return np.concatenate(params), shapes


def _unflatten_params(
    flat_params: Sequence[np.ndarray],
    shapes: Sequence[int],
) -> tuple[np.ndarray]:
    """Reverse the transformation of the flattened parameters.

    Parameters
    ----------
    flat_params
        Single unidimensional array with the values for all the parameters.
    shapes
        Shapes that the output parameters should have.

    Returns
    -------
    tuple[np.ndarray]
        Multiple parameters with various shapes.
    """
    cutoffs = np.cumsum([np.prod(shape) for shape in shapes], dtype=int)

    params = []
    prev_cutoff = 0
    for cutoff, shape in zip(cutoffs, shapes):
        param = flat_params[prev_cutoff:cutoff].reshape(shape)
        prev_cutoff = cutoff
        params.append(param)

    return params


def _fit_local_level(y: np.ndarray) -> LocalLevelParams:
    """Find the optimal parameters of a local level model for the given data.

    Parameters
    ----------
    y
        Time series data.

    Returns
    -------
    LocalLevelParams
        Optimized parameters for the observed data.
    """
    initial_params = LocalLevelParams.init(y)
    flat_params, shapes = _flatten_params(initial_params)

    opt_res = optimize.minimize(
        _objective,
        flat_params,
        (shapes, y),
        bounds=initial_params.bounds,
    )

    if not opt_res.success:
        logging.warning("Optimization finished unsuccessfully: %s", opt_res.message)

    opt_params = _unflatten_params(opt_res.x, shapes)

    return LocalLevelParams(*opt_params)


def _fit_local_trend(y: np.ndarray) -> LocalTrendParams:
    """Find the optimal parameters of a local trend model for the given data.

    Parameters
    ----------
    y
        Time series data.

    Returns
    -------
    LocalTrendParams
        Optimized parameters for the observed data.
    """
    initial_params = LocalTrendParams.init(y)
    flat_params, shapes = _flatten_params(initial_params)

    opt_res = optimize.minimize(
        _objective,
        flat_params,
        (shapes, y),
        constraints=initial_params.constraints,
        options={"maxiter": 200},
    )

    if not opt_res.success:
        logging.warning("Optimization finished unsuccessfully: %s", opt_res.message)

    opt_params = _unflatten_params(opt_res.x, shapes)

    return LocalTrendParams(*opt_params)


def _initialize_X_zero(y: np.ndarray, no_trend: bool) -> np.ndarray:  # noqa: FBT001, N802
    """Initialize the seed state matrix.

    For each of the time series, a linear regression is fitted with the first available ten points
    (less if the series are shorter). The level is set to be equal to the intercept, and the trend,
    if required, is set to be equal to the slope.

    Parameters
    ----------
    y
        Observed time series.
    no_trend
        Whether no trend is required. This should be ``True`` for the
        ``LocalLevelForecaster``.

    Returns
    -------
    np.ndarray
        Initialized seed state matrix.
    """
    regressions = []
    for col in y.T:
        no_nan_col = col[~np.isnan(col)][:10]
        regressions.append(stats.linregress(range(len(no_nan_col)), no_nan_col))

    intercepts = np.array([reg.intercept for reg in regressions])
    if no_trend:
        return intercepts.reshape(1, -1)

    slopes = np.array([reg.slope for reg in regressions])
    return np.vstack([intercepts, slopes])


def _predict_local_trend(horizon: int, X_last: np.ndarray) -> np.ndarray:
    """Predict future values for a time series using the local trend model.

    Parameters
    ----------
    horizon
        Number of steps to predict into the future.
    X_last
        Value of the lattent state corresponding to the last observed observation in the time
        series.

    Returns
    -------
    np.ndarray
        Future values for the time series.
    """
    F = np.tri(2).T
    w = np.ones(2)

    preds = []
    for _ in range(horizon):
        y_hat = w @ X_last
        X_last = F @ X_last
        preds.append(y_hat)

    return np.vstack(preds)


def _objective(flat_params: np.ndarray, shapes: tuple[int], y: np.ndarray) -> float:
    """Objective function to be optimized by the local level and local trend models.

    Parameters
    ----------
    flat_params
        Initial parameters, flattened in a single one dimensional array.
    shapes
        Shapes of the different parameters, so that each of them can be reconstructed
        from the ``flat_params``.
    y
        Observations of the time series to be forecasted.

    Returns
    -------
    float
        Objective function to minimize the negative loglikelihood of the given parameters.
    """
    X_zero, g = _unflatten_params(flat_params, shapes)
    return _neg_log_likelihood(X_zero, g, y)


def _neg_log_likelihood(X_zero: np.ndarray, g: np.ndarray, y: np.ndarray) -> float:
    """Compute the negative log-likelihood loss.

    Parameters
    ----------
    X_zero
        Seed state matrix.
    g
        Persistence vector.
    y
        Observed time series.

    Returns
    -------
    float
        Negative log-likelihood loss.
    """
    n = len(y)

    _, _, errors = _forward(X_zero, g, y)

    if np.isnan(y).any():
        return _adj_neg_log_likelihood(y, errors)
    return np.linalg.slogdet(
        _regularize(np.matmul(errors.T, errors) / n),
    )[1]


def _adj_neg_log_likelihood(y: np.ndarray, errors: np.ndarray) -> float:
    """Compute the negative log-likelihood loss adapted to time series of different lengths.

    Having different lengths in the time series requires some adaptations when computing the
    estimated generalized variance.

    Parameters
    ----------
    y
        Observed time series.
    errors
        Array of errors per timestamp.

    Returns
    -------
    float
        Negative log-likelihood loss adapted to time series of different lengths.
    """
    num_not_nan = (~np.isnan(y)).sum(axis=0)
    covar = (errors.T @ errors) / np.minimum(num_not_nan[:, np.newaxis], num_not_nan)

    adj_gen_var = 0
    for y_t in y:
        selection = _compute_selection_matrix(y_t)
        adjusted_covar = _regularize(selection @ covar @ selection.T)
        _, gen_var_log_t = np.linalg.slogdet(adjusted_covar)
        adj_gen_var += gen_var_log_t

    return adj_gen_var


def _regularize(covar: np.ndarray) -> np.ndarray:
    """Regularize the covariance matrix.

    Add a delta parameter to the diagonal of the covariance matrix to ensure that it is
    non-singular and well conditioned.

    Parameters
    ----------
    covar
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Regularized covariance matrix.
    """
    eigenvals = np.linalg.eigvals(covar)
    delta = max(0, (max(eigenvals) - CONDITION_NUMBER * min(eigenvals)) / (CONDITION_NUMBER - 1))
    return covar + np.diag(np.repeat(delta, len(covar)))


def _compute_selection_matrix(y_t: np.ndarray) -> np.ndarray:
    """Compute matrix that selects the series that are observed (different than 0) at time t.

    Parameters
    ----------
    y_t
        Values of each time series at time t.

    Returns
    -------
    np.ndarray
        Selection matrix.
    """
    selection = np.zeros((np.count_nonzero(~np.isnan(y_t)), len(y_t)))

    i = 0
    for j in range(len(y_t)):
        if not np.isnan(y_t[j]):
            selection[i, j] = 1
            i += 1

    return selection


def _forward(X_zero: np.ndarray, g: np.ndarray, y: np.ndarray) -> tuple:
    """Compute the values of X_t and e_t that emerge a the different t steps.

    These values can be computed recursively from of `X_zero`, `g` and `y`.

    Parameters
    ----------
    X_zero
        Seed state matrix.
    g
        Persistence vector.
    y
        Observed time series.

    Returns
    -------
    tuple
        Latent states, fitted curve and errors for the different time steps.
    """
    n_rows = 1 if X_zero.ndim == 1 else len(X_zero)

    w = np.ones(n_rows)
    F = np.tri(n_rows).T

    latent_states = []
    fitted_curve = []
    errors = []
    X_prev = X_zero
    latent_states.append(X_zero)
    for y_t in y:
        fitted = w @ X_prev
        error = y_t - fitted
        error[np.isnan(error)] = 0.0
        X_prev = F @ X_prev + g @ error.reshape(1, -1)

        errors.append(error)
        latent_states.append(X_prev)
        fitted_curve.append(fitted)

    return latent_states, np.vstack(fitted_curve), np.vstack(errors)


def _validate_idx(idx: pd.Index) -> None:
    """Validate that the time series index is valid by checking that all values are equally spaced.

    Parameter
    --------
        idx: Time series index.

    Returns
    -------
    bool
        Whether the time series index is valid or not.

    Raises
    ------
    InvalidIndexError
        If the index is not valid.
    """
    if not _is_equally_spaced(idx):
        raise InvalidIndexError


def _get_idx_freq(idx: Union[pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex]) -> Optional[str]:
    """Get the frequency of the index, if it has any.

    Parameters
    ----------
    idx
        Index of the time series.

    Returns
    -------
    Optional[str]
        Frequency of the time series.

    Raises
    ------
    FreqInferenceError
        If the index is a PeriodIndex or DatetimeIndex but the frequency cannot be
        inferred.
    """
    if isinstance(idx, (pd.PeriodIndex, pd.DatetimeIndex)):
        if idx.freq:
            idx_freq = idx.freq
        else:
            try:
                idx_freq = pd.infer_freq(idx)
            except (ValueError, TypeError) as exc:
                raise FreqInferenceError from exc
    else:
        idx_freq = None

    return idx_freq


def _is_equally_spaced(idx: pd.Index) -> bool:
    """Check if all the values in a pandas index are equally spaced.

    Parameters
    ----------
        idx: Index of the dataframe.

    Returns
    -------
    bool
        Whether the index is equally spaced or not.
    """
    return len(set(np.diff(idx))) == 1


def _get_preds_idx(
    horizon: int,
    time_idx: Union[pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex],
    freq: Optional[str],
) -> pd.Index:
    """Get the index for the predictions.

    Parameters
    ----------
    horizon
        Number of steps to be predicted into the future.
    time_idx
        Index of the fitted time series.
    freq
        Frequency of the fitted time series index.

    Returns
    -------
    pd.Index
        Index of the predictions.
    """
    if isinstance(time_idx, pd.RangeIndex):
        preds_idx = pd.RangeIndex.from_range(range(horizon)) + len(time_idx)

    elif isinstance(time_idx, pd.PeriodIndex):
        preds_idx = pd.period_range(
            time_idx.max() + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )

    elif isinstance(time_idx, pd.DatetimeIndex):
        preds_idx = pd.date_range(
            time_idx.max() + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )

    else:
        step_size = np.diff(time_idx)[0]
        preds_idx = pd.Index([time_idx.max() + step_size * i for i in range(1, horizon + 1)])

    preds_idx.name = time_idx.name

    return preds_idx
