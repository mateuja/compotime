"""Compositional time series state-space models."""
import abc
from abc import ABC
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.random import Generator
from scipy import linalg, optimize
from scipy.optimize import Bounds, LinearConstraint
from typing_extensions import Self


class Params(ABC):
    """Parameters abstract class."""

    @classmethod
    @abc.abstractmethod
    def init(cls, num_series: int, rng: Generator) -> Self:
        """Initialize parameters.

        Parameters
        ----------
            num_series: Number of time series.
            rng: Random number generator.

        Returns
        -------
            Initialized parameters.
        """
        ...

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over the different parameters.

        Yields
        ------
            parameter.
        """
        yield from vars(self).values()


@dataclass(frozen=True)
class LocalLevelParams(Params):
    """Parameters for the local level model.

    Parameters
    ----------
        X_zero: Seed state matrix.
        g: Persistence vector.

    Notes
    -----
    The seed state matrix (`X_zero`) repesents the value for the transition equation that describes
    how the state vectors evolve over time. The persistence vector (`g`) determines the extend of
    the innovation on the state [1]_. As shown in [2]_, these are the only parameters that need to
    estimated.

    References
    ----------
    .. [1] Hyndman, R.J., Koehler, A.B., Ord, J.K. & Snyder, R.D. 2008.
       Forecasting with exponential smoothing.
       Berlin: Springer.

    .. [2] Snyder, R.D. et al. 2017
       Forecasting compositional time series: A state space approach
       International Journal of Forecasting.
    """

    X_zero: np.ndarray
    g: np.ndarray

    @classmethod
    def init(cls, num_series: int, rng: Generator) -> Self:
        """Initialize parameters.

        Parameters
        ----------
            num_series: Number of time series.
            rng: Random number generator.

        Returns
        -------
            Initialized parameters.
        """
        X_zero = rng.uniform(-4, 1, (1, num_series))
        g = rng.uniform(0, 2, 1)
        return cls(X_zero, g)

    @property
    def bounds(self) -> Bounds:
        """Get the bounds for the parameters of the local level model.

        Returns
        -------
            Bounds for the parameters of the local level model.

        Notes
        -----
        In the local level model, `g` values must be within the range between 0 and 2, both
        included [1]_.

        References
        ----------
        .. [1] Snyder, R.D. et al. 2017.
           Forecasting compositional time series: A state space approach
           International Journal of Forecasting.
        """
        lower, upper = zip(*([(-np.inf, np.inf)] * self.X_zero.size + [(0.0, 2.0)]))
        return Bounds(lower, upper)


@dataclass(frozen=True)
class LocalTrendParams(Params):
    """Parameters for the local level model.

    Parameters
    ----------
        X_zero: Seed state matrix.
        g: Persistence vector.


    Notes
    -----
    The seed state matrix (`X_zero`) repesents the value for the transition equation that describes
    how the state vectors evolve over time. The persistence vector (`g`) determines the extend of
    the innovation on the state [1]_. As shown in [2]_, these are the only parameters that need to
    estimated.

    References
    ----------
    .. [1] Hyndman, R.J., Koehler, A.B., Ord, J.K. & Snyder, R.D. 2008.
       Forecasting with exponential smoothing.
       Berlin: Springer.

    .. [2] Snyder, R.D. et al. 2017.
       Forecasting compositional time series: A state space approach
       International Journal of Forecasting.
    """

    X_zero: np.ndarray
    g: np.ndarray

    @classmethod
    def init(cls, num_series: int, rng: Generator) -> Self:
        """Initialize parameters.

        Parameters
        ----------
            num_series: Number of time series.
            rng: Random number generator.

        Returns
        -------
            Initialized parameters.
        """
        X_zero = rng.uniform(-4, 1, (2, num_series))
        g = rng.uniform(0, 1, (2, 1))
        return cls(X_zero, g)

    @property
    def bounds(self) -> Bounds:
        """Get the bounds for the parameters of the local trend model.

        Returns
        -------
            Bounds for the parameters of the local level model.


        Notes
        -----
        In the local trend model, `g` can be decomposed into :math:`alpha` and :math:`beta`
        parameters, which must be greater than or equal to zero (see [1]_).

        References
        ----------
        .. [1] Snyder, R.D. et al. 2017.
           Forecasting compositional time series: A state space approach
           International Journal of Forecasting.
        """
        lower, upper = zip(*([(-np.inf, np.inf)] * self.X_zero.size + [(0.0, np.inf)] * 2))
        return Bounds(lower, upper)

    @property
    def constraints(self) -> list[LinearConstraint]:
        r"""Get the linear constraints for the parameters of the local trend model.

        Returns
        -------
            Linear constraints for the parameters of the local trend model.


        Notes
        -----
        In the local trend model, `g` can be decomposed into :math:`alpha` and :math:`beta`
        parameters, which must be greater than or equal to zero and satisfy the following
        linear constraint:

        .. math::
            2 \alpha + \beta \\le 4

        (see [1]_).

        References
        ----------
        .. [1] Snyder, R.D. et al. 2017.
           Forecasting compositional time series: A state space approach
           International Journal of Forecasting.
        """
        constraint_matrix = linalg.block_diag(np.eye(self.X_zero.size), np.array([[2, 1], [0, 1]]))
        ub = np.array([np.inf] * self.X_zero.size + [4.0] + [np.inf])
        return [LinearConstraint(constraint_matrix, ub=ub)]


class LocalLevelForecaster:
    r"""Forecast using the local level state-space model.

    Notes
    -----
    The local level model is described by the following equations:
    .. math::
        y_t = x_{t-1} + \\epsilon_t
        l_t = x_{t-1} + g\\epsilon_t

    where :math:`y_t` represents the unbounded time series observations that result from applying
    the log-ratio transform [1]_.

    References
    ----------
    .. [1] Snyder, R.D. et al. 2017.
        Forecasting compositional time series: A state space approach
        International Journal of Forecasting.
    """

    optim_params_: LocalLevelParams
    X_: list[np.ndarray]
    fitted_curve_: pd.DataFrame
    colnames_: pd.Index
    time_idx_: pd.Index

    def fit(self, y: pd.DataFrame, random_state: int = 0, threshold: float = 1e-6) -> Self:
        """Fit the model.

        Parameters
        ----------
            y: Time series dataframe, where rows represent the timestamps and columns the
                different shares series.
            random_state: Random state to initialize the random generator.
            threshold: Minimum value that all time series values must have; set to
            `threshold`, otherwise.

        Returns
        -------
            Fitted instance of the model.
        """
        rng = np.random.default_rng(random_state)
        self.colnames_ = y.columns
        self.time_idx_ = y.index

        y = _treat_zeros(y, threshold)

        log_y = _log_ratio(y.values)

        self.optim_params_ = _fit_local_level(log_y, rng)
        self.X_, fitted_curve, _ = _forward(
            self.optim_params_.X_zero,
            self.optim_params_.g,
            log_y,
        )

        self.fitted_curve_ = pd.DataFrame(_inv_log_ratio(fitted_curve), y.index, y.columns)

        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Predict future values of the time series.

        Parameters
        ----------
            horizon: Number of steps into the future to be predicted.

        Returns
        -------
            Predicted time series.
        """
        # TODO: Improve handling of different types of indexes
        if isinstance(self.time_idx_, pd.PeriodIndex):
            date_range = pd.period_range
        else:
            date_range = pd.date_range

        freq = self.time_idx_.inferred_freq
        preds_idx = date_range(
            self.time_idx_.max() + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )
        return pd.DataFrame(
            _inv_log_ratio(np.tile(self.X_[-1], (horizon, 1))),
            preds_idx,
            self.colnames_,
        )


class LocalTrendForecaster:
    r"""Forecast using the local trend state-space model.

    Notes
    -----
    The local model is described by the following equations:
    .. math::
        y_t = x_{t-1} + \\epsilon_t
        l_t = x_{t-1} + g\\epsilon_t

    where :math:`y_t` represents the unbounded time series observations at timestep t that result
    from applying the log-ratio transform and :math:`x_t` can be decomposed into level and trend
    vectors so that

    .. math:
        x_t = \begin{bmatrix}
            l^{'}_{t} \\
            b^{'}_{t}
        \\end{bmatrix}


    References
    ----------
    .. [1] Snyder, R.D. et al. 2017.
        Forecasting compositional time series: A state space approach
        International Journal of Forecasting.
    """

    optim_params_: LocalTrendParams
    X_: list[np.ndarray]
    fitted_curve_: pd.DataFrame
    colnames_: pd.Index
    time_idx_: pd.Index

    def fit(self, y: pd.DataFrame, random_state: int = 0, threshold: float = 0.001) -> Self:
        """Fit the model.

        Parameters
        ----------
            y: Time series dataframe, where rows represent the timestamps and columns the
                different shares series.
            random_state: Random state to initialize the random generator.
            threshold: Minimum value that all time series values must have; set to
            `threshold`, otherwise.

        Returns
        -------
            Fitted instance of the model.
        """
        rng = np.random.default_rng(random_state)
        self.colnames_ = y.columns
        self.time_idx_ = y.index

        y = _treat_zeros(y, threshold)

        log_y = _log_ratio(y.values)

        self.optim_params_ = _fit_local_trend(log_y, rng)

        self.X_, fitted_curve, _ = _forward(self.optim_params_.X_zero, self.optim_params_.g, log_y)

        self.fitted_curve_ = pd.DataFrame(_inv_log_ratio(fitted_curve), y.index, y.columns)

        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Predict future values of the time series.

        Parameters
        ----------
            horizon: Number of steps into the future to be predicted.

        Returns
        -------
            Predicted time series.
        """
        # TODO: Improve handling of different types of indexes
        if isinstance(self.time_idx_, pd.PeriodIndex):
            date_range = pd.period_range
        else:
            date_range = pd.date_range

        freq = self.time_idx_.inferred_freq
        preds_idx = date_range(
            self.time_idx_.max() + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )
        return pd.DataFrame(
            _inv_log_ratio(_predict_local_trend(horizon, self.X_[-1])),
            preds_idx,
            self.colnames_,
        )


def _treat_zeros(table: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """Replace zeros by `thresh` and adjust series to add up to 1.

    Parameters
    ----------
        table: Table contianing the time series data.
        thresh: Value to convert zeros to.

    Returns
    -------
        Table with corrected time series.
    """
    # ruff: noqa: B023
    table = table.copy()
    m = (table < thresh).sum(axis=1)
    for idx, row in table.iterrows():
        mask = (row < thresh)
        m = mask.sum()
        S = row[~mask].sum()
        if not m:
            continue

        table.loc[idx] = row.mask(mask, thresh).where(mask, lambda x: (1 - thresh * m) * x / S)

    return table


def _log_ratio(array: np.ndarray) -> np.ndarray:
    """Apply log ratio transform to the given time series.

    Parameters
    ----------
    array: Multivariate time series array, where the rows represent the different time steps and
        the columns represent each of the individual series.


    Returns
    -------
    Unbounded time series.
    """
    return np.log(array[:, 1:] / array[:, :1])


def _inv_log_ratio(array: np.ndarray) -> np.ndarray:
    """Apply the inverse function of the log ratio transform to the given time series.

    Parameters
    ----------
        array: Multivariate time series array, where the rows represent the different time steps and
            the columns represent each of the individual series.


    Returns
    -------
        Time series that add up to zero at each time subscript t.
    """
    divisor = 1 + np.exp(array).sum(axis=1)
    array = np.exp(array) / divisor[:, None]
    return np.insert(array, 0, 1 - array.sum(axis=1), axis=1)


def _flatten_params(params: Params) -> tuple[np.ndarray, tuple[int]]:
    """Flatten the given parameters into a single unidimensional array.

    Parameters
    ----------
        params: Parameters.

    Returns
    -------
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
        flat_params: Single unidimensional array with the values for all the parameters.
        shapes: Shapes that the output parameters should have.


    Returns
    -------
        Multiple parameters with various shapes.
    """
    cutoffs = np.cumsum([np.product(shape) for shape in shapes], dtype=int)

    params = []
    prev_cutoff = 0
    for cutoff, shape in zip(cutoffs, shapes):
        param = flat_params[prev_cutoff:cutoff].reshape(shape)
        prev_cutoff = cutoff
        params.append(param)

    return params


def _fit_local_level(y: np.ndarray, rng: Generator) -> LocalLevelParams:
    """Find the optimal parameters of a local level model for the given data.

    Parameters
    ----------
        y: Time series data.
        rng: Random number generator.

    Returns
    -------
        Optimized parameters for the observed data.
    """
    num_series = y.shape[1]
    params = LocalLevelParams.init(num_series, rng)
    flat_params, shapes = _flatten_params(params)

    opt_params = optimize.minimize(
        _objective,
        flat_params,
        (shapes, y),
        method="trust-constr",
        bounds=params.bounds,
    ).x

    opt_params = _unflatten_params(opt_params, shapes)

    return LocalLevelParams(*opt_params)


def _fit_local_trend(y: np.ndarray, rng: Generator) -> LocalTrendParams:
    """Find the optimal parameters of a local trend model for the given data.

    Parameters
    ----------
        y: Time series data.
        rng: Random number generator.

    Returns
    -------
        Optimized parameters for the observed data.
    """
    num_series = y.shape[1]
    params = LocalTrendParams.init(num_series, rng)
    flat_params, shapes = _flatten_params(params)

    opt_params = optimize.minimize(
        _objective,
        flat_params,
        (shapes, y),
        method="trust-constr",
        bounds=params.bounds,
        constraints=params.constraints,
    ).x

    opt_params = _unflatten_params(opt_params, shapes)

    return LocalTrendParams(*opt_params)


def _predict_local_trend(horizon: int, X_last: np.ndarray) -> np.ndarray:
    """Predict future values for a time series using the local trend model.

    Parameters
    ----------
        horizon: Number of steps to predict into the future.
        X_last: Value of the lattent state corresponding to the last observed
            observation in the time series.


    Returns
    -------
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
        flat_params: Initial parameters, flattened in a single one dimensional array.
        shapes: Shapes of the different parameters, so that each of them can be reconstructed
            from the ``flat_params``.
        y: Observations of the time series to be forecasted.

    Returns
    -------
        Objective function to minimize the negative loglikelihood of the given parameters.
    """
    X_zero, g = _unflatten_params(flat_params, shapes)
    return _log_mle_gen_var(X_zero, g, y)


def _log_mle_gen_var(X_zero: np.ndarray, g: np.ndarray, y: np.ndarray) -> float:
    """Compute the logarithm of the maximum likelihood estimator for the generalized variance.

    Parameters
    ----------
        X_zero: Seed state matrix.
        g: Persistence vector.
        y: Observed time series.

    Returns
    -------
        Logarithm of the maximum likelihood estimator for the generalized variance.
    """
    n = len(y)
    _, _, errors = _forward(X_zero, g, y)

    if np.isnan(y).any():
        gen_var_log = _adj_log_mle_gen_var(y, errors)
    else:
        _, gen_var_log = np.linalg.slogdet(
            sum(error.reshape(-1, 1) @ error.reshape(1, -1) for error in errors) / n,
        )
    return gen_var_log


def _adj_log_mle_gen_var(y: np.ndarray, errors: np.ndarray) -> float:
    """Compute the logarithm of the adjusted MLE for the generalized variance.

    Compute the logarithm of the adjusted maximum likelihood estimator for the generalized
    variance for cases with time series of different lenghts (i.e. containing NaN values).

    Parameters
    ----------
        y: Observed time series.
        errors: Array of errors per timestamp.

    Returns
    -------
        Logarithm of the adjusted maximum likelihood estimator for the generalized variance
        for TS with NaN values.
    """
    num_nan = (~np.isnan(y)).sum(axis=0)
    V = np.zeros((y.shape[1], y.shape[1]))
    for i in range(y.shape[1]):
        for j in range(i, y.shape[1]):
            V[i, j] = (errors[:, i] * errors[:, j]).sum() / min(num_nan[i], num_nan[j])
            if i != j:
                V[j, i] = V[i, j]

    adj_gen_var = 0
    for y_t in y:
        D = _select_matrix(y_t)
        V_tilde = D @ V @ D.T
        _, gen_var_log_t = np.linalg.slogdet(V_tilde)
        adj_gen_var += gen_var_log_t

    return adj_gen_var


def _select_matrix(y_t: np.ndarray) -> np.ndarray:
    """Compute matrix that selects the series that are observed (different than 0) at time t.

    Parameters
    ----------
        y_t: Values of each time series at time t.

    Returns
    -------
        Selection matrix.
    """
    D = np.zeros((np.count_nonzero(~np.isnan(y_t)), len(y_t)))

    i = 0
    for j in range(len(y_t)):
        if not np.isnan(y_t[j]):
            D[i, j] = 1
            i += 1

    return D


def _forward(X_zero: np.ndarray, g: np.ndarray, y: np.ndarray) -> tuple:
    """Compute the values of X_t and e_t that emerge a the different t steps.

    These values can be computed recursively from of `X_zero`, `g` and `y`.

    Parameters
    ----------
        X_zero: Seed state matrix.
        g: Persistence vector.
        y: Observed time series.

    Returns
    -------
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
        error = (y_t - fitted)
        error[np.isnan(error)] = 0.
        X_prev = F @ X_prev + g @ error.reshape(1, -1)

        errors.append(error)
        latent_states.append(X_prev)
        fitted_curve.append(fitted)

    return latent_states, np.vstack(fitted_curve), np.vstack(errors)
