from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Self, Sequence
from scipy import optimize
from scipy.optimize import Bounds, LinearConstraint
from scipy import linalg


@dataclass(frozen=True)
class LocalLevelParams:
    X_zero: np.ndarray
    alpha: np.ndarray

    @classmethod
    def init(cls, num_series: int) -> Self:
        X_zero = np.random.uniform(-4, 1, (1, num_series))
        alpha = np.random.uniform(0, 2, 1)
        return cls(X_zero, alpha)

    @property
    def bounds(self) -> Bounds:
        lower, upper = zip(*([(-np.inf, np.inf)] * self.X_zero.size + [(0.0, 2.0)]))
        return Bounds(lower, upper)


@dataclass(frozen=True)
class LocalTrendParams:
    X_zero: np.ndarray
    g: np.ndarray

    @classmethod
    def init(cls, num_series: int) -> Self:
        X_zero = np.random.uniform(-4, 1, num_series * 2)[:, None]
        g = np.random.uniform(0, 1, 1 * 2)[:, None]
        return cls(X_zero, g)

    @property
    def bounds(self) -> Bounds:
        lower, upper = zip(*([(-np.inf, np.inf)] * self.X_zero.size + [(0.0, np.inf)] * 2))
        return Bounds(lower, upper)

    @property
    def constraints(self) -> list[LinearConstraint]:
        constraint_matrix = linalg.block_diag(np.eye(self.X_zero.size), np.array([[2, 1], [0, 1]]))
        ub = np.array([np.inf] * self.X_zero.size + [4.0] + [np.inf])
        return [LinearConstraint(constraint_matrix, ub=ub)]


class LocalTrendForecaster:
    optim_params_: LocalTrendParams
    X_: np.ndarray
    errors_: np.ndarray
    colnames_: pd.Index
    time_idx_: pd.Index

    def fit(self, y: pd.DataFrame) -> Self:
        self.colnames_ = y.columns
        self.time_idx_ = y.index

        self.optim_params_ = _fit_local_level(_log_ratio(y.values))
        self.X_, self.errors_ = _forward(
            self.optim_params_.X_zero, self.optim_params_.alpha, y.values
        )
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        preds_idx = pd.date_range(
            self.time_idx_.max() + 1, periods=horizon, freq=self.time_idx_.freq
        )
        preds = pd.DataFrame(
            _inv_log_ratio(_predict_local_trend(horizon, self.X_[-1, :])), preds_idx, self.colnames_
        )
        return preds


def _predict_local_trend(horizon: int, X_last: np.ndarray):
    F = np.tri(2)
    w = np.ones(2)

    preds = []
    for _ in range(horizon):
        y_hat = w @ X_last
        X_last = F @ X_last
        preds.append(y_hat)

    return preds


class LocalLevelForecaster:
    optim_params_: LocalLevelParams
    X_: np.ndarray
    errors_: np.ndarray
    colnames_: pd.Index
    time_idx_: pd.Index

    def fit(self, y: pd.DataFrame) -> Self:
        self.colnames_ = y.columns
        self.time_idx_ = y.index

        self.optim_params_ = _fit_local_level(_log_ratio(y.values))
        self.X_, self.errors_ = _forward(
            self.optim_params_.X_zero, self.optim_params_.alpha, y.values
        )
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        preds_idx = pd.date_range(
            self.time_idx_.max() + 1, periods=horizon, freq=self.time_idx_.freq
        )
        preds = pd.DataFrame(
            _inv_log_ratio(np.tile(self.X_[-1, :], (horizon, 1))), preds_idx, self.colnames_
        )
        return preds


def _log_ratio(array: np.ndarray) -> np.ndarray:
    return np.log(array[:, 1:] / array[:, :1])


def _inv_log_ratio(array: np.ndarray) -> np.ndarray:
    divisor = 1 + np.exp(array).sum(axis=1)
    array = np.exp(array) / divisor[:, None]
    return np.insert(array, 0, 1 - array.sum(axis=1), axis=1)


def _flatten_params(*params) -> tuple[np.ndarray, tuple[int]]:
    params, shapes = tuple(zip(*((np.ravel(x), x.shape) for x in params)))
    return np.concatenate(params), shapes


def _unflatten_params(
    flat_params: Sequence[np.ndarray], shapes: Sequence[int]
) -> tuple[np.ndarray]:
    cutoffs = np.cumsum([np.product(shape) for shape in shapes], dtype=int)

    params = []
    prev_cutoff = 0
    for cutoff, shape in zip(cutoffs, shapes):
        param = flat_params[prev_cutoff:cutoff].reshape(shape)
        prev_cutoff = cutoff
        params.append(param)

    return params


def _fit_local_trend(y: np.ndarray) -> LocalTrendParams:
    num_series = y.shape[1]
    params = LocalTrendParams.init(num_series)
    flat_params, shapes = _flatten_params(params)

    opt_params = optimize.minimize(
        _objective,
        flat_params,
        (shapes, y),
        method="turst-constr",
        bounds=params.bounds,
        constraints=params.constraints,
    ).x

    return LocalTrendParams(*opt_params)


def _fit_local_level(y: np.ndarray) -> LocalLevelParams:
    """Find the optimal parameters of a local level model for the given data."""
    num_series = y.shape[1]
    params = LocalLevelParams.init(num_series)
    flat_params, shapes = _flatten_params(params)

    opt_params = optimize.minimize(
        _objective, flat_params, (shapes, y), method="trust-constr", bounds=params.bounds
    ).x

    return LocalLevelParams(*opt_params)


def _objective(flat_params: np.ndarray, shapes: tuple[int], y: np.ndarray) -> float:
    X_zero, g = _unflatten_params(flat_params, shapes)
    return _neg_log_likelihood(X_zero, g, y)


def _neg_log_likelihood(X_zero: np.ndarray, g: np.ndarray, y: np.ndarray) -> float:
    n, r = y.shape
    return n * r / 2 * np.log(2 * np.pi) + n / 2 * np.log(_mle_V(X_zero, g, y)) + n * r / 2


def _mle_V(X_zero: np.ndarray, g: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    _, errors = _forward(X_zero, g, y)
    return sum(error @ error for error in errors) / n


def _forward(X_zero: np.ndarray, g: np.ndarray, y: np.ndarray) -> np.ndarray:
    if X_zero.ndim == 1:
        n_rows = 1
    else:
        n_rows = len(X_zero)

    w = np.ones(n_rows)
    F = np.tri(n_rows)

    errors = []
    X_vals = [X_zero]
    X_prev = X_zero
    for y_t in y:
        error = y_t - w @ X_prev
        X_prev = F @ X_prev + g @ error.reshape(1, -1)

        errors.append(error)
        X_vals.append(X_prev)

    return np.vstack(X_vals), np.vstack(errors)
