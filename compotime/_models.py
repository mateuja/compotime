from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Self, Sequence
from scipy import optimize
from scipy.optimize import Bounds, LinearConstraint
from scipy import linalg

class Params:
    def __iter__(self):
        yield from vars(self).values()


@dataclass(frozen=True)
class LocalLevelParams(Params):
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
class LocalTrendParams(Params):
    X_zero: np.ndarray
    g: np.ndarray

    @classmethod
    def init(cls, num_series: int) -> Self:
        X_zero = np.random.uniform(-4, 1, (2, num_series))
        g = np.random.uniform(0, 1, (2, 1))
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

class LocalLevelForecaster:
    optim_params_: LocalLevelParams
    X_: list[np.ndarray]
    fitted_curve_: pd.DataFrame
    colnames_: pd.Index
    time_idx_: pd.Index

    def fit(self, y: pd.DataFrame) -> Self:
        self.colnames_ = y.columns
        self.time_idx_ = y.index
        
        log_y = _log_ratio(y.values)
        
        self.optim_params_ = _fit_local_level(log_y)
        self.X_, fitted_curve, _ = _forward(
            self.optim_params_.X_zero, self.optim_params_.alpha, log_y
        )

        self.fitted_curve_ = pd.DataFrame(_inv_log_ratio(fitted_curve), y.index, y.columns)

        return self

    
    def predict(self, horizon: int) -> pd.DataFrame:
        if isinstance(self.time_idx_, pd.PeriodIndex):
            date_range = pd.period_range
        else:
            date_range = pd.date_range

        freq = self.time_idx_.inferred_freq
        preds_idx = date_range(
            self.time_idx_.max() + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq
        )
        preds = pd.DataFrame(
            _inv_log_ratio(np.tile(self.X_[-1], (horizon, 1))), preds_idx, self.colnames_
        )
        return preds

class LocalTrendForecaster:
    optim_params_: LocalTrendParams
    X_: list[np.ndarray]
    fitted_curve_: pd.DataFrame
    colnames_: pd.Index
    time_idx_: pd.Index

    def fit(self, y: pd.DataFrame) -> Self:
        self.colnames_ = y.columns
        self.time_idx_ = y.index
        
        log_y = _log_ratio(y.values)
        
        self.optim_params_ = _fit_local_trend(log_y)

        self.X_, fitted_curve, _ = _forward(
            self.optim_params_.X_zero, self.optim_params_.g, log_y
        )

        self.fitted_curve_ = pd.DataFrame(_inv_log_ratio(fitted_curve), y.index, y.columns)

        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if isinstance(self.time_idx_, pd.PeriodIndex):
            date_range = pd.period_range
        else:
            date_range = pd.date_range

        freq = self.time_idx_.inferred_freq
        preds_idx = date_range(
            self.time_idx_.max() + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq
        )
        preds = pd.DataFrame(
            _inv_log_ratio(_predict_local_trend(horizon, self.X_[-1])), preds_idx, self.colnames_
        )
        return preds


def _predict_local_trend(horizon: int, X_last: np.ndarray):
    F = np.tri(2).T
    w = np.ones(2)

    preds = []
    for _ in range(horizon):
        y_hat = w @ X_last
        X_last = F @ X_last
        preds.append(y_hat)

    return np.vstack(preds)




def _log_ratio(array: np.ndarray) -> np.ndarray:
    return np.log(array[:, 1:] / array[:, :1])


def _inv_log_ratio(array: np.ndarray) -> np.ndarray:
    divisor = 1 + np.exp(array).sum(axis=1)
    array = np.exp(array) / divisor[:, None]
    return np.insert(array, 0, 1 - array.sum(axis=1), axis=1)


def _flatten_params(params) -> tuple[np.ndarray, tuple[int]]:
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
        method="trust-constr",
        bounds=params.bounds,
        constraints=params.constraints,
    ).x

    opt_params = _unflatten_params(opt_params, shapes)  

    return LocalTrendParams(*opt_params)


def _fit_local_level(y: np.ndarray) -> LocalLevelParams:
    """Find the optimal parameters of a local level model for the given data."""
    num_series = y.shape[1]
    params = LocalLevelParams.init(num_series)
    flat_params, shapes = _flatten_params(params)

    opt_params = optimize.minimize(
        _objective, flat_params, (shapes, y), method="trust-constr", bounds=params.bounds
    ).x

    opt_params = _unflatten_params(opt_params, shapes)  

    return LocalLevelParams(*opt_params)


def _objective(flat_params: np.ndarray, shapes: tuple[int], y: np.ndarray) -> float:
    X_zero, g = _unflatten_params(flat_params, shapes)
    return _neg_log_likelihood(X_zero, g, y)


def _neg_log_likelihood(X_zero: np.ndarray, g: np.ndarray, y: np.ndarray) -> float:
    n, r = y.shape
    return n * r / 2 * np.log(2 * np.pi) + n / 2 * np.log(_mle_V(X_zero, g, y)) + n * r / 2


def _mle_V(X_zero: np.ndarray, g: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    _, _, errors = _forward(X_zero, g, y)
    return sum(error @ error for error in errors) / n


def _forward(X_zero: np.ndarray, g: np.ndarray, y: np.ndarray) -> tuple:
    if X_zero.ndim == 1:
        n_rows = 1
    else:
        n_rows = len(X_zero)

    w = np.ones(n_rows)
    F = np.tri(n_rows).T

    X_vals = []
    fitted_curve = []
    errors = []
    X_prev = X_zero
    X_vals.append(X_zero)
    for y_t in y:
        fitted = w @ X_prev
        error = y_t - fitted
        X_prev = F @ X_prev + g @ error.reshape(1, -1)

        errors.append(error)
        X_vals.append(X_prev)
        fitted_curve.append(fitted)

    return X_vals, np.vstack(fitted_curve), np.vstack(errors)
