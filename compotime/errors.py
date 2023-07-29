"""Errors for compositional time series models."""

_INVALID_TIME_SERIES_ERROR = "It is not possible to apply the model on this type of time series"

_LOG_RATIO_TRANSFORM_ERROR = (
    "It is not possible to apply the log-ratio transform to the given time series"
)

_INVALID_INDEX_ERROR = "The index of the time series should have equally spaced values"

_FREQ_INF_ERROR = "Cannot infer the frequency of the given time series"


class InvalidTimeSeriesError(Exception):
    """Error raised when the given time series are not valid."""

    def __init__(self, msg: str = _INVALID_TIME_SERIES_ERROR) -> None:
        super().__init__(msg)


class LogRatioTransformError(InvalidTimeSeriesError):
    """Error raised when it is not possible to apply the log-ratio transform."""

    def __init__(self, msg: str = _LOG_RATIO_TRANSFORM_ERROR) -> None:
        super().__init__(msg)


class InvalidIndexError(InvalidTimeSeriesError):
    """Error raised when the index of the time series is not valid."""

    def __init__(self, msg: str = _INVALID_INDEX_ERROR) -> None:
        super().__init__(msg)


class FreqInferenceError(InvalidIndexError):
    """Error raised when it is not possible to infer the frequency of the time series index."""

    def __init__(self, msg: str = _FREQ_INF_ERROR) -> None:
        super().__init__(msg)
