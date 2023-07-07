# compotime

compotime is a library for forecasting compositional time series in Python. At the moment, it provides an implementation of the models described in the paper ["Forecasting compositional time series: A state space approach"](https://isidl.com/wp-content/uploads/2017/06/E4001-ISIDL.pdf) (Snyder, R.D. et al, 2017). It is constantly tested to be compatible with the most recent versions of the major machine learning and statistics libraries within the Python ecosystem.

## Documentation

## Basic usage

```python
from compotime import LocalTrendForecaster

time_series = pd.read_csv() # TODO: Read CSV from URL

model = LocalTrendForecaster()
model.fit(time_series)
model.predict(horizon=10)
```

Check the [examples]() section of the documentation for more.

