# compotime

compotime is a library for forecasting compositional time series in Python. At the moment, it provides an implementation of the models described in the paper ["Forecasting compositional time series: A state space approach"](https://isidl.com/wp-content/uploads/2017/06/E4001-ISIDL.pdf) (Snyder, R.D. et al, 2017). It is constantly tested to be compatible with the most recent versions of the major machine learning and statistics libraries within the Python ecosystem.


## Basic usage

This example uses adapted data from the evolution of
[disease burden by risk factor in the US (1990-2019)](https://ourworldindata.org/grapher/disease-burden-by-risk-factor?time=earliest..latest&country=~USA).

```python
import pandas as pd

from compotime import LocalTrendForecaster

URL = "https://raw.githubusercontent.com/mateuja/compotime/feature/configure_docs/examples/data/dbrf.csv"

time_series = pd.read_csv(URL, parse_dates=["Year"], index_col="Year")

model = LocalTrendForecaster()
model.fit(time_series)
model.predict(horizon=10)
```

For more details, see the [**Documentation**](https://mateuja.github.io/compotime/).
