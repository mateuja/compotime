# compotime

compotime is a library for forecasting compositional time series in Python. At the moment, it provides an implementation of the models described in the paper ["Forecasting compositional time series: A state space approach"](https://isidl.com/wp-content/uploads/2017/06/E4001-ISIDL.pdf) (Snyder, R.D. et al, 2017). It is constantly tested to be compatible with the major machine learning and statistics libraries within the Python ecosystem.


## Basic usage

This example uses adapted data on the global [share of energy consumption by source (1965-2021)](https://ourworldindata.org/grapher/share-energy-source-sub).

```python
import pandas as pd

from compotime import LocalTrendForecaster, preprocess

URL = "https://raw.githubusercontent.com/mateuja/compotime/main/examples/data/share_energy_source.csv"

date_parser = lambda x: pd.Period(x, "Y")
time_series = (
  pd.read_csv(URL, parse_dates=["Year"], date_parser=date_parser)
  .set_index("Year")
  .pipe(preprocess.treat_small, 0.001)
)

model = LocalTrendForecaster()
model.fit(time_series)
model.predict(horizon=10)
```

For more details, see the [**Documentation**](https://mateuja.github.io/compotime/).
