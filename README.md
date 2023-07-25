# compotime

![Build](https://github.com/mateuja/compotime/actions/workflows/build.yml/badge.svg?branch=main) [![codecov](https://codecov.io/gh/mateuja/compotime/branch/main/graph/badge.svg?token=9UMGS957L2)](https://codecov.io/gh/mateuja/compotime)

compotime is a library for forecasting compositional time series in Python. At the moment, it provides an implementation of the models described in the paper ["Forecasting compositional time series: A state space approach"](https://isidl.com/wp-content/uploads/2017/06/E4001-ISIDL.pdf) (Snyder, R.D. et al, 2017). It is constantly tested to be compatible with the major machine learning and statistics libraries within the Python ecosystem.

## Quick install

compotime is currently available for python 3.9, 3.10 and 3.11. It can be installed from PyPI:

```bash
pip install compotime
```

## Basic usage

This example uses adapted data on the global [share of energy consumption by source (1965-2021)](https://ourworldindata.org/grapher/share-energy-source-sub).

```python
import pandas as pd

from compotime import LocalTrendForecaster, preprocess

URL = "https://raw.githubusercontent.com/mateuja/compotime/main/examples/data/share_energy_source.csv"

time_series = pd.read_csv(URL, index_col="Year").pipe(preprocess.treat_small, 1e-8)

model = LocalTrendForecaster().fit(time_series)
model.predict(horizon=10)
```

For more details, see the [**Documentation**](https://mateuja.github.io/compotime/).
