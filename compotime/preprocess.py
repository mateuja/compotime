"""Preprocess funcions for TS."""
import pandas as pd


def treat_zeros(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Replace zeros by `thresh` and adjust series to add up to 1.

    Parameters
    ----------
        table: Table contianing the time series data.
        threshold: Value to convert zeros to.

    Returns
    -------
        Table with corrected time series.
    """
    # ruff: noqa: B023
    table = table.copy()
    m = (table < threshold).sum(axis=1)
    for idx, row in table.iterrows():
        mask = (row < threshold)
        m = mask.sum()
        S = row[~mask].sum()
        if not m:
            continue

        table.loc[idx] = row.mask(mask, threshold).where(mask, lambda x: (1 - threshold * m) * x / S)

    return table