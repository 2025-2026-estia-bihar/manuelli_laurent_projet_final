from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def make_lagged_frame(series: pd.Series, lags: Sequence[int], dropna: bool = True) -> pd.DataFrame:
    """
    Build a supervised learning frame for linear/elastic/OLS regressions with lagged targets.
    Returns a DataFrame with the original target and lag columns.
    """
    df = pd.DataFrame({"target": series})
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    return df.dropna().reset_index(drop=True) if dropna else df


@dataclass
class ArimaSearchResult:
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int] | None
    aic: float


def grid_search_arima(
    series: pd.Series,
    p_values: Iterable[int],
    d_values: Iterable[int],
    q_values: Iterable[int],
    seasonal_orders: Iterable[Tuple[int, int, int, int]] | None = None,
    exog: pd.DataFrame | None = None,
    maxiter: int = 200,
) -> ArimaSearchResult | None:
    """
    Brute-force grid search over (p,d,q) and seasonal orders for ARIMA/SARIMAX based on AIC.
    If seasonal_orders is None, runs plain ARIMA. Otherwise tries SARIMAX on each seasonal tuple.
    """
    best: ArimaSearchResult | None = None
    combos = list(product(p_values, d_values, q_values))

    for order in combos:
        if seasonal_orders:
            for seasonal in seasonal_orders:
                result = _fit_candidate(series, order, seasonal, exog=exog, maxiter=maxiter)
                best = _update_best(best, result)
        else:
            result = _fit_candidate(series, order, None, exog=exog, maxiter=maxiter)
            best = _update_best(best, result)
    return best


def _fit_candidate(
    series: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int] | None,
    exog: pd.DataFrame | None,
    maxiter: int,
) -> ArimaSearchResult | None:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if seasonal_order:
                model = SARIMAX(
                    series,
                    order=order,
                    seasonal_order=seasonal_order,
                    exog=exog,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            else:
                model = ARIMA(series, order=order, exog=exog)
            res = model.fit(method_kwargs={"maxiter": maxiter}, disp=False)
            return ArimaSearchResult(order=order, seasonal_order=seasonal_order, aic=res.aic)
    except Exception:
        return None


def _update_best(current: ArimaSearchResult | None, candidate: ArimaSearchResult | None):
    if candidate is None:
        return current
    if current is None or candidate.aic < current.aic:
        return candidate
    return current
