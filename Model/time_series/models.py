from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.api.types import is_datetime64_any_dtype



class BaseTimeSeriesModel(ABC):
    target_col: str

    @abstractmethod
    def fit(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None):
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        ...


def _datetime_to_features(df: pd.DataFrame, drop_original: bool = True) -> pd.DataFrame:
    """
    Convert datetime columns (including object columns with timestamps/strings)
    into numeric features usable by sklearn/statsmodels.
    """
    out = df.copy()

    def _looks_like_datetime_series(s: pd.Series) -> bool:
        # Case 1: datetime dtype (includes tz-aware)
        if is_datetime64_any_dtype(s):
            return True

        # Case 2: object column containing Timestamp
        non_null = s.dropna()
        if non_null.empty:
            return False
        sample = non_null.iloc[0]
        if isinstance(sample, pd.Timestamp):
            return True

        # Case 3: object strings that look like dates (light heuristic)
        if s.dtype == "object":
            parsed = pd.to_datetime(non_null.head(50), errors="coerce", utc=True)
            # If most values parse, consider it a date
            return parsed.notna().mean() > 0.8

        return False

    for col in list(out.columns):
        s = out[col]
        if _looks_like_datetime_series(s) or col.lower() in {"date", "time", "timestamp"}:
            dt = pd.to_datetime(s, errors="coerce", utc=True)

            out[f"{col}_hour"] = dt.dt.hour
            out[f"{col}_dow"] = dt.dt.dayofweek
            out[f"{col}_month"] = dt.dt.month
            out[f"{col}_day"] = dt.dt.day

            # cyclic encoding (often improves models)
            out[f"{col}_hour_sin"] = np.sin(2 * np.pi * out[f"{col}_hour"] / 24.0)
            out[f"{col}_hour_cos"] = np.cos(2 * np.pi * out[f"{col}_hour"] / 24.0)
            out[f"{col}_month_sin"] = np.sin(2 * np.pi * out[f"{col}_month"] / 12.0)
            out[f"{col}_month_cos"] = np.cos(2 * np.pi * out[f"{col}_month"] / 12.0)

            if drop_original and col in out.columns:
                out = out.drop(columns=[col])

    return out



def _to_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure X is strictly numeric (float64) while handling bool/object/NaN.
    """
    out = X.copy()

    # bool -> int
    for col in out.select_dtypes(include=["bool"]).columns:
        out[col] = out[col].astype("int64")

    # object -> numeric (strings -> NaN)
    out = out.apply(pd.to_numeric, errors="coerce")

    # NaN handling (statsmodels and sklearn dislike NaN)
    out = out.ffill().bfill().fillna(0.0)

    return out.astype("float64")


def _prepare_endog_exog(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Return y (float64) and X (float64) ready for fit/predict.
    """
    if target_col not in df.columns:
        raise ValueError(f"Missing target column {target_col}")

    y = pd.to_numeric(df[target_col], errors="coerce").astype("float64")

    if feature_cols:
        X = df[list(feature_cols)].copy()
    else:
        X = df.drop(columns=[target_col]).copy()

    X = _datetime_to_features(X, drop_original=True)
    X = _to_numeric_matrix(X)

    # align index
    X = X.loc[y.index]

    return y, X







@dataclass
class ARIMAModel(BaseTimeSeriesModel):
    order: Tuple[int, int, int]
    target_col: str = "temperature_2m"

    def __post_init__(self):
        self._model = None
        self._result = None

    def fit(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None):
        self._model = ARIMA(df[self.target_col], order=self.order, exog=exog)
        self._result = self._model.fit()
        return self

    def predict(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Model not fitted")
        return self._result.predict(start=0, end=len(df) - 1, exog=exog)


@dataclass
class SARIMAModel(BaseTimeSeriesModel):
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    target_col: str = "temperature_2m"

    def __post_init__(self):
        self._model = None
        self._result = None

    def fit(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None):
        self._model = SARIMAX(df[self.target_col], order=self.order, seasonal_order=self.seasonal_order, exog=exog, enforce_stationarity=False, enforce_invertibility=False)
        self._result = self._model.fit(disp=False)
        return self

    def predict(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Model not fitted")
        return self._result.predict(start=0, end=len(df) - 1, exog=exog)


@dataclass
class SARIMAXModel(BaseTimeSeriesModel):
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    target_col: str = "temperature_2m"

    def __post_init__(self):
        self._model = None
        self._result = None

    def fit(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None):
        y = pd.to_numeric(df[self.target_col], errors="coerce").astype("float64")

        X = None
        if exog is not None:
            X = _datetime_to_features(exog, drop_original=True)
            X = _to_numeric_matrix(X)
            X = X.loc[y.index]
        self._model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            exog=X,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = self._model.fit(disp=False)
        return self

    def predict(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Model not fitted")
        # IMPORTANT: predict exactly len(df) points here
        X = None
        if exog is not None:
            X = _datetime_to_features(exog, drop_original=True)
            X = _to_numeric_matrix(X)

        return self._result.predict(start=0, end=len(df) - 1, exog=X)


@dataclass
class RegressionModel(BaseTimeSeriesModel):
    regressor: Any = None
    target_col: str = "temperature_2m"
    feature_cols: Optional[Sequence[str]] = None

    def __post_init__(self):
        if self.regressor is None:
            self.regressor = RandomForestRegressor(n_estimators=200, random_state=42)

    def fit(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None):
            y, X = _prepare_endog_exog(df, target_col=self.target_col, feature_cols=self.feature_cols)
            self.regressor.fit(X, y)
            return self

    def predict(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        _, X = _prepare_endog_exog(df, target_col=self.target_col, feature_cols=self.feature_cols)
        return self.regressor.predict(X)


@dataclass
class GradientBoostingModel(RegressionModel):
    def __post_init__(self):
        if self.regressor is None:
            self.regressor = GradientBoostingRegressor(random_state=42)
