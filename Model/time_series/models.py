from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


class BaseTimeSeriesModel(ABC):
    target_col: str

    @abstractmethod
    def fit(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None):
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        ...


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
        self._model = SARIMAX(df[self.target_col], order=self.order, seasonal_order=self.seasonal_order, exog=exog, enforce_stationarity=False, enforce_invertibility=False)
        self._result = self._model.fit(disp=False)
        return self

    def predict(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Model not fitted")
        return self._result.predict(start=0, end=len(df) - 1, exog=exog)


@dataclass
class RegressionModel(BaseTimeSeriesModel):
    regressor: Any = None
    target_col: str = "temperature_2m"
    feature_cols: Optional[Sequence[str]] = None

    def __post_init__(self):
        if self.regressor is None:
            self.regressor = RandomForestRegressor(n_estimators=200, random_state=42)

    def fit(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None):
        X = df[self.feature_cols] if self.feature_cols else df.drop(columns=[self.target_col])
        y = df[self.target_col]
        self.regressor.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        X = df[self.feature_cols] if self.feature_cols else df.drop(columns=[self.target_col])
        return self.regressor.predict(X)


@dataclass
class GradientBoostingModel(RegressionModel):
    def __post_init__(self):
        if self.regressor is None:
            self.regressor = GradientBoostingRegressor(random_state=42)
