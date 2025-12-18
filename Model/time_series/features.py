from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class FeatureBuilder:
    target_col: str = "temperature_2m"
    time_col: str = "date"
    lags: Sequence[int] = field(default_factory=lambda: [1, 2, 3, 4, 8])
    rolling_windows: Sequence[int] = field(default_factory=lambda: [2, 4, 8])

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        dt = pd.to_datetime(enriched[self.time_col], utc=True)
        enriched["hour"] = dt.dt.hour
        enriched["dayofweek"] = dt.dt.dayofweek
        enriched["month"] = dt.dt.month
        enriched["sin_day"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        enriched["cos_day"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        return enriched

    def add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        for lag in self.lags:
            enriched[f"{self.target_col}_lag_{lag}"] = enriched[self.target_col].shift(lag)
        return enriched

    def add_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        for window in self.rolling_windows:
            enriched[f"{self.target_col}_roll_mean_{window}"] = enriched[self.target_col].rolling(window).mean()
        return enriched

    def build(self, df: pd.DataFrame, exog_cols: Iterable[str] | None = None) -> pd.DataFrame:
        features = self.add_time_features(df)
        features = self.add_lags(features)
        features = self.add_rolling(features)
        if exog_cols:
            keep = list(exog_cols)
            for col in keep:
                if col not in features.columns:
                    raise ValueError(f"Missing exogenous column: {col}")
        features = features.dropna().reset_index(drop=True)
        return features
