from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class TimeSeriesPreprocessor:
    time_col: str = "date"
    target_col: str = "temperature_2m"
    step_hours: int = 3

    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.time_col not in df.columns:
            raise ValueError(f"Missing time column {self.time_col}")
        if self.target_col not in df.columns:
            raise ValueError(f"Missing target column {self.target_col}")

        cleaned = df.copy()

        # 1) datetime (UTC cohérent avec aggregate_to_step)
        cleaned[self.time_col] = pd.to_datetime(cleaned[self.time_col], utc=True, errors="coerce")
        cleaned = cleaned.dropna(subset=[self.time_col])

        # 2) numeric target (sinon interpolate peut échouer / faire n'importe quoi)
        cleaned[self.target_col] = pd.to_numeric(cleaned[self.target_col], errors="coerce")

        # 3) DatetimeIndex obligatoire pour interpolate(method="time")
        cleaned = cleaned.sort_values(self.time_col).set_index(self.time_col)

        cleaned[self.target_col] = (
            cleaned[self.target_col]
            .interpolate(method="time")
            .bfill()
            .ffill()
        )

        # 4) Revenir à une colonne time
        cleaned = cleaned.reset_index()

        return cleaned

    def aggregate_to_step(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.time_col not in df.columns:
            raise ValueError(f"Missing time column {self.time_col}")

        indexed = df.set_index(pd.to_datetime(df[self.time_col], utc=True))
        aggregated = indexed.resample(f"{self.step_hours}H", label="left", closed="left").mean(numeric_only=True)
        aggregated[self.time_col] = aggregated.index
        aggregated = aggregated.reset_index(drop=True)
        aggregated[self.time_col] = aggregated[self.time_col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return aggregated

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.aggregate_to_step(self.fill_missing(df))
