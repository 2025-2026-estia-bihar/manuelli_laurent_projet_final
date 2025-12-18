from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from Api.weather_service import DEFAULT_HOURLY, fetch_hourly_weather


@dataclass
class WeatherQuery:
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    hourly: Sequence[str] | None = None


class WeatherRepository:
    """Abstraction over any weather data source."""

    def fetch(self, query: WeatherQuery) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        raise NotImplementedError


class OpenMeteoHistoricalRepository(WeatherRepository):
    """
    Fetches historical hourly weather data via Open-Meteo (archive API).
    Single responsibility: retrieving raw data as a DataFrame + metadata dict.
    """

    def __init__(self, default_hourly: Iterable[str] | None = None):
        self.default_hourly = list(default_hourly) if default_hourly else list(DEFAULT_HOURLY)

    def fetch(self, query: WeatherQuery) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        hourly_vars: List[str] = list(query.hourly) if query.hourly else self.default_hourly
        payload = fetch_hourly_weather(
            latitude=query.latitude,
            longitude=query.longitude,
            start_date=query.start_date,
            end_date=query.end_date,
            hourly=hourly_vars,
        )
        df = pd.DataFrame(payload["data"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df, payload.get("metadata", {})
