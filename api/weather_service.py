import os
from functools import lru_cache
from typing import Iterable, List, Sequence

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

DEFAULT_HOURLY: Sequence[str] = ("temperature_2m",)
DEFAULT_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


@lru_cache(maxsize=1)
def _get_client(cache_dir: str = ".cache/openmeteo", expire_after: int = -1, retries: int = 5, backoff: float = 0.2):
    cache_path = os.environ.get("WEATHER_CACHE_DIR", cache_dir)
    expire = int(os.environ.get("WEATHER_CACHE_EXPIRE", str(expire_after)))
    retry_attempts = int(os.environ.get("WEATHER_RETRY", str(retries)))
    backoff_factor = float(os.environ.get("WEATHER_BACKOFF", str(backoff)))

    session = requests_cache.CachedSession(cache_path, expire_after=expire)
    retry_session = retry(session, retries=retry_attempts, backoff_factor=backoff_factor)
    return openmeteo_requests.Client(session=retry_session)


def fetch_hourly_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly: Iterable[str] | None = None,
    url: str = DEFAULT_ARCHIVE_URL,
):
    """Fetch hourly weather time series from Open-Meteo archive API."""

    hourly_vars: List[str] = list(hourly) if hourly else list(DEFAULT_HOURLY)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
    }

    client = _get_client()
    responses = client.weather_api(url, params=params)
    response = responses[0]

    hourly_block = response.Hourly()
    dates = pd.date_range(
        start=pd.to_datetime(hourly_block.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly_block.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly_block.Interval()),
        inclusive="left",
    )

    data = {"date": dates}
    for idx, variable in enumerate(hourly_vars):
        values = hourly_block.Variables(idx).ValuesAsNumpy()
        data[variable] = values

    df = pd.DataFrame(data)
    df["date"] = df["date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "metadata": {
            "coordinates": {"lat": response.Latitude(), "lon": response.Longitude()},
            "elevation_m": response.Elevation(),
            "utc_offset_seconds": response.UtcOffsetSeconds(),
            "interval_seconds": hourly_block.Interval(),
            "hourly": hourly_vars,
        },
        "data": df.to_dict(orient="records"),
    }
