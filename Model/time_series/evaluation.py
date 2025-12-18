from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from datetime import datetime


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def train_test_split_time(df: pd.DataFrame, test_size: int):
    if test_size <= 0 or test_size >= len(df):
        raise ValueError("test_size must be between 1 and len(df)-1")
    return df.iloc[:-test_size].reset_index(drop=True), df.iloc[-test_size:].reset_index(drop=True)


def save_json_results(json_path: str | Path, payload: Dict[str, Any], indent: int = 2):
    """Persist evaluation or search results to JSON (creates parent dirs)."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)


def append_log_line(log_path: str | Path, message: str, timestamp: bool = True):
    """Append a single log line for quick audit/decisions."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"[{datetime.utcnow().isoformat()}] " if timestamp else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + message + "\n")
