# Maize Field Image Classification & Weather Time-Series Forecasting

This repository contains two workflows:
- Image classification of maize fields (Chao/ground, Milho/corn, Ervas/weeds, Milho_ervas/mixed) with FastAPI inference.
- Weather time-series forecasting (Open-Meteo archive) to predict 3h temperature for a city, including stationarity checks and ARIMA/SARIMA grid search.

## Architecture and Data Flows
- API: `Api/inference.py` (FastAPI) exposes `/predict` for images and `/weather/hourly` for hourly weather via `Api/weather_service.py` (cache + retry).
- Image flow: `Data/raw` -> optional `Model/training/preprocess.py` -> training (`Model/training/train.py`) -> weights in `Model/weights` -> inference via FastAPI.
- Weather flow: `OpenMeteoHistoricalRepository` fetches hourly temperatures -> `TimeSeriesPreprocessor` interpolates/aggregates to 3h -> `FeatureBuilder` (lags/rolling/time encodings) -> models (`ARIMA/SARIMA/SARIMAX`, RF, GBR) in `Model/time_series/models.py` -> metrics/logs via notebooks and `evaluation.py`.
- Notebooks: `05_weather_timeseries_data.ipynb` (acquisition/EDA) and `06_weather_timeseries_modeling.ipynb` (stationarity, ACF/PACF, ARIMA/SARIMA grid search, lagged ML baselines, residuals).

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Key dependencies: `torch`, `torchvision`, `fastapi`, `uvicorn`, `openmeteo-requests`, `requests-cache`, `retry-requests`, `pandas`, `statsmodels`.

## Run
- FastAPI (images + weather): `uvicorn Api.inference:app --reload --port 8000`
  - Env overrides: `WEIGHTS_PATH`, `MODEL_NAME`, `IMAGE_SIZE`.
  - Weather endpoint: `GET /weather/hourly?latitude=48.8566&longitude=2.3522&start_date=2023-01-01&end_date=2023-12-31&variables=temperature_2m`
- Notebooks: open `Notebooks/05_weather_timeseries_data.ipynb` then `06_weather_timeseries_modeling.ipynb` (kernel `.venv`).

## Build / Train (Image Classification)
```bash
# Optional preprocessing
python Model/training/preprocess.py ^
  --input-dir Data/raw ^
  --output-dir Data/processed ^
  --size 224 ^
  --limit-per-class 1000

# Train
python Model/training/train.py ^
  --train-dir Data/raw/train ^
  --val-dir Data/raw/val ^
  --model resnet18 ^
  --optimizer adam ^
  --epochs 10 ^
  --batch-size 32 ^
  --dropout 0.3 ^
  --augment realistic ^
  --save-path Model/weights/best_model.pt ^
  --log-dir Monitoring/output

# Evaluate
python Model/training/evaluate.py ^
  --data-dir Data/raw/test ^
  --weights Model/weights/best_model.pt ^
  --confusion-path Visualisation/confusion_matrix.png ^
  --report-path Monitoring/output/metrics.json ^
  --metrics-json Monitoring/output/metrics.json ^
  --training-curves Visualisation/training_curves.png
```

## Weather Time-Series Workflow
- Notebook 05: fetch hourly data via Open-Meteo, interpolate, aggregate to 3h, quick EDA, save to `Data/processed/weather_<city>_3h.csv`.
- Notebook 06: stationarity checks (ADF/KPSS), ACF/PACF plots, ARIMA `(p,d,q)` grid search, SARIMA `(p,d,q)x(P,D,Q,s)` grid search (s=8 for 3h data), SARIMAX with calendar exogenous features, lagged ML baselines (Linear Regression, Random Forest, Gradient Boosting) with configurable lags/rolling windows, residual analysis.

Sample Python snippet:
```python
from Model.time_series import (
    OpenMeteoHistoricalRepository, WeatherQuery, TimeSeriesPreprocessor,
    FeatureBuilder, SARIMAModel, compute_metrics, train_test_split_time,
    save_json_results, append_log_line
)

repo = OpenMeteoHistoricalRepository(default_hourly=['temperature_2m'])
raw_df, meta = repo.fetch(WeatherQuery(48.8566, 2.3522, '2023-01-01', '2023-12-31'))
agg_df = TimeSeriesPreprocessor(step_hours=3).run(raw_df)
feat_df = FeatureBuilder().build(agg_df)
train_df, test_df = train_test_split_time(feat_df, test_size=56)

model = SARIMAModel(order=(2,0,2), seasonal_order=(1,1,1,8)).fit(train_df)
preds = model.predict(test_df)
metrics = compute_metrics(test_df['temperature_2m'], preds)
save_json_results('Monitoring/output/weather_metrics.json', {'model': 'sarima', 'metrics': metrics})
append_log_line('Monitoring/output/weather.log', f\"SARIMA -> {metrics}\")
```

## Testing
- Image classification: run `Model/training/evaluate.py` on held-out data and inspect `Visualisation/` outputs.
- Time-series: use notebook metrics/residual plots; logs/JSON in `Monitoring/output/`.

## Project Layout
```
project_root/
├─ Api/                     # FastAPI app (predict + weather)
├─ Data/                    # raw/processed datasets
├─ Model/
│  ├─ training/             # image training pipeline
│  └─ time_series/          # weather forecasting modules
├─ Monitoring/output/       # logs, metrics
├─ Notebooks/               # EDA, training, weather TS
├─ Visualisations/
└─ README.md
```

## Notes
- Class aliases: `Chao->ground`, `Milho->corn`, `Ervas->weeds`, `Milho_ervas->corn_weeds`.
- Checkpoints store metadata (classes, image size, model) to make reevaluation/inference easier.
- Hyperparameter helpers: `make_lagged_frame`, `grid_search_arima`; logging helpers: `save_json_results`, `append_log_line`.
