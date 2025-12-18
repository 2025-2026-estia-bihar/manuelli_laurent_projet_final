# Maize Field Image Classification & Weather Time-Series Forecasting

This project combines two workflows:
- Image classification of maize fields (Chao/ground, Milho/corn, Ervas/weeds, Milho_ervas/mixed) with FastAPI inference.
- Weather time-series forecasting (Open-Meteo historical API) to predict 3h temperature for a city.

## Architecture & data flows
- API: `Api/inference.py` (FastAPI) exposes `/predict` for images and `/weather/hourly` for hourly weather data via `Api/weather_service.py` (cache + retry).
- Image flow: `Data/raw` -> optional `Model/training/preprocess.py` -> training (`Model/training/train.py`) -> weights in `Model/weights` -> inference via FastAPI.
- Weather flow: `OpenMeteoHistoricalRepository` fetches hourly temps -> `TimeSeriesPreprocessor` interpolates/aggregates to 3h -> `FeatureBuilder` (lags/rolling/time encodings) -> models (`ARIMA/SARIMA/SARIMAX`, RF, GBR) in `Model/time_series/models.py` -> metrics/logs via `evaluation.py` (JSON + log files).
- Notebooks: `05_weather_timeseries_data.ipynb` (acquisition/EDA) and `06_weather_timeseries_modeling.ipynb` (modeling, metrics, residuals).

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Key deps: `torch`, `torchvision`, `fastapi`, `uvicorn`, `openmeteo-requests`, `requests-cache`, `retry-requests`, `pandas`, `statsmodels`.

## Run
- FastAPI (images + weather): `uvicorn Api.inference:app --reload --port 8000`
  - Env overrides: `WEIGHTS_PATH`, `MODEL_NAME`, `IMAGE_SIZE`.
  - Weather endpoint: `GET /weather/hourly?latitude=48.8566&longitude=2.3522&start_date=2023-01-01&end_date=2023-12-31&variables=temperature_2m`
- Notebooks: open `Notebooks/05_weather_timeseries_data.ipynb` then `06_weather_timeseries_modeling.ipynb` (kernel `.venv`).

## Build / Train (image classification)
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

## Experimentation (weather time-series)
Sample Python workflow:
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
append_log_line('Monitoring/output/weather.log', f"SARIMA -> {metrics}")
```

## Testing
- Image classification: use `Model/training/evaluate.py` on held-out data and inspect `Visualisation/` outputs.
- Time-series: check residual plots and metrics in notebooks; logs/JSON in `Monitoring/output/`.

## Project layout
```
project_root/
├── Api/                  # FastAPI app (predict + weather)
├── Data/                 # raw/processed datasets
├── Model/
│   ├── training/         # image training pipeline
│   └── time_series/      # weather forecasting modules
├── Monitoring/output/    # logs, metrics
├── Notebooks/            # EDA, training, weather TS
├── Visualisations/
└── README.md
```

## Notes
- Class aliases: `Chao->ground`, `Milho->corn`, `Ervas->weeds`, `Milho_ervas->corn_weeds`.
- Checkpoints stockent les métadonnées (classes, taille image, modèle) pour réévaluer/inférer facilement.
- Hyperparameter helpers: `make_lagged_frame`, `grid_search_arima`; logging helpers: `save_json_results`, `append_log_line`.
