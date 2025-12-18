# Maize Field Image Classification & Weather Time-Series Forecasting

This repository scaffolds the maize field classification pipeline (Chao/ground, Milho/corn, Ervas/weeds, Milho_ervas/mixed) and a weather time-series workflow leveraging Open-Meteo historical data for 3h temperature forecasting.

## Quick start (image classification)
1. Install dependencies (example on Windows):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Donn?es : d?j? copi?es dans `Data/raw/{train,val,test}` (Chao/Ervas/Milho/Milho_ervas). Optionnel (redimensionner/sous-?chantillonner) :
   ```
   python Model/training/preprocess.py ^
     --input-dir Data/raw ^
     --output-dir Data/processed ^
     --size 224 ^
     --limit-per-class 1000   # optionnel pour aller plus vite
   ```
3. Entra?nement (4 classes par d?faut) :
   ```
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
   ```
   - 3 classes d?abord : `--class-filter Chao Ervas Milho`.
   - Runs rapides : `--limit-per-class 500`.
   - Donn?es pr?process?es : remplace `Data/raw/...` par `Data/processed/train_224` et `Data/processed/val_224` + `--image-size 224`.
4. ?valuation et courbes :
   ```
   python Model/training/evaluate.py ^
     --data-dir Data/raw/test ^
     --weights Model/weights/best_model.pt ^
     --confusion-path Visualisation/confusion_matrix.png ^
     --report-path Monitoring/output/metrics.json ^
     --metrics-json Monitoring/output/metrics.json ^
     --training-curves Visualisation/training_curves.png
   ```
5. API:
   ```
   uvicorn Api.inference:app --reload --port 8000
   ```
   Environment overrides: `WEIGHTS_PATH`, `MODEL_NAME`, `IMAGE_SIZE`. POST an image to `/predict` to get class probabilities.

## Weather time-series (Open-Meteo, 3h temperature forecasting)
- Service: `Api/weather_service.py` exposes cached Open-Meteo archive client with retry.
- FastAPI endpoint: `GET /weather/hourly` with query params `latitude`, `longitude`, `start_date`, `end_date`, `variables` (comma-separated; default `temperature_2m`).
- Core modules: `Model/time_series/` (SOLID-friendly components)
  - `data_access.py`: `OpenMeteoHistoricalRepository`, `WeatherQuery` to fetch hourly data.
  - `preprocessing.py`: `TimeSeriesPreprocessor` for interpolation + 3h aggregation.
  - `features.py`: `FeatureBuilder` (lags, rolling stats, time encodings).
  - `models.py`: wrappers for `ARIMAModel`, `SARIMAModel`, `SARIMAXModel`, `RegressionModel` (RF), `GradientBoostingModel`.
  - `hyperparameter.py`: `make_lagged_frame` (lags for linear regression), `grid_search_arima` (AIC-based grid search over (p,d,q) and seasonal orders).
  - `evaluation.py`: `compute_metrics`, `train_test_split_time`, `save_json_results`, `append_log_line` for metrics/logging decisions.
- Notebooks:
  - `Notebooks/05_weather_timeseries_data.ipynb`: fetch, clean (interpolation), aggregate to 3h, quick EDA.
  - `Notebooks/06_weather_timeseries_modeling.ipynb`: ARIMA/SARIMA/SARIMAX (with exog), RF/GBR regressors, residuals, metrics/log export.
- Dependencies added: `pandas`, `openmeteo-requests`, `requests-cache`, `retry-requests`, `statsmodels`.
- Sample workflow:
  ```
  # Acquire and preprocess
  repo = OpenMeteoHistoricalRepository(default_hourly=['temperature_2m'])
  raw_df, meta = repo.fetch(WeatherQuery(lat, lon, start_date, end_date))
  agg_df = TimeSeriesPreprocessor(step_hours=3).run(raw_df)

  # Features + split
  feat_df = FeatureBuilder().build(agg_df)
  train_df, test_df = train_test_split_time(feat_df, test_size=56)

  # Model + metrics + logging
  sarima = SARIMAModel(order=(2,0,2), seasonal_order=(1,1,1,8)).fit(train_df)
  preds = sarima.predict(test_df)
  metrics = compute_metrics(test_df['temperature_2m'], preds)
  save_json_results('Monitoring/output/weather_metrics.json', {'model': 'sarima', 'metrics': metrics})
  append_log_line('Monitoring/output/weather.log', f"SARIMA -> {metrics}")
  ```

## Project layout
```
project_root/
??? Api/
?   ??? inference.py              # FastAPI app (image predict + weather endpoint)
?   ??? weather_service.py        # Open-Meteo client with cache/retry
??? Data/
?   ??? ...                       # datasets (raw/processed)
??? Model/
?   ??? architectures/
?   ??? training/
?   ??? time_series/              # weather forecasting modules
?   ??? weights/
??? Monitoring/
?   ??? output/                   # logs, metrics
??? Notebooks/                    # EDA, preprocessing, training, weather TS
??? Visualisations/
??? README.md
```

## Notes
- Class aliases: `Chao->ground`, `Milho->corn`, `Ervas->weeds`, `Milho_ervas->corn_weeds`.
- Optimizers: Adam, RMSprop, Adagrad. Dropout, augmentations (`none|light|realistic`) et gel du backbone sont configurables.
- Checkpoints stockent m?tadonn?es (classes, taille image, mod?le) pour r??valuer/inf?rer facilement.
- Notebooks : `Notebook/01_eda.ipynb`, `02_preprocessing.ipynb`, `03_training.ipynb`, `03b_training_variants.ipynb`, `03c_comparison_raw_resized_aug.ipynb`, `04_evaluation.ipynb`, `04b_evaluation_variants.ipynb`, `05_weather_timeseries_data.ipynb`, `06_weather_timeseries_modeling.ipynb`. Pense ? s?lectionner le kernel `.venv`.
