# Maize Field Image Classification

This repository scaffolds the maize field classification pipeline (Chao/ground, Milho/corn, Ervas/weeds, Milho_ervas/mixed). It includes preprocessing, training, evaluation, FastAPI inference, and monitoring outputs.

## Quick start
1. Install dependencies (example on Windows):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Données : déjà copiées dans `Data/raw/{train,val,test}` (Chao/Ervas/Milho/Milho_ervas). Optionnel (redimensionner/sous-échantillonner) :
   ```
   python Model/training/preprocess.py ^
     --input-dir Data/raw ^
     --output-dir Data/processed ^
     --size 224 ^
     --limit-per-class 1000   # optionnel pour aller plus vite
   ```
3. Entraînement (4 classes par défaut) :
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
   - 3 classes d’abord : `--class-filter Chao Ervas Milho`.
   - Runs rapides : `--limit-per-class 500`.
   - Données préprocessées : remplace `Data/raw/...` par `Data/processed/train_224` et `Data/processed/val_224` + `--image-size 224`.
4. Évaluation et courbes :
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

## Project layout
```
project_root/
??? Data/
?   ??? raw/{train,val,test}/
?   ??? processed/{train_224,val_224,test_224}/
?   ??? metadata/dataframe.csv
??? Model/
?   ??? architectures/{baseline_cnn.py,transfer_resnet.py}
?   ??? training/{train.py,evaluate.py,callbacks.py,utils.py,preprocess.py}
?   ??? weights/best_model.pt
??? Api/inference.py
??? Notebook/ (EDA, preprocessing, training, evaluation)
??? Visualisation/ (confusion_matrix.png, training_curves.png, class_distribution.png)
??? Monitoring/output/{logs.txt,metrics.json}
??? README.md
```

## Notes
- Class aliases: `Chao->ground`, `Milho->corn`, `Ervas->weeds`, `Milho_ervas->corn_weeds`.
- Optimizers: Adam, RMSprop, Adagrad. Dropout, augmentations (`none|light|realistic`) et gel du backbone sont configurables.
- Checkpoints stockent métadonnées (classes, taille image, modèle) pour réévaluer/inférer facilement.
- Notebooks : `Notebook/01_eda.ipynb`, `02_preprocessing.ipynb`, `03_training.ipynb`, `03b_training_variants.ipynb`, `03c_comparison_raw_resized_aug.ipynb`, `04_evaluation.ipynb`, `04b_evaluation_variants.ipynb`. Pense à sélectionner le kernel `.venv`.
