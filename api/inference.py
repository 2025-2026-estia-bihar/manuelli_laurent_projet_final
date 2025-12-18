import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, Query
from PIL import Image, ImageOps
from torchvision import transforms

from Model.training.utils import add_repo_to_path, build_transforms, create_model

add_repo_to_path()

from Api.weather_service import DEFAULT_HOURLY, fetch_hourly_weather

DEFAULT_CLASSES = ["ground", "corn", "weeds", "corn_weeds"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = Path(os.environ.get("WEIGHTS_PATH", "Model/weights/best_model.pt"))
MODEL_NAME = os.environ.get("MODEL_NAME", "resnet18")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))

app = FastAPI(title="Corn field classifier")


def load_checkpoint():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    metadata = checkpoint.get("metadata", {})
    classes = metadata.get("class_names", DEFAULT_CLASSES)
    model_name = metadata.get("model_name", MODEL_NAME)
    img_size = metadata.get("img_size", IMAGE_SIZE)
    model = create_model(
        model_name=model_name,
        num_classes=len(classes),
        dropout=metadata.get("dropout", 0.3),
        pretrained=False,
        freeze_backbone=False,
    )
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(DEVICE)
    preprocess = build_transforms(img_size, augment="none")
    return model, classes, preprocess


model, classes, preprocess = load_checkpoint()


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    probs_list = probs.cpu().tolist()
    top_idx = int(torch.argmax(probs).item())
    return {
        "predicted_class": classes[top_idx],
        "probabilities": {cls: float(prob) for cls, prob in zip(classes, probs_list)},
    }


@app.get("/weather/hourly")
async def weather_hourly(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: str = Query(
        ",".join(DEFAULT_HOURLY),
        description="Comma-separated Open-Meteo hourly variables (e.g. temperature_2m,relative_humidity_2m)",
    ),
):
    hourly_vars = [var.strip() for var in variables.split(",") if var.strip()]
    result = fetch_hourly_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        hourly=hourly_vars,
    )
    return result
