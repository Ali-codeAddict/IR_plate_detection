from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

models_path = {
    "Plate detector": MODELS_DIR / "plate_detector.pt",
    "Character detector": MODELS_DIR / "character_detector.pt",
}

for model_name, model_path in models_path.items():
    print(f"\n--- {model_name} ---")
    print(f"Path: {model_path}")

    model = YOLO(str(model_path))

    print(f"Task: {model.task}")
    print(f"Number of classes: {len(model.names)}")
    print(f"Class names: {model.names}")