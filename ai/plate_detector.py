from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models/plate_detector.pt"


class PlateDetector:
    def __init__(self, confidence=0.5):
        self.model = YOLO(str(MODEL_PATH))
        self.confidence = confidence

    def detect(self, image):
        results = self.model.predict(source=image, conf=self.confidence, verbose=False)

        result = results[0]
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append(
                {"bbox": (x1, y1, x2, y2), "confidence": float(box.conf[0])}
            )

        return detections
