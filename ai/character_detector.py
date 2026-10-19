from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "character_detector.pt"


class CharacterDetector:
    def __init__(self, confidence=0.5):
        self.model = YOLO(model=str(MODEL_PATH))
        self.confidence = confidence

    def detect(self, plate_image):
        results = self.model.predict(
            source=plate_image,
            conf=self.confidence,
            verbose=False,
        )

        result = results[0]
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append(
                {
                    "class_id": class_id,
                    "char": self.model.names[class_id],
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "x_center": (x1 + x2) / 2,
                }
            )

        detections.sort(key=lambda item: item["x_center"])

        return detections

    def recognize(self, plate_image):
        detections = self.detect(plate_image)

        text = "".join(detection["char"] for detection in detections)

        return {
            "text": text,
            "characters": detections,
        }
