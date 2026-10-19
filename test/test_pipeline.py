
from pathlib import Path

import cv2

from ai.pipeline import LicensePlateRecognizer


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT_DIR / "test" / "images" / "sample.jpg"
OUTPUT_PATH = ROOT_DIR / "test" / "images" / "pipeline_result.jpg"

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

recognizer = LicensePlateRecognizer()
results = recognizer.recognize(image)

for index, result in enumerate(results, start=1):
    print(f"\nPlate {index}")
    print("Recognized text:", result["plate"])
    print("Plate confidence:", round(result["plate_confidence"], 3))

    x1, y1, x2, y2 = result["bbox"]

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        image,
        result["plate"],
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

if not results:
    print("No license plates detected.")

if not cv2.imwrite(str(OUTPUT_PATH), image):
    raise RuntimeError(f"Could not save image: {OUTPUT_PATH}")

print(f"\nAnnotated image saved to: {OUTPUT_PATH}")
