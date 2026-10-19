from pathlib import Path
import cv2
from ai.plate_detector import PlateDetector

ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT_DIR / "test" / "images" / "sample.jpg"
OUTPUT_PATH = ROOT_DIR / "test" / "images" / "sample_detected.jpg"

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

detector = PlateDetector()
detections =detector.detect(image)

print(f"Found {len(detections)} plate(s)")

for index, detection in enumerate(detections, start=1):
    print(f"Plate {index}: {detection}")

    x1, y1, x2, y2 = detection["bbox"]

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

success = cv2.imwrite(str(OUTPUT_PATH), image)

if not success:
    raise RuntimeError(f"Could not save image: {OUTPUT_PATH}")

print(f"Annotated image saved to: {OUTPUT_PATH}")
