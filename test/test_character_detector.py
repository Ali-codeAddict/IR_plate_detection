
from pathlib import Path

import cv2

from ai.character_detector import CharacterDetector


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT_DIR / "test" / "images" / "plate_crop.jpg"
OUTPUT_PATH = ROOT_DIR / "test" / "images" / "characters_detected.jpg"

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

detector = CharacterDetector()
result = detector.recognize(image)

print("Recognized text:", result["text"])

for character in result["characters"]:
    print(
        f"Character: {character['char']} | "
        f"Confidence: {character['confidence']:.3f}"
    )

    x1, y1, x2, y2 = character["bbox"]

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        image,
        character["char"],
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )

if not cv2.imwrite(str(OUTPUT_PATH), image):
    raise RuntimeError(f"Could not save image: {OUTPUT_PATH}")

print(f"Annotated image saved to: {OUTPUT_PATH}")
