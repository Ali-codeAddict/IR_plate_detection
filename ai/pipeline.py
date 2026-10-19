from ai.plate_detector import PlateDetector
from ai.character_detector import CharacterDetector


class LicensePlateRecognizer:
    def __init__(
        self, plate_confidence: float = 0.5, character_confidence: float = 0.25
    ):
        self.plate_detector = PlateDetector(confidence=plate_confidence)
        self.character_detector = CharacterDetector(confidence=character_confidence)

    def recognize(self, image):
        plate_detections = self.plate_detector.detect(image)

        results = []

        for plate_detected in plate_detections:
            x1, y1, x2, y2 = plate_detected["bbox"]
            

            plate_crop = image[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                continue
            
            character_detection = self.character_detector.recognize(
                plate_crop
            )

            results.append({
                "plate": character_detection["text"],
                "plate_confidence": plate_detected["confidence"],
                "bbox": plate_detected["bbox"],
                "characters": character_detection["characters"],
            })
            
        return results