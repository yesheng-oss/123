import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from answer_card.dataset import convert_labelme_to_yolo
from answer_card.predictor import Detection, save_labelme_json


class LabelConversionTests(unittest.TestCase):
    def test_converts_labelme_rectangles_to_yolo_rows(self):
        data = {
            "imageWidth": 200,
            "imageHeight": 100,
            "shapes": [
                {
                    "label": "th",
                    "shape_type": "rectangle",
                    "points": [[10, 20], [30, 40]],
                },
                {
                    "label": "option",
                    "shape_type": "rectangle",
                    "points": [[100, 50], [140, 90]],
                },
                {
                    "label": "ignored",
                    "shape_type": "rectangle",
                    "points": [[0, 0], [10, 10]],
                },
            ],
        }

        rows, warnings = convert_labelme_to_yolo(data)

        self.assertEqual(warnings, [])
        self.assertEqual(rows, ["0 0.100000 0.300000 0.100000 0.200000", "1 0.600000 0.700000 0.200000 0.400000"])

    def test_skips_invalid_shapes_with_warnings(self):
        data = {
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {"label": "th", "shape_type": "polygon", "points": [[0, 0], [1, 1]]},
                {"label": "option", "shape_type": "rectangle", "points": [[10, 10], [10, 20]]},
            ],
        }

        rows, warnings = convert_labelme_to_yolo(data)

        self.assertEqual(rows, [])
        self.assertEqual(len(warnings), 2)

    def test_clamps_reversed_out_of_bounds_points(self):
        data = {
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {
                    "label": "option",
                    "shape_type": "rectangle",
                    "points": [[120, 80], [-20, 20]],
                }
            ],
        }

        rows, warnings = convert_labelme_to_yolo(data)

        self.assertEqual(warnings, [])
        self.assertEqual(rows, ["1 0.500000 0.500000 1.000000 0.600000"])

    def test_saves_labelme_json_from_detections(self):
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "sample.json"
            detections = [
                Detection(label="th", confidence=0.9, x1=1, y1=2, x2=11, y2=22),
                Detection(label="option", confidence=0.8, x1=20, y1=30, x2=40, y2=50),
            ]

            save_labelme_json(out_path, "sample.jpg", 100, 80, detections)

            saved = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["imagePath"], "sample.jpg")
            self.assertEqual(saved["imageWidth"], 100)
            self.assertEqual(saved["imageHeight"], 80)
            self.assertEqual([shape["label"] for shape in saved["shapes"]], ["th", "option"])
            self.assertEqual(saved["shapes"][0]["points"], [[1.0, 2.0], [11.0, 22.0]])


if __name__ == "__main__":
    unittest.main()
