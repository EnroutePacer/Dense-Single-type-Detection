import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.table_cells.evaluate_local_hybrid_npz import evaluate, evaluate_state


class LocalHybridNpzTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        annotation = root / "val.json"
        annotation.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 1, "width": 640, "height": 640},
                        {"id": 2, "width": 640, "height": 640},
                    ],
                    "categories": [{"id": 1, "name": "cell"}],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
                        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [100, 100, 100, 100], "area": 10000, "iscrowd": 0},
                        # Area is large, but the 15px short side makes it local-small.
                        {"id": 3, "image_id": 2, "category_id": 1, "bbox": [20, 20, 15, 200], "area": 3000, "iscrowd": 0},
                    ],
                }
            ),
            encoding="utf-8",
        )
        boxes = np.zeros((2, 3, 4), dtype=np.float32)
        scores = np.asarray([[0.99, 0.8, 0.7], [0.9, 0.1, 0.05]], dtype=np.float32)
        boxes[0] = [[100, 100, 100, 100], [0, 0, 10, 10], [0, 0, 10, 10]]
        boxes[1] = [[20, 20, 15, 200], [0, 0, 1, 1], [0, 0, 1, 1]]
        prediction = root / "state.predictions.npz"
        np.savez_compressed(
            prediction,
            schema=np.asarray("sgqs_compact_predictions_v1"),
            image_ids=np.asarray([1, 2], dtype=np.int64),
            boxes_xywh=boxes,
            scores=scores,
            category_id=np.asarray(1, dtype=np.int64),
        )
        return annotation, prediction

    def test_local_scope_and_per_image_recall(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation, prediction = self._fixture(Path(directory))
            summary, rows = evaluate_state(annotation, prediction, budgets=(1, 3))
            self.assertEqual(summary["local_small_gt"], 2)
            self.assertEqual(summary["budgets"]["1"]["AR"], 0.5)
            self.assertAlmostEqual(summary["budgets"]["3"]["AR"], 1.0)
            by_image = {row["image_id"]: row for row in rows}
            self.assertEqual(by_image[1]["local_gt_count"], 1)
            self.assertEqual(by_image[1]["tp75_1"], 0)
            self.assertEqual(by_image[1]["tp75_3"], 1)
            self.assertEqual(by_image[2]["recall75_1"], 1.0)

    def test_multi_state_output_is_pairable(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation, prediction = self._fixture(Path(directory))
            output = Path(directory) / "eval"
            result = evaluate(
                annotation,
                [("r0_e34_ema", prediction), ("r5_e34_ema", prediction)],
                output,
                budgets=(1, 3),
            )
            self.assertEqual(set(result["states"]), {"r0_e34_ema", "r5_e34_ema"})
            text = (output / "per_image_recall75.csv").read_text(encoding="utf-8")
            self.assertEqual(text.count("r0_e34_ema"), 2)
            self.assertEqual(text.count("r5_e34_ema"), 2)
            self.assertEqual(
                result["per_image_recall75_sha256"],
                hashlib.sha256(
                    (output / "per_image_recall75.csv").read_bytes()
                ).hexdigest(),
            )

    def test_dense_cohort_is_reported_from_the_same_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation, prediction = self._fixture(Path(directory))
            summary, _ = evaluate_state(
                annotation,
                prediction,
                budgets=(1, 3),
                dense_min_gt=2,
            )
            dense = summary["dense_ge2"]
            self.assertEqual(dense["image_ids"], [1])
            self.assertEqual(dense["metrics"]["images"], 1)
            self.assertEqual(dense["metrics"]["noncrowd_gt"], 2)

    def test_unsorted_image_ids_are_rejected_before_image_mismatch(self):
        """The evaluator fails closed instead of silently mispairing rows."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation, prediction = self._fixture(root)
            with np.load(prediction, allow_pickle=False) as archive:
                arrays = {key: np.asarray(archive[key]) for key in archive.files}
            reverse = np.asarray([1, 0], dtype=np.int64)
            unsorted = root / "unsorted.predictions.npz"
            arrays["image_ids"] = arrays["image_ids"][reverse]
            arrays["boxes_xywh"] = arrays["boxes_xywh"][reverse]
            arrays["scores"] = arrays["scores"][reverse]
            np.savez_compressed(unsorted, **arrays)

            with self.assertRaisesRegex(ValueError, "sorted ascending"):
                evaluate_state(annotation, unsorted, budgets=(1, 3))


if __name__ == "__main__":
    unittest.main()
