import csv
import pathlib
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RepositoryContractTest(unittest.TestCase):
    def test_m_launcher_respects_external_config(self):
        text = (ROOT / "tools/table_cells/train_dfine_m_table_cell_multigpu.sh").read_text(encoding="utf-8")
        self.assertIn('CONFIG_FILE="${CONFIG_FILE:-', text)

    def test_maxdets_300_is_explicit(self):
        text = (ROOT / "engine/data/dataset/coco_eval.py").read_text(encoding="utf-8")
        self.assertIn("[1, 10, 300]", text)

    def test_experiment_registry_ids_and_paths_are_unique(self):
        with (ROOT / "experiments/registry.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        ids = [row["experiment_id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        for row in rows:
            record = ROOT / row["record_path"]
            self.assertTrue(record.is_dir(), row["record_path"])
            metadata = yaml.safe_load((record / "experiment.yaml").read_text(encoding="utf-8"))
            self.assertEqual(metadata["experiment_id"], row["experiment_id"])
            self.assertEqual(metadata["slug"], row["slug"])
            self.assertEqual(metadata["status"], row["status"])
            self.assertEqual(metadata["validity"], row["validity"])

    def test_canonical_research_files_exist(self):
        required = [
            "docs/research/research_question.md",
            "docs/research/facts/README.md",
            "docs/research/hypotheses/README.md",
            "docs/claims/claim_evidence_matrix.yaml",
            "docs/evidence/artifact_registry.csv",
        ]
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
