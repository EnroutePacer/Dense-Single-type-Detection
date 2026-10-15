"""Publication-set, safety, and determinism tests for the bundle manifest."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("build_manifest.py")
SPEC = importlib.util.spec_from_file_location("exp_diag_006_build_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def git(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", os.fspath(root), *arguments],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


class RepositoryFixture:
    def __init__(self, temporary: str) -> None:
        self.repository = Path(temporary)
        git(self.repository, "init", "--quiet")
        self.root = self.repository / "article" / "experiment"
        self.root.mkdir(parents=True)

    def write(self, relative: str, payload: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def add(self, *relative: str) -> None:
        repository_relative = [
            (self.root / value).relative_to(self.repository).as_posix()
            for value in relative
        ]
        git(self.repository, "add", "--", *repository_relative)


class BuildManifestTests(unittest.TestCase):
    def test_git_publication_set_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RepositoryFixture(temporary)
            output = fixture.root / "artifact_manifest.csv"
            expected_files = {
                "README.md": b"bundle\n",
                "data/generated.csv": b"key,value\na,1\n",
                "nested/artifact_manifest.csv": b"nested evidence\n",
            }
            for relative, payload in expected_files.items():
                fixture.write(relative, payload)
            fixture.add("README.md", "nested/artifact_manifest.csv")

            fixture.write("private.log", b"secret\n")
            fixture.write("scratch/cache.pyc", b"bytecode")
            (fixture.repository / ".gitignore").write_text(
                "*.log\n__pycache__/\n*.pyc\n", encoding="utf-8"
            )

            count = MODULE.build_manifest(fixture.root, output)
            first_bytes = output.read_bytes()
            second_count = MODULE.build_manifest(fixture.root, output)

            self.assertEqual(count, len(expected_files))
            self.assertEqual(second_count, count)
            self.assertEqual(output.read_bytes(), first_bytes)
            self.assertEqual(MODULE.check_manifest(fixture.root, output), count)

            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["relative_path"] for row in rows], sorted(expected_files)
            )
            for row in rows:
                payload = expected_files[row["relative_path"]]
                self.assertEqual(int(row["bytes"]), len(payload))
                self.assertEqual(row["sha256"], hashlib.sha256(payload).hexdigest())

    def test_check_detects_missing_content_and_membership_changes_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RepositoryFixture(temporary)
            output = fixture.root / "artifact_manifest.csv"
            tracked = fixture.write("tracked.txt", b"one\n")
            fixture.add("tracked.txt")

            with self.assertRaises(MODULE.StaleManifestError):
                MODULE.check_manifest(fixture.root, output)
            MODULE.build_manifest(fixture.root, output)
            original_manifest = output.read_bytes()

            tracked.write_bytes(b"two\n")
            with self.assertRaises(MODULE.StaleManifestError):
                MODULE.check_manifest(fixture.root, output)
            self.assertEqual(output.read_bytes(), original_manifest)

            tracked.write_bytes(b"one\n")
            fixture.write("new.csv", b"generated\n")
            with self.assertRaises(MODULE.StaleManifestError):
                MODULE.check_manifest(fixture.root, output)
            self.assertEqual(output.read_bytes(), original_manifest)

    def test_large_ordinary_file_fails_but_exact_lfs_attribute_allows_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RepositoryFixture(temporary)
            output = fixture.root / "artifact_manifest.csv"
            fixture.write("large.bin", b"x" * 33)

            with mock.patch.object(MODULE, "MAX_ORDINARY_FILE_BYTES", 32):
                with self.assertRaisesRegex(MODULE.ManifestError, "filter=lfs"):
                    MODULE.build_manifest(fixture.root, output)

                fixture.write(".gitattributes", b"large.bin filter=LFS\n")
                with self.assertRaisesRegex(MODULE.ManifestError, "filter=lfs"):
                    MODULE.build_manifest(fixture.root, output)

                fixture.write(".gitattributes", b"large.bin filter=lfs\n")
                self.assertEqual(MODULE.build_manifest(fixture.root, output), 2)

    def test_output_outside_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RepositoryFixture(temporary)
            fixture.write("evidence.txt", b"evidence\n")
            outside = fixture.repository / "outside.csv"
            with self.assertRaisesRegex(MODULE.ManifestError, "escapes"):
                MODULE.build_manifest(fixture.root, outside)
            self.assertFalse(outside.exists())

    def test_symlink_publication_entry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RepositoryFixture(temporary)
            target = fixture.write("target.txt", b"target\n")
            link = fixture.root / "link.txt"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable on this platform: {exc}")
            fixture.add("link.txt")
            with self.assertRaisesRegex(MODULE.ManifestError, "symlink"):
                MODULE.build_manifest(fixture.root, fixture.root / "artifact_manifest.csv")

    def test_atomic_replace_failure_preserves_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RepositoryFixture(temporary)
            output = fixture.root / "artifact_manifest.csv"
            fixture.write("evidence.txt", b"evidence\n")
            output.write_bytes(b"previous manifest\n")

            with mock.patch.object(MODULE.os, "replace", side_effect=OSError("boom")):
                with self.assertRaisesRegex(OSError, "boom"):
                    MODULE.build_manifest(fixture.root, output)
            self.assertEqual(output.read_bytes(), b"previous manifest\n")
            self.assertEqual(list(fixture.root.glob(".artifact_manifest.csv.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
