"""Build or verify the EXP-DIAG-007 publication manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "artifact_manifest.csv"
MAX_ORDINARY_BYTES = 10 * 1024 * 1024


class ManifestError(RuntimeError):
    pass


def git(repository: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repository), *arguments],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise ManifestError(completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout


def decode_paths(payload: bytes) -> list[PurePosixPath]:
    result = []
    for raw in payload.split(b"\0"):
        if not raw:
            continue
        path = PurePosixPath(os.fsdecode(raw))
        if path.is_absolute() or ".." in path.parts:
            raise ManifestError(f"unsafe Git path: {path}")
        result.append(path)
    return result


def publication_paths() -> tuple[Path, list[PurePosixPath]]:
    repository = Path(os.fsdecode(git(ROOT, "rev-parse", "--show-toplevel").strip())).resolve()
    root = ROOT.resolve()
    relative_root = root.relative_to(repository).as_posix()
    tracked = decode_paths(
        git(repository, "ls-files", "--cached", "--full-name", "-z", "--", relative_root)
    )
    untracked = decode_paths(
        git(
            repository,
            "ls-files",
            "--others",
            "--exclude-standard",
            "--full-name",
            "-z",
            "--",
            relative_root,
        )
    )
    return repository, sorted(set(tracked + untracked), key=lambda value: value.as_posix())


def lfs_status(repository: Path, paths: list[PurePosixPath]) -> dict[PurePosixPath, bool]:
    if not paths:
        return {}
    payload = b"".join(os.fsencode(path.as_posix()) + b"\0" for path in paths)
    fields = git(repository, "check-attr", "-z", "--stdin", "filter", input_bytes=payload).split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 3:
        raise ManifestError("malformed git check-attr output")
    result = {}
    for index in range(0, len(fields), 3):
        path, attribute, value = fields[index : index + 3]
        if attribute != b"filter":
            raise ManifestError("unexpected git attribute")
        result[PurePosixPath(os.fsdecode(path))] = value == b"lfs"
    return result


def hash_file(path: Path) -> tuple[int, str]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ManifestError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    after = path.lstat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != after.st_size:
        raise ManifestError(f"file changed while hashing: {path}")
    return size, digest.hexdigest()


def expected_bytes() -> tuple[bytes, int]:
    repository, paths = publication_paths()
    output_relative = OUTPUT.resolve().relative_to(repository)
    lfs = lfs_status(repository, paths)
    rows = []
    for repository_path in paths:
        if repository_path == PurePosixPath(output_relative.as_posix()):
            continue
        path = repository.joinpath(*repository_path.parts)
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ManifestError(f"file escapes publication root: {path}") from exc
        if path.is_symlink():
            raise ManifestError(f"symlink is not publishable: {path}")
        size, digest = hash_file(path)
        if size > MAX_ORDINARY_BYTES and not lfs.get(repository_path, False):
            raise ManifestError(f"file exceeds 10 MiB without Git LFS: {relative.as_posix()}")
        rows.append((relative.as_posix(), size, digest))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("relative_path", "bytes", "sha256"))
    writer.writerows(sorted(rows))
    return buffer.getvalue().encode("utf-8"), len(rows)


def write_atomic(payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".artifact_manifest.", suffix=".tmp", dir=ROOT)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, OUTPUT)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload, count = expected_bytes()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != payload:
            raise SystemExit("artifact manifest is missing or stale")
        print(f"verified {count} publication files")
    else:
        write_atomic(payload)
        print(f"wrote {count} publication files")


if __name__ == "__main__":
    main()
