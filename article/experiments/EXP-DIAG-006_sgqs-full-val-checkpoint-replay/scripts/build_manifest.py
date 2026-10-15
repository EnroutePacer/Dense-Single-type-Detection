"""Build and verify the deterministic publication manifest for EXP-DIAG-006.

The publication set is defined by Git: files already in the index plus untracked
files that are not ignored. This intentionally permits freshly generated
evidence to be manifested before ``git add`` while keeping local/private files
out of the bundle.
"""

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
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "artifact_manifest.csv"
MAX_ORDINARY_FILE_BYTES = 10 * 1024 * 1024


class ManifestError(RuntimeError):
    """Base class for fail-closed publication-manifest errors."""


class StaleManifestError(ManifestError):
    """Raised when ``--check`` finds a missing or stale manifest."""


def _run_git(
    repository: Path,
    arguments: Sequence[str],
    *,
    input_bytes: bytes | None = None,
) -> bytes:
    command = ["git", "-C", os.fspath(repository), *arguments]
    try:
        completed = subprocess.run(
            command,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - platform installation failure
        raise ManifestError(f"could not execute Git: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ManifestError(
            f"Git command failed ({' '.join(arguments)}): {detail or 'unknown error'}"
        )
    return completed.stdout


def _git_root(root: Path) -> Path:
    output = _run_git(root, ["rev-parse", "--show-toplevel"])
    try:
        repository = Path(os.fsdecode(output.rstrip(b"\r\n"))).resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise ManifestError("Git returned an invalid repository root") from exc
    return repository


def _relative_to(path: Path, parent: Path, label: str) -> Path:
    try:
        return path.relative_to(parent)
    except ValueError as exc:
        raise ManifestError(f"{label} escapes the publication root: {path}") from exc


def _reject_symlink_chain(path: Path, root: Path, label: str) -> None:
    """Reject symlinks at *path* or in any component below *root*."""

    relative = _relative_to(path, root, label)
    current = root
    paths = [root]
    for part in relative.parts:
        current = current / part
        paths.append(current)
    for component in paths:
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            # The output file is allowed not to exist. Input existence is
            # checked separately by the stable reader.
            continue
        if stat.S_ISLNK(mode):
            raise ManifestError(f"{label} contains a symlink: {component}")


def _decode_git_paths(payload: bytes) -> list[PurePosixPath]:
    paths: list[PurePosixPath] = []
    for raw in payload.split(b"\0"):
        if not raw:
            continue
        decoded = os.fsdecode(raw)
        relative = PurePosixPath(decoded)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ManifestError(f"Git returned an unsafe path: {decoded!r}")
        paths.append(relative)
    return paths


def _git_publication_paths(repository: Path, publication_root: Path) -> list[PurePosixPath]:
    root_relative = _relative_to(publication_root, repository, "publication root")
    pathspec = root_relative.as_posix() if root_relative.parts else "."
    common = ["ls-files", "--full-name", "-z"]
    tracked = _decode_git_paths(
        _run_git(repository, [*common, "--cached", "--", pathspec])
    )
    publishable_untracked = _decode_git_paths(
        _run_git(
            repository,
            [*common, "--others", "--exclude-standard", "--", pathspec],
        )
    )
    return sorted(
        set(tracked).union(publishable_untracked),
        key=lambda path: path.as_posix(),
    )


def _lfs_attributes(
    repository: Path, repository_paths: Sequence[PurePosixPath]
) -> dict[PurePosixPath, bool]:
    if not repository_paths:
        return {}
    stdin = b"".join(os.fsencode(path.as_posix()) + b"\0" for path in repository_paths)
    payload = _run_git(
        repository,
        ["check-attr", "-z", "--stdin", "filter"],
        input_bytes=stdin,
    )
    fields = payload.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 3:
        raise ManifestError("git check-attr returned malformed NUL-delimited output")

    result: dict[PurePosixPath, bool] = {}
    expected = set(repository_paths)
    for index in range(0, len(fields), 3):
        path_raw, attribute_raw, value_raw = fields[index : index + 3]
        path = PurePosixPath(os.fsdecode(path_raw))
        attribute = attribute_raw.decode("ascii", errors="strict")
        value = value_raw.decode("utf-8", errors="replace")
        if attribute != "filter" or path not in expected or path in result:
            raise ManifestError("git check-attr returned unexpected path/attribute data")
        result[path] = value == "lfs"
    if set(result) != expected:
        raise ManifestError("git check-attr did not report every publication file")
    return result


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def hash_stable_file(path: Path) -> tuple[int, str]:
    """Return byte count and SHA-256, failing if the file changes while read."""

    try:
        before_path = path.lstat()
    except OSError as exc:
        raise ManifestError(f"publication file is unavailable: {path}") from exc
    if not stat.S_ISREG(before_path.st_mode):
        raise ManifestError(f"publication entry is not a regular file: {path}")

    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                byte_count += len(chunk)
                digest.update(chunk)
            after_handle = os.fstat(handle.fileno())
        after_path = path.lstat()
    except OSError as exc:
        raise ManifestError(f"could not read publication file: {path}") from exc

    identities = {
        _stat_identity(before_path),
        _stat_identity(before_handle),
        _stat_identity(after_handle),
        _stat_identity(after_path),
    }
    if len(identities) != 1 or byte_count != after_handle.st_size:
        raise ManifestError(f"publication file changed while hashing: {path}")
    return byte_count, digest.hexdigest()


def iter_bundle_files(root: Path, output: Path) -> Iterable[Path]:
    """Yield the safe Git publication set in POSIX-relative-path order."""

    root_absolute = root.absolute()
    if root_absolute.is_symlink():
        raise ManifestError(f"publication root is a symlink: {root_absolute}")
    try:
        root_resolved = root_absolute.resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"publication root is unavailable: {root_absolute}") from exc
    if not root_resolved.is_dir():
        raise ManifestError(f"publication root is not a directory: {root_resolved}")

    output_absolute = output.absolute()
    output_relative = _relative_to(output_absolute, root_absolute, "manifest output")
    _reject_symlink_chain(output_absolute, root_absolute, "manifest output")
    output_resolved = root_resolved.joinpath(*output_relative.parts)

    repository = _git_root(root_resolved)
    publication_paths = _git_publication_paths(repository, root_resolved)
    lfs_by_path = _lfs_attributes(repository, publication_paths)

    candidates: list[Path] = []
    for repository_path in publication_paths:
        candidate = repository.joinpath(*repository_path.parts)
        try:
            relative = candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise ManifestError(f"Git path escapes the publication root: {candidate}") from exc
        if candidate == output_resolved:
            continue
        _reject_symlink_chain(candidate, root_resolved, "publication file")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ManifestError(f"publication file is unavailable: {candidate}") from exc
        _relative_to(resolved, root_resolved, "publication file")
        try:
            file_stat = candidate.lstat()
        except OSError as exc:
            raise ManifestError(f"publication file is unavailable: {candidate}") from exc
        if not stat.S_ISREG(file_stat.st_mode):
            raise ManifestError(f"publication entry is not a regular file: {candidate}")
        if (
            file_stat.st_size > MAX_ORDINARY_FILE_BYTES
            and not lfs_by_path[repository_path]
        ):
            raise ManifestError(
                "ordinary publication file exceeds 10 MiB without exact "
                f"filter=lfs attribute: {relative.as_posix()} ({file_stat.st_size} bytes)"
            )
        candidates.append(candidate)

    yield from sorted(
        candidates,
        key=lambda path: path.relative_to(root_resolved).as_posix(),
    )


def _manifest_bytes(root: Path, output: Path) -> tuple[bytes, int]:
    root_resolved = root.resolve(strict=True)
    rows: list[tuple[str, int, str]] = []
    for path in iter_bundle_files(root, output):
        byte_count, digest = hash_stable_file(path)
        rows.append((path.relative_to(root_resolved).as_posix(), byte_count, digest))

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["relative_path", "bytes", "sha256"])
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8"), len(rows)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary, 0o644)
        except OSError:  # pragma: no cover - Windows ACLs can reject chmod
            pass
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def build_manifest(root: Path = ROOT, output: Path = OUTPUT) -> int:
    """Atomically write the publication manifest and return its row count."""

    payload, count = _manifest_bytes(root, output)
    _write_atomic(output.absolute(), payload)
    return count


def check_manifest(root: Path = ROOT, output: Path = OUTPUT) -> int:
    """Verify that the existing manifest exactly matches the publication set."""

    payload, count = _manifest_bytes(root, output)
    try:
        actual = output.read_bytes()
    except FileNotFoundError as exc:
        raise StaleManifestError(f"publication manifest is missing: {output}") from exc
    if actual != payload:
        raise StaleManifestError(
            f"publication manifest is stale: {output}; rebuild it without --check"
        )
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the manifest is missing or differs; do not write it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.check:
            count = check_manifest()
            print(f"verified {count} entries in {OUTPUT}")
        else:
            count = build_manifest()
            print(f"wrote {count} entries to {OUTPUT}")
    except ManifestError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
