#!/usr/bin/env python3
"""Check repository-relative Markdown links in the public tree."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_TOP_LEVEL = {
    "_local",
    "00_overview",
    "01_sources",
    "02_experiments",
    "03_evidence",
    "04_analysis",
    "05_paper",
    "99_archive",
}
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def markdown_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*.md")
        if path.relative_to(ROOT).parts[0] not in EXCLUDED_TOP_LEVEL
    ]


def main() -> int:
    broken: list[str] = []
    checked = 0
    for document in markdown_files():
        text = document.read_text(encoding="utf-8-sig")
        for match in LINK_RE.finditer(text):
            target = match.group(1).strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:", "legacy://")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            checked += 1
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                line = text.count("\n", 0, match.start()) + 1
                broken.append(f"{document.relative_to(ROOT)}:{line} -> {target}")

    if broken:
        print("Broken Markdown links:")
        print("\n".join(f"- {item}" for item in broken))
        return 1
    print(f"Markdown links OK: {checked} local link(s) in {len(markdown_files())} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
