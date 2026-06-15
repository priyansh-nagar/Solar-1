"""
extract_uploads.py — unzip PRADAN day folders into /tmp/solexs_data/
=====================================================================
Run this after dragging ZIP files into the Replit file pane:

    python pipeline/scripts/extract_uploads.py

It scans the workspace root and common drop locations for files matching
AL1_SLX_L1_YYYYMMDD_*.zip, extracts each into /tmp/solexs_data/, and
reports what is now ready for aggregate_days.py.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

# ── Where to look for uploaded ZIPs ───────────────────────────────────────
SEARCH_ROOTS = [
    Path("/home/runner/workspace"),       # Replit file-pane drops land here
    Path("/home/runner/workspace/attached_assets"),
    Path("/tmp"),
]
DEST = Path("/tmp/solexs_data")
PATTERN = re.compile(r"AL1_SLX_L1_\d{8}.*\.zip", re.IGNORECASE)


def find_zips() -> list[Path]:
    found: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.iterdir():
            if p.is_file() and PATTERN.match(p.name):
                found.append(p)
    return sorted(found)


def extract(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        top_dirs = {Path(n).parts[0] for n in zf.namelist() if "/" in n}
        zf.extractall(dest)
    if top_dirs:
        extracted = dest / sorted(top_dirs)[0]
        print(f"  → extracted to {extracted}")
        return extracted
    return dest


def main() -> None:
    zips = find_zips()
    if not zips:
        print(
            "No AL1_SLX_L1_*.zip files found in workspace.\n"
            "Drag the PRADAN ZIP files into the Replit file pane (left sidebar),\n"
            "then re-run this script."
        )
        sys.exit(1)

    print(f"Found {len(zips)} ZIP file(s):\n")
    for z in zips:
        print(f"  {z.name}  ({z.stat().st_size / 1_048_576:.1f} MB)")
        try:
            extract(z, DEST)
        except Exception as e:
            print(f"  ✗ Failed to extract {z.name}: {e}")

    print(f"\nContents of {DEST}:")
    for d in sorted(DEST.iterdir()):
        if d.is_dir():
            n_files = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"  {d.name}/  ({n_files} files)")

    print("\nAll done — run:  python pipeline/scripts/aggregate_days.py")


if __name__ == "__main__":
    main()
