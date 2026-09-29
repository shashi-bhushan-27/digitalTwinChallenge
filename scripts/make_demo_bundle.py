"""Pack the dashboard's demo cache and the trained models into one release asset.

    uv run python -m dashboard.prepare          # first: build data/demo
    uv run python scripts/make_demo_bundle.py   # writes dist/<bundle>.zip and dashboard/demo_bundle.json

The zip is byte-for-byte reproducible (sorted entries, fixed timestamps), so its SHA-256 only changes
when the content does. Fitted twins (pickles) are left out: `dashboard.fetch_demo` refits them locally.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "v0.1"
NAME = f"t2d_twin_demo_{VERSION}.zip"
URL = f"https://github.com/shashi-bhushan-27/digitalTwinChallenge/releases/download/{VERSION}/{NAME}"
SOURCES = ["data/demo", "models/artifacts"]


def files() -> list[Path]:
    out = []
    for src in SOURCES:
        for p in sorted((ROOT / src).rglob("*")):
            if p.is_file() and p.suffix != ".pkl" and p.name != ".gitkeep":
                out.append(p)
    return out


def main() -> None:
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    zpath = dist / NAME
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files():
            info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix(), date_time=(2026, 10, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes(), compresslevel=9)
    data = zpath.read_bytes()
    meta = {"version": VERSION, "file": NAME, "url": URL, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "contents": SOURCES}
    (ROOT / "dashboard" / "demo_bundle.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote dist/{NAME} ({len(data) / 1e6:.1f} MB, {len(files())} files), sha256 {meta['sha256'][:16]}...")


if __name__ == "__main__":
    main()
