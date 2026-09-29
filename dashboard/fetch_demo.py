"""Download the demo cache and trained models from the GitHub release, so the dashboard runs without
regenerating the dataset and retraining (which takes about 1.5 hours).

    uv run python -m dashboard.fetch_demo            # download, verify, extract, fit the 3 demo twins (~1 min)
    uv run streamlit run dashboard/app.py

The bundle holds only synthetic data (data/demo: 60 held-out test patients) and LightGBM model files
(models/artifacts). Its SHA-256 is checked against dashboard/demo_bundle.json before anything is written.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from common import config

ROOT = config.REPO_ROOT
INFO = Path(__file__).with_name("demo_bundle.json")


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def extract(data: bytes, expected_sha256: str) -> list[str]:
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise SystemExit(f"checksum mismatch: got {digest}, expected {expected_sha256}; nothing was extracted")
    root = ROOT.resolve()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        for n in names:
            target = (root / n).resolve()
            rel = target.relative_to(root).as_posix() if root in target.parents else ""
            if not rel.startswith(("data/demo/", "models/artifacts/")):
                raise SystemExit(f"unexpected path in bundle: {n}")
        z.extractall(root)
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", help="use a local copy of the bundle instead of downloading it")
    ap.add_argument("--no-fit", action="store_true", help="skip fitting the demo scenes' twins (the dashboard fits them on demand)")
    args = ap.parse_args()

    info = json.loads(INFO.read_text(encoding="utf-8"))
    if args.zip:
        data = Path(args.zip).read_bytes()
    else:
        print(f"downloading {info['url']} ({info['bytes'] / 1e6:.0f} MB)")
        data = download(info["url"])
    names = extract(data, info["sha256"])
    print(f"checksum ok; extracted {len(names)} files into data/demo and models/artifacts")

    if not args.no_fit:
        from dashboard import service

        dd = service.load()
        for pid in sorted({s["patient_id"] for s in dd.scenes}):
            print(f"personalising the twin for demo patient {pid} (about 20 s)")
            service.twin_for(dd, pid)
    print("done: uv run streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
