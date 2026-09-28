"""Download the real open datasets used ONLY for calibration and validation.

    uv run python -m validation.real_data.download            # both datasets
    uv run python -m validation.real_data.download shanghai   # one dataset

* Shanghai T2DM (Zhao et al., Scientific Data 2023), figshare article 21600933,
  licence CC BY 4.0, one zip of about 3.7 MB.
* CGMacros v1.0.0 (PhysioNet, open access), licence CC BY-NC-SA 4.0. The archive is about
  657 MB, mostly meal photos, so only the CSV members are fetched with HTTP range requests.

Files land in data/real/<dataset>/ (gitignored). Checksums are verified where the publisher
provides them. If a download is blocked, fetch the files manually from the URLs below and unzip
them into the same folders; the loaders only need the CSV/XLSX files.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from common.config import REPO_ROOT

REAL_DIR = REPO_ROOT / "data" / "real"

SHANGHAI_ARTICLE_API = "https://api.figshare.com/v2/articles/21600933"
CGMACROS_BASE = "https://physionet.org/files/cgmacros/1.0.0/"
CGMACROS_ZIP = "CGMacros_dateshifted365.zip"
CGMACROS_SIDE_FILES = ["DataDictionary_Bio.csv", "DataDictionary_CGMacros-00X.csv", "LICENSE.txt"]

USER_AGENT = "t2d-digital-twin-validation/0.1 (research; non-commercial)"


def _get(url: str, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, so zipfile can read single members."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url = url
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self._cache: dict[int, bytes] = {}
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def _block(self, idx: int) -> bytes:
        if idx not in self._cache:
            start = idx * self.block
            end = min(start + self.block, self.size) - 1
            data = _get(self.url, {"Range": f"bytes={start}-{end}"})
            self.bytes_fetched += len(data)
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[idx] = data
        return self._cache[idx]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            idx, off = divmod(self.pos, self.block)
            chunk = self._block(idx)[off : off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def download_shanghai(dest: Path = REAL_DIR / "shanghai") -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    meta = json.loads(_get(SHANGHAI_ARTICLE_API))
    (dest / "figshare_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for f in meta["files"]:
        target = dest / f["name"]
        if not target.exists():
            print(f"[shanghai] downloading {f['name']} ({f['size'] / 1e6:.1f} MB)")
            data = _get(f["download_url"])
            md5 = hashlib.md5(data).hexdigest()
            if f.get("computed_md5") and md5 != f["computed_md5"]:
                raise RuntimeError(f"MD5 mismatch for {f['name']}: {md5} != {f['computed_md5']}")
            target.write_bytes(data)
        if target.suffix == ".zip":
            with zipfile.ZipFile(target) as zf:
                zf.extractall(dest)
    print(f"[shanghai] licence: {meta.get('license', {}).get('name')} -> {dest}")
    return dest


def download_cgmacros(dest: Path = REAL_DIR / "cgmacros") -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name in CGMACROS_SIDE_FILES:
        target = dest / name
        if not target.exists():
            target.write_bytes(_get(CGMACROS_BASE + name))
    remote = HttpRangeFile(CGMACROS_BASE + CGMACROS_ZIP)
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 16)) as zf:
        members = [m for m in zf.infolist() if m.filename.lower().endswith(".csv")]
        print(f"[cgmacros] {len(members)} CSV members in a {remote.size / 1e6:.0f} MB archive")
        for m in members:
            target = dest / m.filename
            if target.exists() and target.stat().st_size == m.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(m))
    print(f"[cgmacros] fetched {remote.bytes_fetched / 1e6:.1f} MB via range requests -> {dest}")
    return dest


def main(argv: list[str]) -> None:
    targets = argv or ["shanghai", "cgmacros"]
    for t in targets:
        {"shanghai": download_shanghai, "cgmacros": download_cgmacros}[t]()


if __name__ == "__main__":
    main(sys.argv[1:])
