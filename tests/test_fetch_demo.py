"""The release-bundle download refuses bad checksums and unexpected paths before writing anything."""

import hashlib
import io
import zipfile

import pytest

from dashboard import fetch_demo


def _zip(names: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in names:
            z.writestr(n, "x")
    return buf.getvalue()


def test_checksum_mismatch_extracts_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_demo, "ROOT", tmp_path)
    data = _zip(["data/demo/scenes.json"])
    with pytest.raises(SystemExit, match="checksum mismatch"):
        fetch_demo.extract(data, "0" * 64)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("bad", ["../evil.txt", "configs/physiology.yaml", "data/demo/../../evil.txt"])
def test_unexpected_paths_are_refused(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(fetch_demo, "ROOT", tmp_path)
    data = _zip(["data/demo/scenes.json", bad])
    with pytest.raises(SystemExit, match="unexpected path"):
        fetch_demo.extract(data, hashlib.sha256(data).hexdigest())
    assert not (tmp_path / "data").exists()


def test_valid_bundle_is_extracted(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_demo, "ROOT", tmp_path)
    data = _zip(["data/demo/scenes.json", "models/artifacts/fusion/meta.json"])
    names = fetch_demo.extract(data, hashlib.sha256(data).hexdigest())
    assert len(names) == 2 and (tmp_path / "models" / "artifacts" / "fusion" / "meta.json").exists()
