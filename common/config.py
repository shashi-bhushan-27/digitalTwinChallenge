"""YAML configuration with provenance tags.

Convention used in every file under configs/:

* A mapping with a ``source`` key is a provenance record. ``verified: false`` means the value
  was taken from literature (or is an informed assumption) and has not yet been checked against
  the source; it must be quoted as "to verify" in docs (BUILD_PROMPT rule 5).
* A provenance record that also has a ``value`` key is replaced by that value when the config
  is loaded, so code reads plain numbers.
* Plain numbers without a record are design choices, explained in docs/decisions.md.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class ProvenanceRecord:
    path: str
    source: str
    verified: bool
    value: Any
    details: str = ""


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def collect_provenance(node: Any, path: str = "") -> list[ProvenanceRecord]:
    """All provenance records in a raw (unresolved) config tree."""
    found: list[ProvenanceRecord] = []
    if isinstance(node, dict):
        if "source" in node:
            found.append(
                ProvenanceRecord(
                    path=path or "<root>",
                    source=str(node.get("source", "")),
                    verified=bool(node.get("verified", False)),
                    value=node.get("value"),
                    details=", ".join(f"{k}={v}" for k, v in node.items()
                                      if k not in ("source", "verified", "value") and not isinstance(v, dict | list)),
                )
            )
        for key, val in node.items():
            if key in ("source", "verified", "value"):
                continue
            found.extend(collect_provenance(val, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, list):
        for i, val in enumerate(node):
            found.extend(collect_provenance(val, f"{path}[{i}]"))
    return found


def resolve(node: Any) -> Any:
    """Replace {value, source, verified} records by their value; drop tags elsewhere."""
    if isinstance(node, dict):
        if "source" in node and "value" in node:
            return resolve(node["value"])
        return {k: resolve(v) for k, v in node.items() if k not in ("source", "verified")}
    if isinstance(node, list):
        return [resolve(v) for v in node]
    return node


def load_raw(name: str, config_dir: Path = CONFIG_DIR) -> dict:
    path = Path(name)
    if not path.suffix:
        path = path.with_suffix(".yaml")
    if not path.is_absolute():
        path = config_dir / path
    return _read_yaml(path)


def load(name: str, overlays: list[dict] | None = None, config_dir: Path = CONFIG_DIR) -> dict:
    raw = load_raw(name, config_dir)
    for overlay in overlays or []:
        raw = deep_merge(raw, overlay)
    return resolve(raw)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
