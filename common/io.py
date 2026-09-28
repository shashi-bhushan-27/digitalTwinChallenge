"""Deterministic table I/O and content hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a table without writer-specific metadata so identical data gives identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
    table = table.replace_schema_metadata(None)
    pq.write_table(table, path, compression="zstd", write_statistics=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def content_hash(df: pd.DataFrame) -> str:
    """Hash of the table contents (column names, dtypes and values), independent of file format."""
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update("|".join(map(str, df.dtypes)).encode())
    h.update(pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values.tobytes())
    return h.hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
