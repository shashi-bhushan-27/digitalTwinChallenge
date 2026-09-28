"""Deterministic random streams.

Every random draw in the pipeline comes from a generator derived from one master seed plus a
list of string keys (layer, patient id, purpose). Streams are therefore independent of the
order in which layers or patients are processed: adding a patient or a new random draw in one
layer does not shift the numbers any other layer sees.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _key_words(keys: tuple[object, ...]) -> list[int]:
    digest = hashlib.sha256("/".join(str(k) for k in keys).encode("utf-8")).digest()
    return [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]


def seed_sequence(master_seed: int, *keys: object) -> np.random.SeedSequence:
    """SeedSequence for (master_seed, keys...)."""
    return np.random.SeedSequence([int(master_seed), *_key_words(keys)])


def rng(master_seed: int, *keys: object) -> np.random.Generator:
    """Independent numpy Generator for (master_seed, keys...)."""
    return np.random.Generator(np.random.PCG64(seed_sequence(master_seed, *keys)))
