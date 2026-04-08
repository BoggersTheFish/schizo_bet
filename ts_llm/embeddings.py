"""
Pluggable token embeddings for TS-LLM: deterministic hash vectors (zero-deps),
Gaussian random, pretrained Sentence-Transformers (optional), or explicit tables.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union


def normalize_vector(v: Sequence[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [float(x) / s for x in v]


def hash_embedding(text: str, dim: int) -> List[float]:
    """Deterministic unit vector from token string (stable across runs)."""
    raw: List[float] = []
    seed = text.encode("utf-8")
    i = 0
    while len(raw) < dim:
        chunk = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        for b in chunk:
            raw.append((b / 127.5) - 1.0)
        i += 1
    return normalize_vector(raw[:dim])


class EmbeddingBackend(ABC):
    dimension: int

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass


class RandomEmbeddingBackend(EmbeddingBackend):
    def __init__(self, dim: int, rng: random.Random) -> None:
        self.dimension = dim
        self._rng = rng

    def embed(self, text: str) -> List[float]:
        v = [self._rng.gauss(0, 1) for _ in range(self.dimension)]
        return normalize_vector(v)


class HashEmbeddingBackend(EmbeddingBackend):
    def __init__(self, dim: int) -> None:
        self.dimension = dim

    def embed(self, text: str) -> List[float]:
        return hash_embedding(text, self.dimension)


class DictEmbeddingBackend(EmbeddingBackend):
    """Known tokens from a table; OOV uses fallback backend."""

    def __init__(
        self,
        table: Dict[str, Sequence[float]],
        dim: int,
        fallback: EmbeddingBackend,
    ) -> None:
        self._table = {k: normalize_vector(v) for k, v in table.items()}
        self.dimension = dim
        self._fallback = fallback

    def update(self, table: Dict[str, Sequence[float]]) -> None:
        for k, v in table.items():
            self._table[str(k)] = normalize_vector(v)

    def embed(self, text: str) -> List[float]:
        if text in self._table:
            return list(self._table[text])
        return self._fallback.embed(text)


class SentenceTransformerBackend(EmbeddingBackend):
    """Real semantic embeddings via ``sentence-transformers`` (install separately)."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "embedding_mode='sentence_transformer' requires: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> List[float]:
        import numpy as np

        v = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        arr = np.asarray(v, dtype=float).ravel()
        return [float(x) for x in arr]


def build_backend(
    mode: str,
    dim: int,
    rng: random.Random,
    sentence_transformer_model: str,
    embedding_table: Optional[Dict[str, Sequence[float]]] = None,
) -> Optional[EmbeddingBackend]:
    mode = (mode or "off").lower().strip()
    if mode in ("off", "none", ""):
        return None
    if mode == "random":
        if dim <= 0:
            raise ValueError("embedding_mode='random' requires embedding_dim > 0")
        return RandomEmbeddingBackend(dim, rng)
    if mode == "hash":
        return HashEmbeddingBackend(dim)
    if mode in ("dict", "table"):
        if not embedding_table:
            raise ValueError("embedding_mode=dict requires embedding_table")
        fb = HashEmbeddingBackend(dim)
        return DictEmbeddingBackend(embedding_table, dim, fb)
    if mode in ("sentence_transformer", "st", "minilm"):
        name = sentence_transformer_model or "all-MiniLM-L6-v2"
        return SentenceTransformerBackend(name)
    raise ValueError(
        f"Unknown embedding_mode={mode!r}; use off, random, hash, dict, sentence_transformer"
    )


def load_embedding_json(path: Union[str, bytes]) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("JSON root must be an object token -> vector")
    out: Dict[str, List[float]] = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)):
            out[str(k)] = [float(x) for x in v]
        else:
            raise ValueError(f"Value for {k!r} must be a list of floats")
    return out
