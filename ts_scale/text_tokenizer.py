"""Byte-level and optional HuggingFace BPE tokenizers for ts_scale.train."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import List, Sequence


class TextTokenizerBase(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @property
    def name(self) -> str:
        return "base"

    def save_meta(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"type": self.name, "vocab_size": self.vocab_size},
                f,
                indent=2,
            )

    @staticmethod
    def load(path: str | None, meta_path: str | None = None) -> "TextTokenizerBase":
        if meta_path and os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            t = meta.get("type", "byte")
            if t == "byte":
                return ByteTokenizer()
            if t == "bpe":
                if not path or not os.path.isfile(path):
                    raise FileNotFoundError("BPE tokenizer path missing for resume")
                return BPETokenizer.load(path)
        return ByteTokenizer()


class ByteTokenizer(TextTokenizerBase):
    """UTF-8 bytes as token ids (0–255)."""

    @property
    def vocab_size(self) -> int:
        return 256

    @property
    def name(self) -> str:
        return "byte"

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="replace"))


class BPETokenizer(TextTokenizerBase):
    """Wrapper around HuggingFace `tokenizers` (ByteLevel BPE)."""

    def __init__(self, inner) -> None:
        self._tok = inner

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def name(self) -> str:
        return "bpe"

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids

    def save(self, path: str) -> None:
        self._tok.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(path))

    @classmethod
    def train_on_text(
        cls,
        text: str,
        vocab_size: int,
        save_path: str,
    ) -> "BPETokenizer":
        try:
            from tokenizers import Tokenizer
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.trainers import BpeTrainer
        except ImportError as e:
            raise ImportError(
                "BPE requires: pip install tokenizers"
            ) from e

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            special_tokens=["<pad>", "<unk>"],
        )
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as f:
            f.write(text)
            tmp = f.name
        try:
            tokenizer.train([tmp], trainer)
        finally:
            os.unlink(tmp)
        d = os.path.dirname(os.path.abspath(save_path))
        if d:
            os.makedirs(d, exist_ok=True)
        tokenizer.save(save_path)
        return cls(Tokenizer.from_file(save_path))


def encode_text(tok: TextTokenizerBase, text: str) -> List[int]:
    return tok.encode(text)


def read_text_file(path: str, max_chars: int | None = None) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        if max_chars is None:
            return f.read()
        return f.read(max_chars)
