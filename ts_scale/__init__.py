"""
Scaled TS-LM: tension-graph backbone, MoE, VQ Schrödinger, eval hooks.

``TSLanguageModel`` is loaded lazily so ``import ts_scale`` does not require
``torch`` until you access the class.

Install: ``pip install torch`` then ``python -m ts_scale.train``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ts_scale.config import TSLMConfig

__all__ = ["TSLMConfig", "TSLanguageModel"]


def __getattr__(name: str) -> Any:
    if name == "TSLanguageModel":
        from ts_scale.model import TSLanguageModel

        return TSLanguageModel
    raise AttributeError(name)


if TYPE_CHECKING:
    from ts_scale.model import TSLanguageModel
