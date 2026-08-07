"""Counterfactual learning-to-rank for marketplace search logs in PySpark.

The public API is re-exported lazily, so importing the package (for example
to read ``__version__``) works without pyspark installed; pyspark is only
imported when an exported name is first used.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bias import add_ips_weights, estimate_position_bias
    from .datagen import (
        generate_catalog,
        generate_impressions,
        generate_log,
        generate_searches,
    )
    from .evaluate import mrr_top_grade, ndcg_at_k
    from .features import FEATURE_COLS, build_features, item_statistics
    from .pipeline import PipelineConfig, run
    from .rank import score, train_reranker

__version__ = "0.2.2"

_EXPORTS = {
    "generate_catalog": "datagen",
    "generate_searches": "datagen",
    "generate_impressions": "datagen",
    "generate_log": "datagen",
    "estimate_position_bias": "bias",
    "add_ips_weights": "bias",
    "FEATURE_COLS": "features",
    "item_statistics": "features",
    "build_features": "features",
    "train_reranker": "rank",
    "score": "rank",
    "ndcg_at_k": "evaluate",
    "mrr_top_grade": "evaluate",
    "PipelineConfig": "pipeline",
    "run": "pipeline",
}

__all__ = [
    "FEATURE_COLS",
    "PipelineConfig",
    "__version__",
    "add_ips_weights",
    "build_features",
    "estimate_position_bias",
    "generate_catalog",
    "generate_impressions",
    "generate_log",
    "generate_searches",
    "item_statistics",
    "mrr_top_grade",
    "ndcg_at_k",
    "run",
    "score",
    "train_reranker",
]


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
