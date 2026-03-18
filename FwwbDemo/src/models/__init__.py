"""
Model definitions for SSL and risk prediction.
"""
from .losses import NTXentLoss
from .scarf import (
    require_tabulars3l_runtime,
    build_scarf_transformer_model,
    extract_embeddings,
    run_hdbscan_clustering,
)
from .transformer import TransformerEncoder

try:
    from .scarf_lightning import SCARFLightning
except ImportError:
    pass  # TabularS3L not available

__all__ = [
    "NTXentLoss",
    "TransformerEncoder",
    "require_tabulars3l_runtime",
    "build_scarf_transformer_model",
    "extract_embeddings",
    "run_hdbscan_clustering",
    "SCARFLightning",
]
