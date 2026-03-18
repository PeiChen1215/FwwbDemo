from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
LOCAL_TS3L_ROOT = BASE_DIR / "TabularS3L-main" / "TabularS3L-main"


def require_tabulars3l_runtime():
    """Load local TabularS3L runtime dependencies."""
    local_root = LOCAL_TS3L_ROOT.resolve()
    if local_root.exists() and str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

    try:
        import torch
        from torch.utils.data import DataLoader
        import pytorch_lightning as pl

        ts3l_module = importlib.import_module("ts3l")
        SCARFLightning = importlib.import_module("ts3l.pl_modules").SCARFLightning
        SCARFConfig = importlib.import_module("ts3l.utils.scarf_utils").SCARFConfig
        SCARFDataset = importlib.import_module("ts3l.utils.scarf_utils").SCARFDataset
        TS3LDataModule = importlib.import_module("ts3l.utils").TS3LDataModule
        FTEmbeddingConfig = importlib.import_module("ts3l.utils.embedding_utils").FTEmbeddingConfig
        TransformerBackboneConfig = importlib.import_module("ts3l.utils.backbone_utils").TransformerBackboneConfig
        get_category_cardinality = importlib.import_module("ts3l.utils").get_category_cardinality
    except Exception as exc:
        raise RuntimeError(
            "TabularS3L runtime is not ready. Make sure local source and dependencies are available. "
            f"Original import error: {exc}"
        ) from exc

    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "pl": pl,
        "ts3l": ts3l_module,
        "SCARFLightning": SCARFLightning,
        "SCARFConfig": SCARFConfig,
        "SCARFDataset": SCARFDataset,
        "TS3LDataModule": TS3LDataModule,
        "FTEmbeddingConfig": FTEmbeddingConfig,
        "TransformerBackboneConfig": TransformerBackboneConfig,
        "get_category_cardinality": get_category_cardinality,
        "local_root": local_root,
    }


def build_scarf_transformer_model(
    runtime,
    train_df,
    continuous_cols,
    category_cols,
    latent_dim,
    corruption_rate=0.3,
):
    """Build SCARF with feature tokenizer + transformer backbone."""
    FTEmbeddingConfig = runtime["FTEmbeddingConfig"]
    TransformerBackboneConfig = runtime["TransformerBackboneConfig"]
    SCARFConfig = runtime["SCARFConfig"]
    SCARFLightning = runtime["SCARFLightning"]
    get_category_cardinality = runtime["get_category_cardinality"]

    input_dim = len(category_cols) + len(continuous_cols)
    cat_cardinality = get_category_cardinality(train_df, category_cols)

    embedding_config = FTEmbeddingConfig(
        input_dim=input_dim,
        emb_dim=latent_dim,
        cont_nums=len(continuous_cols),
        cat_cardinality=cat_cardinality,
        required_token_dim=2,
    )
    backbone_config = TransformerBackboneConfig(
        d_model=embedding_config.emb_dim,
        hidden_dim=128,
        encoder_depth=3,
        n_head=4,
        ffn_factor=2.0,
    )

    scarf_config = SCARFConfig(
        task="classification",
        embedding_config=embedding_config,
        backbone_config=backbone_config,
        output_dim=2,
        loss_fn="CrossEntropyLoss",
        metric="f1_score",
        corruption_rate=corruption_rate,
    )

    model = SCARFLightning(scarf_config)
    return model, scarf_config


def extract_embeddings(torch, data_loader_cls, dataset, lightning_module, batch_size=512) -> np.ndarray:
    """Extract latent embeddings from the trained SCARF encoder."""
    loader = data_loader_cls(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    lightning_module.eval()
    model = lightning_module.model

    with torch.no_grad():
        for batch in loader:
            x_batch = batch["input"] if isinstance(batch, dict) else batch[0]
            embedded = model.embedding_module(x_batch)
            latent = model.encoder(embedded)
            if latent.dim() == 3:
                latent = latent.mean(dim=1)
            embeddings.append(latent.detach().cpu().numpy())

    return np.vstack(embeddings)


def run_hdbscan_clustering(embeddings: np.ndarray, min_cluster_ratio: float):
    """Run HDBSCAN and mark `-1` labels as outliers."""
    try:
        import hdbscan
    except Exception as exc:
        raise RuntimeError(
            "The `hdbscan` package is not installed in the current environment. "
            "Install it first with `pip install hdbscan` and rerun `train.py`."
        ) from exc

    min_cluster_size = max(10, int(len(embeddings) * min_cluster_ratio))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric="euclidean",
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    outlier_mask = cluster_labels == -1
    inlier_mask = ~outlier_mask
    return cluster_labels, inlier_mask, outlier_mask
