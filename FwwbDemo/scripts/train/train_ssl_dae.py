from __future__ import annotations

from pathlib import Path
import importlib
import json
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "prepared" / "06_tabulars3l"
OUTPUT_DIR = BASE_DIR / "outputs" / "results"
LOCAL_TS3L_ROOT = BASE_DIR / "TabularS3L-main" / "TabularS3L-main"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
LATENT_DIM = 16
PRETRAIN_EPOCHS = 200
BATCH_SIZE = 256
K_SEARCH_VALUES = range(2, 7)
MIN_CLUSTER_RATIO = 0.03
OUTLIER_DISTANCE_PERCENTILE = 99.5
LOG1P_COLS = [
    "internet_hours_sum",
    "internet_hours_avg_per_month",
]
SIGNED_LOG1P_COLS = [
    "internet_diff_mean",
]


def prefer_local_tabulars3l() -> Path | None:
    local_root = LOCAL_TS3L_ROOT.resolve()
    if local_root.exists():
        local_root_str = str(local_root)
        if local_root_str not in sys.path:
            sys.path.insert(0, local_root_str)
        return local_root
    return None


def require_tabulars3l_runtime():
    local_root = prefer_local_tabulars3l()

    try:
        import torch
        from torch.utils.data import DataLoader
        import pytorch_lightning as pl

        ts3l_module = importlib.import_module("ts3l")
        DAELightning = importlib.import_module("ts3l.pl_modules").DAELightning
        DAEConfig = importlib.import_module("ts3l.utils.dae_utils").DAEConfig
        DAEDataset = importlib.import_module("ts3l.utils.dae_utils").DAEDataset
        DAECollateFN = importlib.import_module("ts3l.utils.dae_utils").DAECollateFN
        TS3LDataModule = importlib.import_module("ts3l.utils").TS3LDataModule
        IdentityEmbeddingConfig = importlib.import_module("ts3l.utils.embedding_utils").IdentityEmbeddingConfig
        MLPBackboneConfig = importlib.import_module("ts3l.utils.backbone_utils").MLPBackboneConfig
        get_category_cardinality = importlib.import_module("ts3l.utils").get_category_cardinality
    except Exception as exc:
        source_hint = (
            f"Local source was checked at: {local_root}"
            if local_root is not None
            else "Local TabularS3L source directory was not found."
        )
        raise RuntimeError(
            "Missing runtime dependencies for TabularS3L local integration.\n"
            f"{source_hint}\n"
            "Please make sure your IDE environment has at least:\n"
            "  pip install torch torchvision torchaudio\n"
            "  pip install pytorch-lightning torchmetrics\n"
            "If you want to install the downloaded repo itself, run in the repo root:\n"
            "  pip install -e .\n"
            f"Original import error: {exc}"
        ) from exc

    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "pl": pl,
        "ts3l_module": ts3l_module,
        "DAELightning": DAELightning,
        "DAEConfig": DAEConfig,
        "DAEDataset": DAEDataset,
        "DAECollateFN": DAECollateFN,
        "TS3LDataModule": TS3LDataModule,
        "IdentityEmbeddingConfig": IdentityEmbeddingConfig,
        "MLPBackboneConfig": MLPBackboneConfig,
        "get_category_cardinality": get_category_cardinality,
        "local_root": local_root,
    }


def load_metadata() -> dict:
    return json.loads((DATA_DIR / "a14_tabulars3l_metadata.json").read_text(encoding="utf-8"))


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def preprocess_for_ts3l(
    df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    out = df[feature_cols].copy()

    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].median())

    for col in LOG1P_COLS:
        if col in out.columns:
            out[col] = np.log1p(np.clip(out[col], a_min=0, a_max=None))

    for col in SIGNED_LOG1P_COLS:
        if col in out.columns:
            out[col] = np.sign(out[col]) * np.log1p(np.abs(out[col]))

    for col in categorical_cols:
        out[col] = out[col].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN")
        out[col] = pd.Categorical(out[col]).codes.astype(np.int64)

    ordered_cols = categorical_cols + numeric_cols
    return out[ordered_cols]


def choose_best_k(
    embeddings: np.ndarray,
    k_values: Iterable[int],
    min_cluster_ratio: float,
) -> tuple[int, list[dict[str, float | int | bool]], np.ndarray]:
    evaluations = []
    best_valid_k = None
    best_valid_score = -1.0
    best_fallback_k = None
    best_fallback_score = -1.0
    best_labels = None
    min_cluster_size_threshold = max(1, int(np.ceil(len(embeddings) * min_cluster_ratio)))

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        _, counts = np.unique(labels, return_counts=True)
        min_cluster_size = int(counts.min())
        max_cluster_size = int(counts.max())
        is_valid = min_cluster_size >= min_cluster_size_threshold
        evaluations.append(
            {
                "k": k,
                "silhouette_score": round(float(score), 6),
                "min_cluster_size": min_cluster_size,
                "max_cluster_size": max_cluster_size,
                "valid_clustering": is_valid,
            }
        )

        if score > best_fallback_score:
            best_fallback_score = score
            best_fallback_k = k

        if is_valid and score > best_valid_score:
            best_valid_score = score
            best_valid_k = k
            best_labels = labels

    if best_valid_k is not None and best_labels is not None:
        return best_valid_k, evaluations, best_labels

    if best_fallback_k is None:
        raise ValueError("Failed to choose best k.")

    fallback_model = KMeans(n_clusters=best_fallback_k, random_state=RANDOM_STATE, n_init=20)
    fallback_labels = fallback_model.fit_predict(embeddings)
    return best_fallback_k, evaluations, fallback_labels


def extract_embeddings_from_dae(torch, data_loader_cls, dataset, dae_lightning_module, batch_size: int = 512) -> np.ndarray:
    loader = data_loader_cls(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []

    dae_lightning_module.eval()
    model = dae_lightning_module.model

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x_batch = batch[0]
            else:
                x_batch = batch

            embedded = model.embedding_module(x_batch)
            latent = model.encoder(embedded)
            embeddings.append(latent.detach().cpu().numpy())

    return np.vstack(embeddings)


def detect_outliers_by_distance(
    embeddings: np.ndarray,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    center = embeddings.mean(axis=0, keepdims=True)
    dist = np.linalg.norm(embeddings - center, axis=1)
    threshold = float(np.percentile(dist, percentile))
    outlier_mask = dist > threshold
    return outlier_mask, dist, threshold


def save_cluster_outputs(
    last_semester_df: pd.DataFrame,
    raw_embeddings: np.ndarray,
    scaled_embeddings: np.ndarray,
    inlier_mask: np.ndarray,
    outlier_mask: np.ndarray,
    distance_to_center: np.ndarray,
    outlier_threshold: float,
    best_k: int,
    k_search: list[dict[str, float | int | bool]],
    cluster_labels: np.ndarray,
) -> None:
    embedding_cols = [f"emb_{i:02d}" for i in range(raw_embeddings.shape[1])]
    embedding_scaled_cols = [f"emb_scaled_{i:02d}" for i in range(scaled_embeddings.shape[1])]

    student_cluster_df = last_semester_df.copy()
    if "risk_event_current" not in student_cluster_df.columns:
        student_cluster_df["risk_event_current"] = np.nan
    for idx, col in enumerate(embedding_cols):
        student_cluster_df[col] = raw_embeddings[:, idx]
    for idx, col in enumerate(embedding_scaled_cols):
        student_cluster_df[col] = scaled_embeddings[:, idx]
    full_cluster_labels = np.full(len(student_cluster_df), -1, dtype=int)
    full_cluster_labels[inlier_mask] = cluster_labels
    student_cluster_df["cluster"] = full_cluster_labels
    student_cluster_df["is_outlier"] = outlier_mask.astype(int)
    student_cluster_df["embedding_distance_to_center"] = distance_to_center
    student_cluster_df["outlier_distance_threshold"] = outlier_threshold
    student_cluster_df["risk_label_next_term_numeric"] = pd.to_numeric(
        student_cluster_df["risk_label_next_term"], errors="coerce"
    )
    student_cluster_df.to_csv(
        OUTPUT_DIR / "tabulars3l_student_last_semester_clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cluster_summary = (
        student_cluster_df.groupby("cluster")
        .agg(
            student_count=("student_id", "count"),
            outlier_count=("is_outlier", "sum"),
            avg_score_mean=("avg_score", "mean"),
            fail_course_count_mean=("fail_course_count", "mean"),
            internet_hours_sum_mean=("internet_hours_sum", "mean"),
            online_learning_bfb_snapshot_mean=("online_learning_bfb_snapshot", "mean"),
            physical_test_score_mean=("physical_test_score", "mean"),
            current_risk_rate=("risk_event_current", "mean"),
            next_term_risk_rate=("risk_label_next_term_numeric", "mean"),
        )
        .reset_index()
    )
    cluster_summary = cluster_summary.sort_values("cluster").reset_index(drop=True)
    cluster_summary.to_csv(
        OUTPUT_DIR / "tabulars3l_student_cluster_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    inlier_cluster_summary = (
        student_cluster_df[student_cluster_df["is_outlier"] == 0]
        .groupby("cluster")
        .agg(
            student_count=("student_id", "count"),
            avg_score_mean=("avg_score", "mean"),
            fail_course_count_mean=("fail_course_count", "mean"),
            internet_hours_sum_mean=("internet_hours_sum", "mean"),
            online_learning_bfb_snapshot_mean=("online_learning_bfb_snapshot", "mean"),
            physical_test_score_mean=("physical_test_score", "mean"),
            current_risk_rate=("risk_event_current", "mean"),
            next_term_risk_rate=("risk_label_next_term_numeric", "mean"),
        )
        .reset_index()
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    inlier_cluster_summary.to_csv(
        OUTPUT_DIR / "tabulars3l_inlier_cluster_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(k_search).to_csv(
        OUTPUT_DIR / "tabulars3l_cluster_k_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(scaled_embeddings)
    pca_df = student_cluster_df[
        ["student_id", "school_year", "semester", "cluster", "is_outlier", "risk_event_current", "risk_label_next_term_numeric", "embedding_distance_to_center"]
    ].copy()
    pca_df["pca_1"] = reduced[:, 0]
    pca_df["pca_2"] = reduced[:, 1]
    pca_df.to_csv(OUTPUT_DIR / "tabulars3l_student_cluster_pca_coords.csv", index=False, encoding="utf-8-sig")

    cluster_categories = pd.Categorical(student_cluster_df["cluster"].astype(str))
    cluster_color_codes = cluster_categories.codes
    cluster_labels_for_legend = list(cluster_categories.categories)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_color_codes, s=18, alpha=0.7, cmap="tab10")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("TabularS3L DAE Student Embedding Clusters (Full)")
    handles, _ = scatter.legend_elements()
    plt.legend(handles, cluster_labels_for_legend, title="Cluster", loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tabulars3l_student_cluster_pca.png", dpi=180)
    plt.close()

    x_low, x_high = np.percentile(reduced[:, 0], [1, 99])
    y_low, y_high = np.percentile(reduced[:, 1], [1, 99])
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_color_codes, s=18, alpha=0.7, cmap="tab10")
    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("TabularS3L DAE Student Embedding Clusters (Zoomed 1%-99%)")
    handles, _ = scatter.legend_elements()
    plt.legend(handles, cluster_labels_for_legend, title="Cluster", loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tabulars3l_student_cluster_pca_zoom.png", dpi=180)
    plt.close()

    risk_color = student_cluster_df["risk_label_next_term_numeric"].fillna(-1).to_numpy()
    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap("coolwarm", 3)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=risk_color, s=18, alpha=0.75, cmap=cmap)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("PCA Space Colored by Next-Term Risk (-1 unlabeled, 0 safe, 1 risk)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tabulars3l_student_risk_pca.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced[:, 0], reduced[:, 1], c="#4c78a8", s=18, alpha=0.45, label="Inlier")
    if outlier_mask.any():
        plt.scatter(
            reduced[outlier_mask, 0],
            reduced[outlier_mask, 1],
            c="#e45756",
            s=42,
            alpha=0.95,
            marker="x",
            label="Outlier",
        )
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("PCA Space with Outlier Highlight")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tabulars3l_outlier_pca.png", dpi=180)
    plt.close()

    cluster_sizes = (
        student_cluster_df.groupby("cluster")
        .size()
        .reset_index(name="student_count")
        .sort_values("cluster")
    )
    plt.figure(figsize=(8, 5))
    plt.bar(cluster_sizes["cluster"].astype(str), cluster_sizes["student_count"], color="#34699A")
    plt.xlabel("Cluster")
    plt.ylabel("Student Count")
    plt.title("Cluster Size Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tabulars3l_cluster_size_bar.png", dpi=180)
    plt.close()

    if len(scaled_embeddings) >= 200:
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(scaled_embeddings) // 80)),
            learning_rate="auto",
            init="pca",
            random_state=RANDOM_STATE,
        )
        tsne_reduced = tsne.fit_transform(scaled_embeddings)
        tsne_df = student_cluster_df[
            ["student_id", "school_year", "semester", "cluster", "is_outlier", "risk_event_current", "risk_label_next_term_numeric", "embedding_distance_to_center"]
        ].copy()
        tsne_df["tsne_1"] = tsne_reduced[:, 0]
        tsne_df["tsne_2"] = tsne_reduced[:, 1]
        tsne_df.to_csv(OUTPUT_DIR / "tabulars3l_student_cluster_tsne_coords.csv", index=False, encoding="utf-8-sig")

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(tsne_reduced[:, 0], tsne_reduced[:, 1], c=cluster_color_codes, s=18, alpha=0.7, cmap="tab10")
        plt.xlabel("t-SNE-1")
        plt.ylabel("t-SNE-2")
        plt.title("TabularS3L DAE Student Embedding Clusters (t-SNE)")
        handles, _ = scatter.legend_elements()
        plt.legend(handles, cluster_labels_for_legend, title="Cluster", loc="best")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "tabulars3l_student_cluster_tsne.png", dpi=180)
        plt.close()

    outlier_df = student_cluster_df[
        ["student_id", "school_year", "semester", "cluster", "is_outlier", "college", "major", "avg_score", "fail_course_count", "internet_hours_sum", "online_learning_bfb_snapshot", "risk_label_next_term_numeric", "embedding_distance_to_center"]
    ].copy()
    outlier_df = outlier_df.sort_values("embedding_distance_to_center", ascending=False)
    outlier_df.head(50).to_csv(OUTPUT_DIR / "tabulars3l_top_outliers.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    runtime = require_tabulars3l_runtime()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch = runtime["torch"]
    DataLoader = runtime["DataLoader"]
    pl = runtime["pl"]
    DAELightning = runtime["DAELightning"]
    DAEConfig = runtime["DAEConfig"]
    DAEDataset = runtime["DAEDataset"]
    DAECollateFN = runtime["DAECollateFN"]
    TS3LDataModule = runtime["TS3LDataModule"]
    IdentityEmbeddingConfig = runtime["IdentityEmbeddingConfig"]
    MLPBackboneConfig = runtime["MLPBackboneConfig"]
    get_category_cardinality = runtime["get_category_cardinality"]

    metadata = load_metadata()
    feature_cols = metadata["feature_cols"]
    numeric_cols = metadata["numeric_cols"]
    categorical_cols = metadata["categorical_cols"]

    ordered_feature_cols = metadata.get("ordered_feature_cols", categorical_cols + numeric_cols)
    input_dim = len(ordered_feature_cols)

    ssl_pool_df = load_split("ssl_pool")
    last_semester_df = load_split("last_semester_for_clustering")

    ssl_features = preprocess_for_ts3l(ssl_pool_df, feature_cols, numeric_cols, categorical_cols)
    last_features = preprocess_for_ts3l(last_semester_df, feature_cols, numeric_cols, categorical_cols)

    split_point = int(len(ssl_features) * 0.9)
    ssl_train_df = ssl_features.iloc[:split_point].reset_index(drop=True)
    ssl_valid_df = ssl_features.iloc[split_point:].reset_index(drop=True)

    cat_cardinality = get_category_cardinality(ssl_features, categorical_cols)

    embedding_config = IdentityEmbeddingConfig(input_dim=input_dim)
    backbone_config = MLPBackboneConfig(
        input_dim=input_dim,
        hidden_dims=128,
        output_dim=LATENT_DIM,
        n_hiddens=2,
    )

    dae_config = DAEConfig(
        task="classification",
        embedding_config=embedding_config,
        backbone_config=backbone_config,
        output_dim=2,
        loss_fn="CrossEntropyLoss",
        metric="f1_score",
        cat_cardinality=cat_cardinality,
        num_continuous=len(numeric_cols),
        noise_type="Swap",
        noise_ratio=0.3,
        mask_loss_weight=1.0,
    )

    dae_model = DAELightning(dae_config)

    train_ds = DAEDataset(
        X=ssl_train_df,
        continuous_cols=numeric_cols,
        category_cols=categorical_cols,
    )
    valid_ds = DAEDataset(
        X=ssl_valid_df,
        continuous_cols=numeric_cols,
        category_cols=categorical_cols,
    )
    datamodule = TS3LDataModule(
        train_ds=train_ds,
        val_ds=valid_ds,
        batch_size=BATCH_SIZE,
        train_sampler="random",
        train_collate_fn=DAECollateFN(dae_config),
        valid_collate_fn=DAECollateFN(dae_config),
        n_jobs=0,
    )

    dae_model.set_first_phase()

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=PRETRAIN_EPOCHS,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(dae_model, datamodule=datamodule)

    ssl_dataset = DAEDataset(
        X=ssl_features,
        continuous_cols=numeric_cols,
        category_cols=categorical_cols,
    )
    last_dataset = DAEDataset(
        X=last_features,
        continuous_cols=numeric_cols,
        category_cols=categorical_cols,
    )

    semester_embeddings = extract_embeddings_from_dae(torch, DataLoader, ssl_dataset, dae_model)
    last_embeddings = extract_embeddings_from_dae(torch, DataLoader, last_dataset, dae_model)
    scaler = StandardScaler()
    last_embeddings_scaled = scaler.fit_transform(last_embeddings)

    embedding_cols = [f"emb_{i:02d}" for i in range(last_embeddings.shape[1])]
    semester_embedding_df = ssl_pool_df.copy().iloc[: semester_embeddings.shape[0]].copy()
    for idx, col in enumerate(embedding_cols):
        semester_embedding_df[col] = semester_embeddings[:, idx]
    semester_embedding_df.to_csv(
        OUTPUT_DIR / "tabulars3l_semester_embeddings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    outlier_mask, distance_to_center, outlier_threshold = detect_outliers_by_distance(
        last_embeddings_scaled,
        OUTLIER_DISTANCE_PERCENTILE,
    )
    inlier_mask = ~outlier_mask
    clustering_embeddings = last_embeddings_scaled[inlier_mask]

    best_k, k_search, cluster_labels = choose_best_k(
        clustering_embeddings,
        K_SEARCH_VALUES,
        MIN_CLUSTER_RATIO,
    )
    save_cluster_outputs(
        last_semester_df,
        last_embeddings,
        last_embeddings_scaled,
        inlier_mask,
        outlier_mask,
        distance_to_center,
        outlier_threshold,
        best_k,
        k_search,
        cluster_labels,
    )

    summary = {
        "repo_reference": "https://github.com/Alcoholrithm/TabularS3L",
        "local_repo_root": str(runtime["local_root"]) if runtime["local_root"] is not None else "",
        "method": "local TabularS3L DAE first-phase embedding + KMeans",
        "input_dim": input_dim,
        "latent_dim": LATENT_DIM,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "k_search_values": list(K_SEARCH_VALUES),
        "min_cluster_ratio": MIN_CLUSTER_RATIO,
        "min_cluster_size_threshold": int(np.ceil(len(last_semester_df) * MIN_CLUSTER_RATIO)),
        "outlier_distance_percentile": OUTLIER_DISTANCE_PERCENTILE,
        "outlier_threshold": float(outlier_threshold),
        "outlier_count": int(outlier_mask.sum()),
        "inlier_count": int(inlier_mask.sum()),
        "ssl_pool_rows": int(len(ssl_pool_df)),
        "cluster_rows": int(len(last_semester_df)),
        "ordered_feature_cols": ordered_feature_cols,
        "categorical_cols": categorical_cols,
        "continuous_cols": numeric_cols,
        "cat_cardinality": cat_cardinality,
        "best_k": int(best_k),
        "k_search": k_search,
    }
    (OUTPUT_DIR / "tabulars3l_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Local TabularS3L DAE pipeline finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
