from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from model import (
    require_tabulars3l_runtime,
    build_scarf_transformer_model,
    extract_embeddings,
    run_hdbscan_clustering,
)


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "prepared" / "06_tabulars3l"
OUTPUT_DIR = BASE_DIR / "prepared" / "08_tabulars3l_transformer_scarf"

RANDOM_STATE = 42
PRETRAIN_EPOCHS = 200
BATCH_SIZE = 256
MIN_CLUSTER_RATIO = 0.03
LATENT_DIM = 64


def load_metadata() -> dict:
    return json.loads((INPUT_DIR / "a14_tabulars3l_metadata.json").read_text(encoding="utf-8"))


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(INPUT_DIR / f"{name}.csv")


def preprocess_for_ts3l(df, feature_cols, numeric_cols, categorical_cols):
    """Keep preprocessing aligned with the current TabularS3L pipeline."""
    out = df[feature_cols].copy()
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(out[col].median())
    for col in ["internet_hours_sum", "internet_hours_avg_per_month"]:
        if col in out.columns:
            out[col] = np.log1p(np.clip(out[col], a_min=0, a_max=None))
    if "internet_diff_mean" in out.columns:
        out["internet_diff_mean"] = np.sign(out["internet_diff_mean"]) * np.log1p(np.abs(out["internet_diff_mean"]))
    for col in categorical_cols:
        out[col] = out[col].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN")
        out[col] = pd.Categorical(out[col]).codes.astype(np.int64)
    return out[categorical_cols + numeric_cols]


def save_and_visualize(last_semester_df, scaled_embeddings, cluster_labels, outlier_mask):
    """Save clustering results and a simple PCA view."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    student_cluster_df = last_semester_df.copy()
    student_cluster_df["cluster"] = cluster_labels
    student_cluster_df["is_outlier"] = outlier_mask.astype(int)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(scaled_embeddings)

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, s=18, alpha=0.7, cmap="tab10")
    plt.title("TabularS3L SCARF-Transformer Student Clusters (HDBSCAN)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scarf_student_cluster_pca.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced[~outlier_mask, 0], reduced[~outlier_mask, 1], c="#4c78a8", s=18, alpha=0.45, label="Inlier")
    if outlier_mask.any():
        plt.scatter(reduced[outlier_mask, 0], reduced[outlier_mask, 1], c="#e45756", s=42, marker="x", label="Outlier")
    plt.title("PCA Space with HDBSCAN Outlier Highlight (-1 Label)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scarf_outlier_pca.png", dpi=180)
    plt.close()

    student_cluster_df.to_csv(OUTPUT_DIR / "scarf_student_last_semester_clusters.csv", index=False, encoding="utf-8-sig")
    return student_cluster_df


def main() -> None:
    runtime = require_tabulars3l_runtime()
    torch, pl, DataLoader = runtime["torch"], runtime["pl"], runtime["DataLoader"]
    SCARFDataset, TS3LDataModule = runtime["SCARFDataset"], runtime["TS3LDataModule"]

    metadata = load_metadata()
    feature_cols = metadata["feature_cols"]
    numeric_cols = metadata["numeric_cols"]
    categorical_cols = metadata["categorical_cols"]

    ssl_pool_df = load_split("ssl_pool")
    last_semester_df = load_split("last_semester_for_clustering")

    ssl_features = preprocess_for_ts3l(ssl_pool_df, feature_cols, numeric_cols, categorical_cols)
    last_features = preprocess_for_ts3l(last_semester_df, feature_cols, numeric_cols, categorical_cols)

    split_point = int(len(ssl_features) * 0.9)
    ssl_train_df = ssl_features.iloc[:split_point].reset_index(drop=True)
    ssl_valid_df = ssl_features.iloc[split_point:].reset_index(drop=True)

    scarf_model, scarf_config = build_scarf_transformer_model(
        runtime,
        ssl_train_df,
        numeric_cols,
        categorical_cols,
        LATENT_DIM,
    )

    train_ds = SCARFDataset(
        X=ssl_train_df,
        config=scarf_config,
        continuous_cols=numeric_cols,
        category_cols=categorical_cols,
    )
    valid_ds = SCARFDataset(
        X=ssl_valid_df,
        config=scarf_config,
        continuous_cols=numeric_cols,
        category_cols=categorical_cols,
    )
    datamodule = TS3LDataModule(
        train_ds=train_ds,
        val_ds=valid_ds,
        batch_size=BATCH_SIZE,
        train_sampler="random",
        n_jobs=0,
    )

    scarf_model.set_first_phase()
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=PRETRAIN_EPOCHS,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(scarf_model, datamodule=datamodule)

    last_dataset = SCARFDataset(
        X=last_features,
        config=scarf_config,
        continuous_cols=numeric_cols,
        category_cols=categorical_cols,
        is_second_phase=True,
    )
    last_embeddings = extract_embeddings(torch, DataLoader, last_dataset, scarf_model, BATCH_SIZE)
    scaled_embeddings = StandardScaler().fit_transform(last_embeddings)

    cluster_labels, inlier_mask, outlier_mask = run_hdbscan_clustering(scaled_embeddings, MIN_CLUSTER_RATIO)
    print(f"HDBSCAN detected {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} student portrait groups")
    print(f"Captured outlier students: {outlier_mask.sum()}")

    save_and_visualize(last_semester_df, scaled_embeddings, cluster_labels, outlier_mask)
    print("SCARF-Transformer clustering pipeline finished.")


if __name__ == "__main__":
    main()
