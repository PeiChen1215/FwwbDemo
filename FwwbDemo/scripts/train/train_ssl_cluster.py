from __future__ import annotations

from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("prepared/03_datasets/student_semester_base.csv")
OUTPUT_DIR = Path("prepared/05_ssl_clustering")

ID_COLS = ["student_id", "school_year", "semester", "term_order"]
UNUSED_COLS = [
    "risk_event_current",
    "risk_event_type_codes",
    "risk_event_reason_codes",
    "risk_label_next_term",
    "physical_test_bmi",
]

NUMERIC_COLS = [
    "selected_course_count",
    "score_course_count",
    "avg_score",
    "score_std",
    "fail_course_count",
    "fail_ratio",
    "avg_gpa",
    "credit_sum",
    "resit_exam_count",
    "internet_month_count",
    "internet_hours_sum",
    "internet_hours_avg_per_month",
    "internet_diff_mean",
    "online_learning_bfb_snapshot",
    "physical_test_score",
    "bmi_height_cm",
    "bmi_weight_kg",
]

CATEGORICAL_COLS = [
    "gender",
    "grade",
    "college",
    "major",
    "semester",
]

RANDOM_STATE = 42
LATENT_DIM = 16


def parse_bmi_text(value: object) -> tuple[float | np.nan, float | np.nan]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, np.nan

    text = str(value).strip()
    if not text:
        return np.nan, np.nan

    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$", text)
    if not match:
        return np.nan, np.nan

    return float(match.group(1)), float(match.group(2))


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    bmi_pairs = df["physical_test_bmi"].apply(parse_bmi_text)
    df["bmi_height_cm"] = bmi_pairs.apply(lambda x: x[0])
    df["bmi_weight_kg"] = bmi_pairs.apply(lambda x: x[1])

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN")

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ]
    )


def corrupt_features(x: np.ndarray, corruption_rate: float = 0.25, noise_std: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)
    x_noisy = x.copy()

    mask = rng.random(x_noisy.shape) < corruption_rate
    x_noisy[mask] = 0.0

    noise = rng.normal(loc=0.0, scale=noise_std, size=x_noisy.shape)
    x_noisy = x_noisy + noise
    return x_noisy


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def extract_latent_embedding(model: MLPRegressor, x: np.ndarray, latent_layer_index: int = 1) -> np.ndarray:
    activations = x

    for layer_idx, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
        activations = activations @ coef + intercept

        is_output_layer = layer_idx == len(model.coefs_) - 1
        if not is_output_layer:
            activations = relu(activations)

        if layer_idx == latent_layer_index:
            return activations

    raise ValueError("Latent layer index is out of range.")


def choose_best_k(embeddings: np.ndarray, k_values: range) -> tuple[int, list[dict[str, float | int]]]:
    evaluations: list[dict[str, float | int]] = []
    best_k = None
    best_score = -1.0

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        evaluations.append({"k": k, "silhouette_score": round(float(score), 6)})
        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        raise ValueError("Unable to choose cluster number.")

    return best_k, evaluations


def save_cluster_profile(student_cluster_df: pd.DataFrame) -> None:
    summary = (
        student_cluster_df.groupby("cluster")
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
    )
    summary.to_csv(OUTPUT_DIR / "student_cluster_summary.csv", index=False, encoding="utf-8-sig")


def save_cluster_distribution(student_cluster_df: pd.DataFrame) -> None:
    college_dist = (
        student_cluster_df.groupby(["cluster", "college"])
        .size()
        .reset_index(name="count")
        .sort_values(["cluster", "count"], ascending=[True, False])
    )
    college_dist.to_csv(OUTPUT_DIR / "student_cluster_college_distribution.csv", index=False, encoding="utf-8-sig")

    major_dist = (
        student_cluster_df.groupby(["cluster", "major"])
        .size()
        .reset_index(name="count")
        .sort_values(["cluster", "count"], ascending=[True, False])
    )
    major_dist.to_csv(OUTPUT_DIR / "student_cluster_major_distribution.csv", index=False, encoding="utf-8-sig")


def save_pca_plot(embeddings: np.ndarray, labels: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, s=18, alpha=0.7, cmap="tab10")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("Student Last-Semester Embedding Clusters")
    plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "student_cluster_pca.png", dpi=180)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)
    feature_df = df.drop(columns=[col for col in UNUSED_COLS if col in df.columns])

    preprocessor = build_preprocessor()
    x_clean = preprocessor.fit_transform(feature_df[NUMERIC_COLS + CATEGORICAL_COLS])
    x_clean = np.asarray(x_clean, dtype=np.float32)
    x_noisy = corrupt_features(x_clean)

    autoencoder = MLPRegressor(
        hidden_layer_sizes=(128, LATENT_DIM, 128),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=RANDOM_STATE,
        verbose=True,
    )

    autoencoder.fit(x_noisy, x_clean)
    semester_embeddings = extract_latent_embedding(autoencoder, x_clean, latent_layer_index=1)

    embedding_cols = [f"emb_{i:02d}" for i in range(semester_embeddings.shape[1])]
    output_feature_cols = [
        col
        for col in list(dict.fromkeys(NUMERIC_COLS + CATEGORICAL_COLS))
        if col not in ID_COLS and col not in {"risk_event_current", "risk_label_next_term"}
    ]
    selected_cols = ID_COLS + ["risk_event_current", "risk_label_next_term"] + output_feature_cols
    semester_embedding_df = df[selected_cols].copy()
    semester_embedding_df = semester_embedding_df.loc[:, ~semester_embedding_df.columns.duplicated()].copy()
    semester_embedding_df["term_order"] = pd.to_numeric(
        semester_embedding_df["term_order"], errors="coerce"
    )
    for idx, col in enumerate(embedding_cols):
        semester_embedding_df[col] = semester_embeddings[:, idx]

    semester_embedding_df.to_csv(
        OUTPUT_DIR / "semester_embeddings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    semester_embedding_df["risk_label_next_term_numeric"] = pd.to_numeric(
        semester_embedding_df["risk_label_next_term"], errors="coerce"
    )

    last_semester_df = (
        semester_embedding_df.sort_values(["student_id", "term_order", "school_year", "semester"])
        .groupby("student_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    x_last = last_semester_df[embedding_cols].to_numpy(dtype=np.float32)
    best_k, cluster_eval = choose_best_k(x_last, range(3, 9))

    cluster_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    cluster_labels = cluster_model.fit_predict(x_last)

    student_cluster_df = last_semester_df.copy()
    student_cluster_df["cluster"] = cluster_labels
    student_cluster_df.to_csv(
        OUTPUT_DIR / "student_last_semester_clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_cluster_profile(student_cluster_df)
    save_cluster_distribution(student_cluster_df)
    save_pca_plot(x_last, cluster_labels)

    pd.DataFrame(cluster_eval).to_csv(
        OUTPUT_DIR / "cluster_k_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "data_path": str(DATA_PATH),
        "semester_rows": int(len(df)),
        "student_rows_for_clustering": int(len(student_cluster_df)),
        "latent_dim": LATENT_DIM,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "best_k": int(best_k),
        "k_search": cluster_eval,
    }

    (OUTPUT_DIR / "ssl_cluster_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("SSL embedding + clustering finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
