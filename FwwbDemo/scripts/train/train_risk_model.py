from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent.parent
BASE_DATA_PATH = BASE_DIR / "prepared" / "03_datasets" / "student_semester_base.csv"
EMBEDDING_PATH = BASE_DIR / "prepared" / "06_tabulars3l" / "tabulars3l_semester_embeddings.csv"
CLUSTER_ASSIGN_PATH = BASE_DIR / "prepared" / "06_tabulars3l" / "tabulars3l_semester_cluster_assignment.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "results"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "risk_label_next_term"
TRAIN_YEARS = {"2020-2021", "2021-2022", "2022-2023"}
TEST_YEARS = {"2023-2024"}
KEY_COLS = ["student_id", "school_year", "semester", "term_order"]

BASE_NUMERIC_COLS = [
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

BASE_CATEGORICAL_COLS = [
    "gender",
    "grade",
    "college",
    "major",
    "semester",
]


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


def load_base_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL].astype(str).str.strip() != "")]
    df = df[df["school_year"].isin(TRAIN_YEARS | TEST_YEARS)].copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("Int64")
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    bmi_pairs = df["physical_test_bmi"].apply(parse_bmi_text)
    df["bmi_height_cm"] = bmi_pairs.apply(lambda x: x[0])
    df["bmi_weight_kg"] = bmi_pairs.apply(lambda x: x[1])

    for col in BASE_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BASE_CATEGORICAL_COLS:
        df[col] = df[col].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN")

    return df


def load_embedding_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_") and not c.startswith("emb_scaled_")]
    use_cols = KEY_COLS + emb_cols
    return df[use_cols].copy()


def load_cluster_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    use_cols = KEY_COLS + ["cluster", "is_outlier"]
    if "cluster_name" in df.columns:
        use_cols.append("cluster_name")
    if "embedding_distance_to_center" in df.columns:
        use_cols.append("embedding_distance_to_center")
    out = df[use_cols].copy()
    if "cluster_name" not in out.columns:
        out["cluster_name"] = out["cluster"].apply(lambda x: f"Cluster_{x}" if x != -1 else "Outlier")
    out["cluster_name"] = out["cluster_name"].fillna("UNKNOWN").astype(str)
    out["is_outlier"] = pd.to_numeric(out["is_outlier"], errors="coerce").fillna(0)
    out["embedding_distance_to_center"] = pd.to_numeric(out["embedding_distance_to_center"], errors="coerce")
    return out


def merge_all(base_df: pd.DataFrame, emb_df: pd.DataFrame, cluster_df: pd.DataFrame) -> pd.DataFrame:
    merged = base_df.merge(emb_df, on=KEY_COLS, how="left")
    merged = merged.merge(cluster_df, on=KEY_COLS, how="left")
    if "cluster_name" in merged.columns:
        merged["cluster_name"] = merged["cluster_name"].fillna("NO_CLUSTER")
    else:
        merged["cluster_name"] = merged["cluster"].apply(lambda x: f"Cluster_{x}" if pd.notna(x) and x != -1 else "Outlier")
    merged["is_outlier"] = pd.to_numeric(merged["is_outlier"], errors="coerce").fillna(0)
    merged["embedding_distance_to_center"] = pd.to_numeric(
        merged["embedding_distance_to_center"], errors="coerce"
    )
    return merged


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["school_year"].isin(TRAIN_YEARS)].copy()
    test_df = df[df["school_year"].isin(TEST_YEARS)].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty.")
    return train_df, test_df


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def evaluate(model_name: str, feature_set: str, pipeline: Pipeline, test_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    x_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int)
    y_pred = pipeline.predict(x_test)
    y_score = pipeline.predict_proba(x_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "feature_set": feature_set,
        "model": model_name,
        "test_rows": int(len(test_df)),
        "positive_rows": int(y_test.sum()),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_test, y_score)), 6),
        "pr_auc": round(float(average_precision_score(y_test, y_score)), 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_predictions(feature_set: str, model_name: str, pipeline: Pipeline, test_df: pd.DataFrame, feature_cols: list[str]) -> None:
    x_test = test_df[feature_cols]
    pred_df = test_df[KEY_COLS + [TARGET_COL]].copy()
    pred_df["pred_label"] = pipeline.predict(x_test)
    pred_df["pred_proba"] = pipeline.predict_proba(x_test)[:, 1]
    pred_df.to_csv(
        OUTPUT_DIR / f"predictions_{feature_set}_{model_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_df = load_base_df(BASE_DATA_PATH)
    emb_df = load_embedding_df(EMBEDDING_PATH)
    cluster_df = load_cluster_df(CLUSTER_ASSIGN_PATH)
    df = merge_all(base_df, emb_df, cluster_df)
    train_df, test_df = split_by_time(df)

    embedding_cols = [c for c in df.columns if c.startswith("emb_") and not c.startswith("emb_scaled_")]
    cluster_numeric_cols = ["is_outlier", "embedding_distance_to_center"]
    cluster_categorical_cols = ["cluster_name"]

    feature_sets = {
        "baseline_only": {
            "numeric": BASE_NUMERIC_COLS,
            "categorical": BASE_CATEGORICAL_COLS,
        },
        "embedding_only": {
            "numeric": embedding_cols + cluster_numeric_cols,
            "categorical": cluster_categorical_cols,
        },
        "baseline_plus_embedding": {
            "numeric": BASE_NUMERIC_COLS + embedding_cols + cluster_numeric_cols,
            "categorical": BASE_CATEGORICAL_COLS + cluster_categorical_cols,
        },
    }

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    metrics_rows = []

    for feature_set_name, cols in feature_sets.items():
        feature_cols = cols["numeric"] + cols["categorical"]
        x_train = train_df[feature_cols]
        y_train = train_df[TARGET_COL].astype(int)
        preprocessor = build_preprocessor(cols["numeric"], cols["categorical"])

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", model),
                ]
            )
            pipeline.fit(x_train, y_train)
            metrics_rows.append(evaluate(model_name, feature_set_name, pipeline, test_df, feature_cols))
            save_predictions(feature_set_name, model_name, pipeline, test_df, feature_cols)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["feature_set", "pr_auc"], ascending=[True, False])
    metrics_df.to_csv(OUTPUT_DIR / "embedding_risk_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "base_data_path": str(BASE_DATA_PATH),
        "embedding_path": str(EMBEDDING_PATH),
        "cluster_assignment_path": str(CLUSTER_ASSIGN_PATH),
        "train_years": sorted(TRAIN_YEARS),
        "test_years": sorted(TEST_YEARS),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_positive_rows": int(train_df[TARGET_COL].sum()),
        "test_positive_rows": int(test_df[TARGET_COL].sum()),
        "embedding_cols": embedding_cols,
        "cluster_numeric_cols": cluster_numeric_cols,
        "cluster_categorical_cols": cluster_categorical_cols,
        "feature_sets": feature_sets,
    }
    (OUTPUT_DIR / "embedding_risk_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Embedding-enhanced risk modeling finished.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
