from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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


DATA_PATH = Path("prepared/03_datasets/student_semester_base.csv")
OUTPUT_DIR = Path("prepared/04_models")

TARGET_COL = "risk_label_next_term"
TRAIN_YEARS = {"2020-2021", "2021-2022", "2022-2023"}
TEST_YEARS = {"2023-2024"}

ID_COLS = ["student_id", "school_year", "semester", "term_order"]
DROP_COLS = [
    "risk_event_current",
    "risk_event_type_codes",
    "risk_event_reason_codes",
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

    df = df[df[TARGET_COL].notna() & (df[TARGET_COL].astype(str).str.strip() != "")]
    df = df[df["school_year"].isin(TRAIN_YEARS | TEST_YEARS)].copy()

    df[TARGET_COL] = df[TARGET_COL].astype(int)

    bmi_pairs = df["physical_test_bmi"].apply(parse_bmi_text)
    df["bmi_height_cm"] = bmi_pairs.apply(lambda x: x[0])
    df["bmi_weight_kg"] = bmi_pairs.apply(lambda x: x[1])

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN")

    return df


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["school_year"].isin(TRAIN_YEARS)].copy()
    test_df = df[df["school_year"].isin(TEST_YEARS)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Check school_year values.")

    return train_df, test_df


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
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ]
    )


def evaluate_model(model_name: str, pipeline: Pipeline, test_df: pd.DataFrame) -> dict[str, float | int | str]:
    x_test = test_df[NUMERIC_COLS + CATEGORICAL_COLS]
    y_test = test_df[TARGET_COL].astype(int)

    y_pred = pipeline.predict(x_test)
    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(x_test)[:, 1]
    else:
        y_score = y_pred.astype(float)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
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


def save_predictions(model_name: str, pipeline: Pipeline, test_df: pd.DataFrame) -> None:
    x_test = test_df[NUMERIC_COLS + CATEGORICAL_COLS]
    y_pred = pipeline.predict(x_test)
    y_score = pipeline.predict_proba(x_test)[:, 1]

    pred_df = test_df[ID_COLS + [TARGET_COL]].copy()
    pred_df["pred_label"] = y_pred
    pred_df["pred_proba"] = y_score
    pred_df.to_csv(OUTPUT_DIR / f"predictions_{model_name}.csv", index=False, encoding="utf-8-sig")


def save_feature_importance(model_name: str, pipeline: Pipeline) -> None:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    importance_df.head(100).to_csv(
        OUTPUT_DIR / f"feature_importance_{model_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)
    train_df, test_df = split_by_time(df)

    x_train = train_df[NUMERIC_COLS + CATEGORICAL_COLS]
    y_train = train_df[TARGET_COL].astype(int)

    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            class_weight="balanced",
            random_state=42,
        )
    except Exception:
        pass

    metrics_rows: list[dict[str, float | int | str]] = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", model),
            ]
        )

        pipeline.fit(x_train, y_train)
        metrics_rows.append(evaluate_model(model_name, pipeline, test_df))
        save_predictions(model_name, pipeline, test_df)
        save_feature_importance(model_name, pipeline)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("pr_auc", ascending=False)
    metrics_df.to_csv(OUTPUT_DIR / "baseline_metrics.csv", index=False, encoding="utf-8-sig")

    run_summary = {
        "data_path": str(DATA_PATH),
        "train_years": sorted(TRAIN_YEARS),
        "test_years": sorted(TEST_YEARS),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_positive_rows": int(train_df[TARGET_COL].sum()),
        "test_positive_rows": int(test_df[TARGET_COL].sum()),
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "drop_cols": DROP_COLS,
        "metrics_preview": metrics_rows,
    }

    (OUTPUT_DIR / "baseline_run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Training finished.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
