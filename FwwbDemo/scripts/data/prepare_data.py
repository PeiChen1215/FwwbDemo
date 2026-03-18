from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "prepared" / "03_datasets" / "student_semester_base.csv"
OUTPUT_DIR = BASE_DIR / "prepared" / "06_tabulars3l"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "risk_label_next_term"
SSL_YEARS = {"2020-2021", "2021-2022", "2022-2023", "2023-2024"}
SUPERVISED_TRAIN_YEARS = {"2020-2021", "2021-2022"}
SUPERVISED_VALID_YEARS = {"2022-2023"}
SUPERVISED_TEST_YEARS = {"2023-2024"}

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

    bmi_pairs = df["physical_test_bmi"].apply(parse_bmi_text)
    df["bmi_height_cm"] = bmi_pairs.apply(lambda x: x[0])
    df["bmi_weight_kg"] = bmi_pairs.apply(lambda x: x[1])

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN")

    return df


def save_split(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)

    feature_cols = list(dict.fromkeys(NUMERIC_COLS + CATEGORICAL_COLS))
    output_feature_cols = [
        col for col in feature_cols if col not in ID_COLS and col != TARGET_COL
    ]
    selected_cols = ID_COLS + ["risk_event_current"] + output_feature_cols + [TARGET_COL]
    feature_df = df[selected_cols].copy()
    feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()].copy()

    ssl_pool_df = feature_df[feature_df["school_year"].isin(SSL_YEARS)].copy()
    labeled_df = feature_df[
        feature_df[TARGET_COL].notna() & (feature_df[TARGET_COL].astype(str).str.strip() != "")
    ].copy()
    labeled_df[TARGET_COL] = labeled_df[TARGET_COL].astype(int)

    supervised_train_df = labeled_df[labeled_df["school_year"].isin(SUPERVISED_TRAIN_YEARS)].copy()
    supervised_valid_df = labeled_df[labeled_df["school_year"].isin(SUPERVISED_VALID_YEARS)].copy()
    supervised_test_df = labeled_df[labeled_df["school_year"].isin(SUPERVISED_TEST_YEARS)].copy()

    last_semester_df = (
        feature_df.sort_values(["student_id", "term_order", "school_year", "semester"])
        .groupby("student_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    save_split(ssl_pool_df, "ssl_pool")
    save_split(supervised_train_df, "supervised_train")
    save_split(supervised_valid_df, "supervised_valid")
    save_split(supervised_test_df, "supervised_test")
    save_split(last_semester_df, "last_semester_for_clustering")

    metadata = {
        "source_data": str(DATA_PATH),
        "feature_cols": feature_cols,
        "ordered_feature_cols": CATEGORICAL_COLS + NUMERIC_COLS,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "categorical_col_indices": list(range(len(CATEGORICAL_COLS))),
        "continuous_col_indices": list(range(len(CATEGORICAL_COLS), len(CATEGORICAL_COLS) + len(NUMERIC_COLS))),
        "id_cols": ID_COLS,
        "target_col": TARGET_COL,
        "ssl_years": sorted(SSL_YEARS),
        "supervised_train_years": sorted(SUPERVISED_TRAIN_YEARS),
        "supervised_valid_years": sorted(SUPERVISED_VALID_YEARS),
        "supervised_test_years": sorted(SUPERVISED_TEST_YEARS),
        "ssl_pool_rows": int(len(ssl_pool_df)),
        "supervised_train_rows": int(len(supervised_train_df)),
        "supervised_valid_rows": int(len(supervised_valid_df)),
        "supervised_test_rows": int(len(supervised_test_df)),
        "last_semester_rows": int(len(last_semester_df)),
        "supervised_positive_rows": {
            "train": int(supervised_train_df[TARGET_COL].sum()),
            "valid": int(supervised_valid_df[TARGET_COL].sum()),
            "test": int(supervised_test_df[TARGET_COL].sum()),
        },
    }

    (OUTPUT_DIR / "a14_tabulars3l_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("TabularS3L input files prepared.")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
