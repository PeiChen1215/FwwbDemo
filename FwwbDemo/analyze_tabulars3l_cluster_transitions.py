from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "prepared" / "06_tabulars3l"


def load_run_summary() -> dict:
    return json.loads((DATA_DIR / "tabulars3l_run_summary.json").read_text(encoding="utf-8"))


def load_cluster_labels() -> dict[int, str]:
    path = DATA_DIR / "tabulars3l_cluster_labels.csv"
    if not path.exists():
        return {-1: "极端异常预警型", 0: "稳定发展型", 1: "学业脆弱型"}

    df = pd.read_csv(path)
    return {int(row["cluster_id"]): str(row["cluster_name_cn"]) for _, row in df.iterrows()}


def get_embedding_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("emb_") and not col.startswith("emb_scaled_")]


def get_scaled_embedding_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("emb_scaled_")]


def assign_semester_clusters(
    semester_df: pd.DataFrame,
    last_df: pd.DataFrame,
    outlier_threshold: float,
    cluster_name_map: dict[int, str],
) -> pd.DataFrame:
    raw_emb_cols = get_embedding_columns(last_df)
    scaled_emb_cols = get_scaled_embedding_columns(last_df)

    scaler = StandardScaler()
    scaler.fit(last_df[raw_emb_cols].astype(float))

    semester_scaled = scaler.transform(semester_df[raw_emb_cols].astype(float))
    last_scaled = scaler.transform(last_df[raw_emb_cols].astype(float))

    center = last_scaled.mean(axis=0, keepdims=True)
    distance = np.linalg.norm(semester_scaled - center, axis=1)
    semester_outlier = distance > outlier_threshold

    inlier_last = last_df[last_df["cluster"].astype(int) >= 0].copy()
    cluster_centers = (
        inlier_last.groupby(inlier_last["cluster"].astype(int))[scaled_emb_cols]
        .mean()
        .sort_index()
    )

    center_matrix = cluster_centers.to_numpy(dtype=float)
    center_cluster_ids = cluster_centers.index.to_list()

    dists = np.linalg.norm(semester_scaled[:, None, :] - center_matrix[None, :, :], axis=2)
    nearest_idx = dists.argmin(axis=1)
    nearest_cluster_ids = np.array([center_cluster_ids[idx] for idx in nearest_idx], dtype=int)

    final_cluster_ids = nearest_cluster_ids.copy()
    final_cluster_ids[semester_outlier] = -1

    out = semester_df.copy()
    for idx, col in enumerate(scaled_emb_cols):
        out[col] = semester_scaled[:, idx]

    out["cluster"] = final_cluster_ids
    out["cluster_name"] = out["cluster"].map(cluster_name_map)
    out["is_outlier"] = semester_outlier.astype(int)
    out["embedding_distance_to_center"] = distance
    out["risk_label_next_term_numeric"] = pd.to_numeric(out["risk_label_next_term"], errors="coerce")

    return out


def build_transition_rows(assigned_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    ordered = assigned_df.copy()
    ordered["term_order"] = pd.to_numeric(ordered["term_order"], errors="coerce")
    ordered = ordered.sort_values(["student_id", "term_order", "school_year", "semester"])

    for student_id, group in ordered.groupby("student_id"):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue

        for idx in range(len(group) - 1):
            current = group.iloc[idx]
            nxt = group.iloc[idx + 1]
            rows.append(
                {
                    "student_id": student_id,
                    "from_school_year": current["school_year"],
                    "from_semester": current["semester"],
                    "to_school_year": nxt["school_year"],
                    "to_semester": nxt["semester"],
                    "from_cluster": int(current["cluster"]),
                    "from_cluster_name": current["cluster_name"],
                    "to_cluster": int(nxt["cluster"]),
                    "to_cluster_name": nxt["cluster_name"],
                    "from_is_outlier": int(current["is_outlier"]),
                    "to_is_outlier": int(nxt["is_outlier"]),
                    "from_avg_score": pd.to_numeric(current["avg_score"], errors="coerce"),
                    "from_fail_course_count": pd.to_numeric(current["fail_course_count"], errors="coerce"),
                    "from_internet_hours_sum": pd.to_numeric(current["internet_hours_sum"], errors="coerce"),
                    "risk_label_next_term_numeric": pd.to_numeric(current["risk_label_next_term_numeric"], errors="coerce"),
                    "next_term_risk_event_current": pd.to_numeric(nxt["risk_event_current"], errors="coerce"),
                }
            )

    return pd.DataFrame(rows)


def save_transition_outputs(assigned_df: pd.DataFrame, transition_df: pd.DataFrame) -> dict:
    assigned_df.to_csv(DATA_DIR / "tabulars3l_semester_cluster_assignment.csv", index=False, encoding="utf-8-sig")
    transition_df.to_csv(DATA_DIR / "tabulars3l_cluster_transition_detail.csv", index=False, encoding="utf-8-sig")

    transition_counts = (
        transition_df.groupby(["from_cluster", "from_cluster_name", "to_cluster", "to_cluster_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["from_cluster", "to_cluster"])
    )
    transition_counts.to_csv(DATA_DIR / "tabulars3l_cluster_transition_counts.csv", index=False, encoding="utf-8-sig")

    transition_rates = transition_counts.copy()
    transition_rates["row_total"] = transition_rates.groupby("from_cluster")["count"].transform("sum")
    transition_rates["rate"] = transition_rates["count"] / transition_rates["row_total"]
    transition_rates.to_csv(DATA_DIR / "tabulars3l_cluster_transition_rates.csv", index=False, encoding="utf-8-sig")

    transition_risk = (
        transition_df.groupby(["from_cluster", "from_cluster_name", "to_cluster", "to_cluster_name"])
        .agg(
            transition_count=("student_id", "count"),
            next_term_risk_rate=("risk_label_next_term_numeric", "mean"),
            next_term_risk_event_rate=("next_term_risk_event_current", "mean"),
            from_avg_score_mean=("from_avg_score", "mean"),
            from_fail_course_count_mean=("from_fail_course_count", "mean"),
            from_internet_hours_sum_mean=("from_internet_hours_sum", "mean"),
        )
        .reset_index()
        .sort_values(["from_cluster", "to_cluster"])
    )
    transition_risk.to_csv(DATA_DIR / "tabulars3l_cluster_transition_risk.csv", index=False, encoding="utf-8-sig")

    semester_cluster_counts = (
        assigned_df.groupby(["school_year", "semester", "cluster", "cluster_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["school_year", "semester", "cluster"])
    )
    semester_cluster_counts.to_csv(DATA_DIR / "tabulars3l_semester_cluster_counts.csv", index=False, encoding="utf-8-sig")

    return {
        "transition_counts": transition_counts,
        "transition_rates": transition_rates,
        "transition_risk": transition_risk,
        "semester_cluster_counts": semester_cluster_counts,
    }


def build_markdown_summary(
    assigned_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    outputs: dict,
) -> str:
    transition_rates = outputs["transition_rates"]
    transition_risk = outputs["transition_risk"]
    semester_cluster_counts = outputs["semester_cluster_counts"]

    top_transitions = (
        transition_rates.sort_values("count", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )
    top_risk_transitions = (
        transition_risk[transition_risk["transition_count"] >= 20]
        .sort_values("next_term_risk_rate", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    same_cluster_rate = (
        transition_df[transition_df["from_cluster"] == transition_df["to_cluster"]].shape[0] / len(transition_df)
        if len(transition_df) > 0
        else 0.0
    )

    lines = [
        "# TabularS3L Cluster Transition Analysis",
        "",
        f"- semester assignment rows: {len(assigned_df)}",
        f"- transition pairs: {len(transition_df)}",
        f"- same-cluster transition rate: {same_cluster_rate:.4f}",
        f"- outlier semester rows: {int(assigned_df['is_outlier'].sum())}",
        "",
        "## Reading",
        "",
        "- We assign every student-semester row to the nearest main cluster in the TabularS3L embedding space.",
        "- Extreme outlier semesters are kept as `cluster = -1` and are not forced into normal student portraits.",
        "- This lets us observe not only final portraits, but also how students move between portraits over time.",
        "",
        "## Top Transition Paths",
        "",
    ]

    for row in top_transitions:
        lines.append(
            f"- {row['from_cluster_name']} -> {row['to_cluster_name']}: count={int(row['count'])}, rate={row['rate']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Highest-Risk Transition Patterns",
            "",
        ]
    )

    for row in top_risk_transitions:
        lines.append(
            "- "
            f"{row['from_cluster_name']} -> {row['to_cluster_name']}: "
            f"count={int(row['transition_count'])}, "
            f"next_term_risk_rate={row['next_term_risk_rate']:.4f}, "
            f"from_avg_score_mean={row['from_avg_score_mean']:.2f}, "
            f"from_fail_course_count_mean={row['from_fail_course_count_mean']:.2f}"
        )

    lines.extend(
        [
            "",
            "## Semester Distribution Snapshot",
            "",
        ]
    )

    for _, row in semester_cluster_counts.head(30).iterrows():
        lines.append(
            f"- {row['school_year']} semester {row['semester']} | {row['cluster_name']}: {int(row['count'])}"
        )

    lines.extend(
        [
            "",
            "## Competition Message",
            "",
            "The TabularS3L-based student portraits are not static labels only.",
            "They can be tracked across semesters, which supports a stronger story of early warning and intervention timing.",
            "A key value of this analysis is that it turns portrait learning into dynamic student-state monitoring.",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    run_summary = load_run_summary()
    cluster_name_map = load_cluster_labels()

    semester_df = pd.read_csv(DATA_DIR / "tabulars3l_semester_embeddings.csv")
    last_df = pd.read_csv(DATA_DIR / "tabulars3l_student_last_semester_clusters.csv")

    assigned_df = assign_semester_clusters(
        semester_df=semester_df,
        last_df=last_df,
        outlier_threshold=float(run_summary["outlier_threshold"]),
        cluster_name_map=cluster_name_map,
    )

    transition_df = build_transition_rows(assigned_df)
    outputs = save_transition_outputs(assigned_df, transition_df)

    md_text = build_markdown_summary(assigned_df, transition_df, outputs)
    (DATA_DIR / "tabulars3l_cluster_transition_analysis.md").write_text(md_text, encoding="utf-8")

    summary = {
        "semester_assignment_rows": int(len(assigned_df)),
        "transition_rows": int(len(transition_df)),
        "outlier_semester_rows": int(assigned_df["is_outlier"].sum()),
        "cluster_name_map": cluster_name_map,
    }
    (DATA_DIR / "tabulars3l_transition_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("TabularS3L transition analysis finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
