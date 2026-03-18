from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = BASE_DIR / "outputs" / "results"


def load_cluster_summary() -> pd.DataFrame:
    return pd.read_csv(INPUT_DIR / "transformer_inlier_cluster_summary.csv")


def load_student_clusters() -> pd.DataFrame:
    return pd.read_csv(INPUT_DIR / "transformer_student_last_semester_clusters.csv")


def choose_cluster_roles(summary: pd.DataFrame) -> dict[int, str]:
    cluster_ids = summary["cluster"].astype(int).tolist()
    if len(cluster_ids) < 4:
        raise ValueError("Transformer inlier clusters are fewer than 4; cannot build four-mode patterns.")

    work = summary.copy()
    work["cluster"] = work["cluster"].astype(int)

    excellent_cluster = int(
        work.sort_values(
            ["avg_score_mean", "physical_test_score_mean", "fail_course_count_mean"],
            ascending=[False, False, True],
        ).iloc[0]["cluster"]
    )

    warning_cluster = int(
        work.sort_values(
            ["avg_score_mean", "fail_course_count_mean"],
            ascending=[True, False],
        ).iloc[0]["cluster"]
    )

    remaining = work[~work["cluster"].isin([excellent_cluster, warning_cluster])].copy()

    active_risk_cluster = int(
        remaining.sort_values(
            ["next_term_risk_rate", "internet_hours_sum_mean", "physical_test_score_mean"],
            ascending=[False, False, True],
        ).iloc[0]["cluster"]
    )

    remaining = remaining[remaining["cluster"] != active_risk_cluster].copy()

    if remaining.shape[0] == 1:
        steady_core_cluster = int(remaining.iloc[0]["cluster"])
        physical_weak_cluster = steady_core_cluster
        split_same = True
    else:
        steady_core_cluster = int(
            remaining.sort_values(
                ["avg_score_mean", "physical_test_score_mean", "fail_course_count_mean"],
                ascending=[False, False, True],
            ).iloc[0]["cluster"]
        )
        physical_weak_cluster = int(
            remaining.sort_values(
                ["physical_test_score_mean", "next_term_risk_rate"],
                ascending=[True, False],
            ).iloc[0]["cluster"]
        )
        split_same = False

    mapping = {
        excellent_cluster: "卓越稳健型",
        active_risk_cluster: "行为活跃风险型",
        warning_cluster: "学业预警型",
    }

    if split_same:
        mapping[steady_core_cluster] = "稳定中坚型"
    else:
        mapping[steady_core_cluster] = "稳定中坚型"
        mapping[physical_weak_cluster] = "体能薄弱波动型"

        for cluster_id in remaining["cluster"].astype(int).tolist():
            if cluster_id not in mapping:
                mapping[cluster_id] = "稳定中坚型"

    return mapping


def apply_mode_mapping(student_df: pd.DataFrame, mode_map: dict[int, str]) -> pd.DataFrame:
    out = student_df.copy()
    out["cluster"] = out["cluster"].astype(int)
    out["is_outlier"] = pd.to_numeric(out["is_outlier"], errors="coerce").fillna(0).astype(int)
    out["risk_label_next_term_numeric"] = pd.to_numeric(out["risk_label_next_term"], errors="coerce")

    out["mode_name"] = out["cluster"].map(mode_map)
    out.loc[out["cluster"] == -1, "mode_name"] = "极端异常预警型"
    out["mode_name"] = out["mode_name"].fillna("稳定中坚型")
    return out


def build_mode_summary(mode_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        mode_df.groupby("mode_name")
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

    order = [
        "卓越稳健型",
        "稳定中坚型",
        "行为活跃风险型",
        "体能薄弱波动型",
        "学业预警型",
        "极端异常预警型",
    ]
    summary["mode_order"] = summary["mode_name"].apply(lambda x: order.index(x) if x in order else 999)
    summary = summary.sort_values("mode_order").drop(columns=["mode_order"]).reset_index(drop=True)
    return summary


def build_markdown(mode_summary: pd.DataFrame) -> str:
    lines = [
        "# Transformer Four-Mode Personas",
        "",
        "This file consolidates the current Transformer clustering result into a business-facing four-mode portrait system.",
        "",
    ]

    for _, row in mode_summary.iterrows():
        lines.extend(
            [
                f"## {row['mode_name']}",
                "",
                f"- student_count: {int(row['student_count'])}",
                f"- avg_score_mean: {row['avg_score_mean']:.2f}" if pd.notna(row["avg_score_mean"]) else "- avg_score_mean: NA",
                f"- fail_course_count_mean: {row['fail_course_count_mean']:.4f}" if pd.notna(row["fail_course_count_mean"]) else "- fail_course_count_mean: NA",
                f"- internet_hours_sum_mean: {row['internet_hours_sum_mean']:.4f}" if pd.notna(row["internet_hours_sum_mean"]) else "- internet_hours_sum_mean: NA",
                f"- online_learning_bfb_snapshot_mean: {row['online_learning_bfb_snapshot_mean']:.2f}" if pd.notna(row["online_learning_bfb_snapshot_mean"]) else "- online_learning_bfb_snapshot_mean: NA",
                f"- physical_test_score_mean: {row['physical_test_score_mean']:.2f}" if pd.notna(row["physical_test_score_mean"]) else "- physical_test_score_mean: NA",
                f"- next_term_risk_rate: {row['next_term_risk_rate']:.4f}" if pd.notna(row["next_term_risk_rate"]) else "- next_term_risk_rate: NA",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    summary_df = load_cluster_summary()
    student_df = load_student_clusters()

    mode_map = choose_cluster_roles(summary_df)
    mode_df = apply_mode_mapping(student_df, mode_map)
    mode_summary = build_mode_summary(mode_df)

    mode_labels = pd.DataFrame(
        [
            {"cluster_id": int(cluster_id), "mode_name": mode_name}
            for cluster_id, mode_name in sorted(mode_map.items(), key=lambda x: x[0])
        ]
        + [{"cluster_id": -1, "mode_name": "极端异常预警型"}]
    )

    mode_df.to_csv(INPUT_DIR / "transformer_four_mode_assignment.csv", index=False, encoding="utf-8-sig")
    mode_summary.to_csv(INPUT_DIR / "transformer_four_mode_summary.csv", index=False, encoding="utf-8-sig")
    mode_labels.to_csv(INPUT_DIR / "transformer_four_mode_labels.csv", index=False, encoding="utf-8-sig")

    md_text = build_markdown(mode_summary)
    (INPUT_DIR / "transformer_four_mode_personas.md").write_text(md_text, encoding="utf-8")

    run_summary = {
        "source_cluster_summary": str(INPUT_DIR / "transformer_inlier_cluster_summary.csv"),
        "source_student_clusters": str(INPUT_DIR / "transformer_student_last_semester_clusters.csv"),
        "mode_map": {str(k): v for k, v in mode_map.items()},
        "mode_count": int(mode_summary.shape[0]),
    }
    (INPUT_DIR / "transformer_four_mode_run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Transformer four-mode grouping finished.")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
