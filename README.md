# Prepared Workspace

## Purpose

This folder is the working area for the A14 modeling pipeline.
It keeps the key bridge outputs and the manifests needed for the next step.
The raw official files remain untouched in the parent directory.

## Structure

- `00_docs`
  - copied documentation and inventory files
- `01_keys`
  - canonical key outputs such as `student_master.csv`
- `02_manifests`
  - modeling-focused manifests for feature engineering and label design

## Ready Now

- `01_keys/student_master.csv`
  - canonical student table
  - 2500 rows
- `01_keys/account_master.csv`
  - account-level bridge table
  - 2498 rows currently resolve as `exact_login_alias`
- `01_keys/student_key_bridge.csv`
  - standardized source-key to `student_id` bridge
- `01_keys/student_key_bridge_summary.csv`
  - compact summary of bridge coverage

## V1 Modeling Scope

For the first trainable `student-semester` dataset, start with these sources:

- `student_master.csv`
- `学生成绩.xlsx`
- `学生选课信息.xlsx`
- `学籍异动.xlsx`
- `上网统计.xlsx`
- `奖学金获奖.xlsx`
- `本科生综合测评.xlsx`
- `体测数据.xlsx`
- `线上学习（综合表现）.xlsx`

These are enough to build:

- risk label candidates
- academic performance features
- semester-level behavior features
- strong baseline models

## Important Notes

- The current bridge export focuses on high-priority exact-id tables and the online-learning account alias.
- Large account-event tables such as `课堂任务参与.xlsx` and `学生签到记录.xlsx` are not fully bridged yet in this pass.
- This is intentional: it keeps the pipeline stable and lets us move directly into sample-table construction.
- If needed later, those large event tables can be added with a faster parser or a Python-based ETL step.

## Immediate Next Output

The next file we should build is:

- `student_semester_base.csv`

Recommended granularity:

- one row per `student_id + school_year + semester`

Recommended core columns:

- `student_id`
- `school_year`
- `semester`
- `risk_label`
- `gpa_like_features`
- `fail_course_count`
- `internet_usage_features`
- `scholarship_or_eval_features`
- `physical_features`
- `online_learning_features`
