"""
对比实验：原始特征 vs SSL嵌入特征
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

print("="*60)
print("Model Comparison: Raw Features vs SSL Embeddings")
print("="*60)

# 加载数据
print("\n[1] Loading data...")
base_df = pd.read_csv("prepared/03_datasets/student_semester_base.csv")
emb_df = pd.read_csv("prepared/06_tabulars3l/tabulars3l_semester_embeddings.csv")

df = base_df[base_df['risk_label_next_term'].notna()].copy()
df['risk_label_next_term'] = df['risk_label_next_term'].astype(int)
print(f"  Samples: {len(df)}, Positive: {df['risk_label_next_term'].sum()}")

# 原始特征
raw_numeric = ['selected_course_count', 'score_course_count', 'avg_score', 'score_std',
               'fail_course_count', 'fail_ratio', 'avg_gpa', 'credit_sum', 'resit_exam_count',
               'internet_month_count', 'internet_hours_sum', 'internet_hours_avg_per_month',
               'internet_diff_mean', 'online_learning_bfb_snapshot', 'physical_test_score']
for col in raw_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

X_raw = df[raw_numeric].values

# SSL嵌入特征
emb_cols = [c for c in emb_df.columns if c.startswith('emb_') and not c.startswith('emb_scaled')]
print(f"  Raw features: {len(raw_numeric)}, SSL dims: {len(emb_cols)}")

df_merged = df.merge(emb_df[['student_id', 'school_year', 'semester'] + emb_cols], 
                     on=['student_id', 'school_year', 'semester'], how='left')
X_emb = df_merged[emb_cols].fillna(0).values

y = df['risk_label_next_term'].values

# 划分数据
X_raw_train, X_raw_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)
X_emb_train, X_emb_test, _, _ = train_test_split(X_emb, y, test_size=0.2, random_state=42, stratify=y)

print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

# 训练并评估
print("\n[2] Training models...")
results = []

# Raw + RF
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_raw_train, y_train)
auc = roc_auc_score(y_test, rf.predict_proba(X_raw_test)[:, 1])
results.append(('Raw + RandomForest', auc))
print(f"  Raw + RF: AUC = {auc:.4f}")

# Raw + LR
scaler = StandardScaler()
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(scaler.fit_transform(X_raw_train), y_train)
auc = roc_auc_score(y_test, lr.predict_proba(scaler.transform(X_raw_test))[:, 1])
results.append(('Raw + LogisticReg', auc))
print(f"  Raw + LR: AUC = {auc:.4f}")

# SSL + RF
rf2 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf2.fit(X_emb_train, y_train)
auc = roc_auc_score(y_test, rf2.predict_proba(X_emb_test)[:, 1])
results.append(('SSL + RandomForest', auc))
print(f"  SSL + RF: AUC = {auc:.4f}")

# SSL + LR
scaler2 = StandardScaler()
lr2 = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr2.fit(scaler2.fit_transform(X_emb_train), y_train)
auc = roc_auc_score(y_test, lr2.predict_proba(scaler2.transform(X_emb_test))[:, 1])
results.append(('SSL + LogisticReg', auc))
print(f"  SSL + LR: AUC = {auc:.4f}")

# 结果汇总
print("\n" + "="*60)
print("Results Summary")
print("="*60)
results_df = pd.DataFrame(results, columns=['Model', 'AUC']).sort_values('AUC', ascending=False)

for i, row in results_df.iterrows():
    status = "[OK]" if row['AUC'] >= 0.80 else "[LOW]"
    print(f"  {row['Model']:25s} AUC={row['AUC']:.4f} {status}")

best_auc = results_df['AUC'].max()
print("\n" + "="*60)
if best_auc >= 0.80:
    print(f"SUCCESS! Best AUC = {best_auc:.4f} >= 0.80")
else:
    print(f"Best AUC = {best_auc:.4f}, target is 0.80")
print("="*60)

# 保存
results_df.to_csv('comparison_results.csv', index=False)
print("\nSaved to: comparison_results.csv")
