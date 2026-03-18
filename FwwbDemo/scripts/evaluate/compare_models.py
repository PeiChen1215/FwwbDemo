"""
简化版：对比原始特征 vs SSL嵌入特征的模型性能
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

print("="*60)
print("Model Comparison: Raw Features vs SSL Embeddings")
print("="*60)

# 1. 加载数据
print("\n[1/5] Loading data...")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
base_df = pd.read_csv(BASE_DIR / "prepared" / "03_datasets" / "student_semester_base.csv")
emb_df = pd.read_csv(BASE_DIR / "prepared" / "06_tabulars3l" / "tabulars3l_semester_embeddings.csv")

# 过滤有效标签
df = base_df[base_df['risk_label_next_term'].notna()].copy()
df['risk_label_next_term'] = df['risk_label_next_term'].astype(int)

print(f"  Samples: {len(df)}")
print(f"  Positive: {df['risk_label_next_term'].sum()} ({df['risk_label_next_term'].mean()*100:.1f}%)")

# 2. 准备特征
print("\n[2/5] Preparing features...")

# 原始特征（数值+类别）
raw_numeric = ['selected_course_count', 'score_course_count', 'avg_score', 'score_std',
               'fail_course_count', 'fail_ratio', 'avg_gpa', 'credit_sum', 'resit_exam_count',
               'internet_month_count', 'internet_hours_sum', 'internet_hours_avg_per_month',
               'internet_diff_mean', 'online_learning_bfb_snapshot', 'physical_test_score']

# 处理缺失值
for col in raw_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# 类别特征编码
df['gender_encoded'] = df['gender'].map({'男': 0, '女': 1}).fillna(0)
df['semester_encoded'] = df['semester'].astype(str).str.replace('1', '0').str.replace('2', '1').fillna(0).astype(int)

raw_features = raw_numeric + ['gender_encoded', 'semester_encoded']
X_raw = df[raw_features].copy()

# SSL嵌入特征 (64维)
emb_cols = [c for c in emb_df.columns if c.startswith('emb_') and not c.startswith('emb_scaled')]
print(f"  Raw features: {len(raw_features)}")
print(f"  SSL embedding dims: {len(emb_cols)}")

# 合并嵌入特征
df_with_emb = df.merge(emb_df[['student_id', 'school_year', 'semester'] + emb_cols], 
                       on=['student_id', 'school_year', 'semester'], 
                       how='left')

# 填充缺失的embedding
df_with_emb[emb_cols] = df_with_emb[emb_cols].fillna(0)
X_emb = df_with_emb[emb_cols].copy()

# 合并特征
X_combined = pd.concat([X_raw.reset_index(drop=True), X_emb.reset_index(drop=True)], axis=1)

y = df['risk_label_next_term'].values

# 3. 划分数据集
print("\n[3/5] Splitting data...")
X_raw_train, X_raw_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)
X_emb_train, X_emb_test, _, _ = train_test_split(X_emb, y, test_size=0.2, random_state=42, stratify=y)
X_comb_train, X_comb_test, _, _ = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

# 4. 训练模型
print("\n[4/5] Training models...")

results = []

# 标准化
scaler_raw = StandardScaler()
X_raw_train_s = scaler_raw.fit_transform(X_raw_train)
X_raw_test_s = scaler_raw.transform(X_raw_test)

scaler_emb = StandardScaler()
X_emb_train_s = scaler_emb.fit_transform(X_emb_train)
X_emb_test_s = scaler_emb.transform(X_emb_test)

scaler_comb = StandardScaler()
X_comb_train_s = scaler_comb.fit_transform(X_comb_train)
X_comb_test_s = scaler_comb.transform(X_comb_test)

# 模型1: 原始特征 + RandomForest
print("  Training Raw + RF...")
rf_raw = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_raw.fit(X_raw_train, y_train)
y_pred_rf_raw = rf_raw.predict_proba(X_raw_test)[:, 1]
auc_rf_raw = roc_auc_score(y_test, y_pred_rf_raw)
results.append(('Raw + RandomForest', auc_rf_raw))

# 模型2: 原始特征 + LogisticRegression
print("  Training Raw + LR...")
lr_raw = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_raw.fit(X_raw_train_s, y_train)
y_pred_lr_raw = lr_raw.predict_proba(X_raw_test_s)[:, 1]
auc_lr_raw = roc_auc_score(y_test, y_pred_lr_raw)
results.append(('Raw + LogisticReg', auc_lr_raw))

# 模型3: SSL Embedding + RandomForest
print("  Training SSL Emb + RF...")
rf_emb = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_emb.fit(X_emb_train, y_train)
y_pred_rf_emb = rf_emb.predict_proba(X_emb_test)[:, 1]
auc_rf_emb = roc_auc_score(y_test, y_pred_rf_emb)
results.append(('SSL Emb + RandomForest', auc_rf_emb))

# 模型4: SSL Embedding + LogisticRegression
print("  Training SSL Emb + LR...")
lr_emb = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_emb.fit(X_emb_train_s, y_train)
y_pred_lr_emb = lr_emb.predict_proba(X_emb_test_s)[:, 1]
auc_lr_emb = roc_auc_score(y_test, y_pred_lr_emb)
results.append(('SSL Emb + LogisticReg', auc_lr_emb))

# 模型5: Combined + RandomForest
print("  Training Combined + RF...")
rf_comb = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_comb.fit(X_comb_train, y_train)
y_pred_rf_comb = rf_comb.predict_proba(X_comb_test)[:, 1]
auc_rf_comb = roc_auc_score(y_test, y_pred_rf_comb)
results.append(('Combined + RandomForest', auc_rf_comb))

# 5. 结果对比
print("\n" + "="*60)
print("[5/5] Results Comparison")
print("="*60)

results_df = pd.DataFrame(results, columns=['Model', 'AUC'])
results_df = results_df.sort_values('AUC', ascending=False)

print("\nRanking:")
for i, (_, row) in enumerate(results_df.iterrows(), 1):
    marker = " [BEST]" if i == 1 else ""
    status = "[OK]" if row['AUC'] >= 0.80 else "[WARN]"
    print(f"  {i}. {row['Model']:30s} AUC={row['AUC']:.4f} {status}{marker}")

print("\n" + "="*60)
best_auc = results_df['AUC'].max()
if best_auc >= 0.80:
    print(f"[SUCCESS] Best AUC = {best_auc:.4f} (>= 0.80)")
else:
    print(f"[WARNING] Best AUC = {best_auc:.4f}, need improvement")
print("="*60)

# 保存结果
results_df.to_csv('model_comparison_results.csv', index=False)
print("\nResults saved to: model_comparison_results.csv")
