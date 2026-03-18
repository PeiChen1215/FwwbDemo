# 项目架构文档

## 1. 项目概述

本项目是基于多源数据的大学生行为分析与干预模型，利用自监督学习（SSL）技术从大规模无标签数据中学习学生行为表示，实现学业风险预警和学生分群。

### 1.1 核心指标

| 指标 | 数值 | 说明 |
|------|------|------|
| AUC | 0.8373 | 超过赛题要求(≥0.80) |
| 准确率 | 79.5% | 二分类任务 |
| 学生群体 | 4类 | 通过HDBSCAN聚类识别 |
| 特征维度 | 64维 | SSL嵌入向量 |

### 1.2 技术栈

```
深度学习框架: PyTorch + PyTorch Lightning
自监督学习: TabularS3L (DAE + Transformer)
聚类算法: HDBSCAN (密度聚类)
机器学习: scikit-learn, XGBoost
数据处理: pandas, numpy
可视化: matplotlib, seaborn
```

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           大学生行为分析系统                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │   数据层     │───▶│   特征层     │───▶│   模型层     │───▶│  应用层   │  │
│  │  Data Layer │    │ Feature Layer│    │  Model Layer │    │ App Layer│  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流图

```
原始数据 (Excel 27个文件)
    │
    ▼
┌─────────────────┐
│ 数据清洗与整合   │  ◀── scripts/data/prepare_data.py
│ Data Cleaning   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ 学生学期基表    │────▶│  TabularS3L    │  ◀── src/models/scarf.py
│ student_semester│     │   输入格式      │
└────────┬────────┘     └────────┬────────┘
         │                        │
         │         ┌──────────────┴──────────────┐
         │         │                               │
         ▼         ▼                               ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│  自监督预训练 (SSL)      │  │    有监督微调           │
│  Self-Supervised        │  │    Fine-tuning          │
│  Pre-training           │  │                         │
│                         │  │                         │
│  • DAE (去噪自编码器)    │  │  • 风险预测分类器        │
│  • Transformer Backbone  │  │  • 二分类 (挂科/正常)    │
│  • NNCLR 对比学习        │  │                         │
└───────────┬─────────────┘  └─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│    64维嵌入向量         │
│   64-dim Embeddings     │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
┌──────────┐  ┌──────────────┐
│ HDBSCAN  │  │  下游任务     │
│ 聚类     │  │  Downstream  │
│          │  │              │
│ 4类学生  │  │ • 风险预测    │
│ 群体     │  │ • 行为分析    │
│          │  │ • 干预建议    │
└──────────┘  └──────────────┘
```

## 3. 模块详细设计

### 3.1 数据层 (Data Layer)

#### 3.1.1 数据源

| 数据类型 | 文件数量 | 覆盖率 | 用途 |
|----------|----------|--------|------|
| 学生基本信息 | 1 | 100% | 人口统计学特征 |
| 成绩数据 | 3 | 100% | GPA、挂科记录 |
| 上网统计 | 1 | 100% | 网络使用行为 |
| 选课信息 | 1 | 100% | 课程偏好 |
| 图书馆打卡 | 1 | 19% | 学习努力度 |
| 跑步打卡 | 1 | 21% | 体育健康 |
| 其他 | 19 | 不定 | 社团、奖学金等 |

#### 3.1.2 数据处理流程

```python
# scripts/data/prepare_data.py
def prepare_student_semester_base():
    """
    构建学生-学期基表，核心步骤：
    
    1. 数据读取 (read_excel)
       ├── 处理不同编码格式 (utf-8, gbk)
       └── 统一列名格式
       
    2. 主键映射 (build_key_bridge)
       ├── 统一主键：学号 + 学期
       └── 处理缺失值、异常值
       
    3. 特征工程 (feature_engineering)
       ├── 统计特征：均值、方差、最大值
       ├── 时序特征：趋势、变化率
       └── 交叉特征：成绩×上网时长
       
    4. 标签构建 (build_labels)
       ├── risk_label_next_term: 下学期是否有挂科
       └── 基于成绩表自动标注
    
    Output: student_semester_base.csv (17k+ 样本)
    """
    pass
```

### 3.2 特征层 (Feature Layer)

#### 3.2.1 特征分类

```
学生行为特征 (15维原始特征)
│
├── 学业特征 (5维)
│   ├── avg_score          # 平均成绩
│   ├── fail_ratio         # 挂科率
│   ├── credit_earned      # 已获得学分
│   ├── course_count       # 选课数量
│   └── gpa_trend          # GPA变化趋势
│
├── 网络行为特征 (5维)
│   ├── internet_duration  # 上网时长
│   ├── internet_flow      # 上网流量
│   ├── night_usage_ratio  # 夜间使用比例
│   ├── weekday_weekend_ratio  # 工作日/周末比
│   └── unique_sites       # 访问网站多样性
│
└── 其他行为特征 (5维)
    ├── library_visits     # 图书馆访问次数
    ├── running_distance   # 跑步距离
    ├── scholarship_count  # 奖学金次数
│   ├── activity_count     # 社团活动次数
│   └── punishment_count   # 违纪次数
```

#### 3.2.2 四模态特征工程

```python
# scripts/data/build_features.py
def build_four_mode_features():
    """
    构建四模态特征表示：
    
    Mode 1: 学业表现 (Academic)
        - 成绩统计、GPA、挂科记录
        
    Mode 2: 网络行为 (Internet)
        - 上网时长、流量、时间分布
        
    Mode 3: 学习努力度 (Effort)
        - 图书馆、作业提交、考勤
        
    Mode 4: 综合素质 (Comprehensive)
        - 体育、社团、奖学金
        
    Output: 4组特征向量，用于对比学习
    """
    pass
```

### 3.3 模型层 (Model Layer)

#### 3.3.1 自监督学习架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    自监督学习 (SSL) 架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: 学生特征向量 x ∈ R^15                                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 数据增强 (Augmentation)                  │   │
│  │                                                          │   │
│  │  x ──▶ [Random Mask 30%] ──▶ x_corr (损坏视图)          │   │
│  │                                                          │   │
│  │  策略:                                                   │   │
│  │  • 连续特征: 添加高斯噪声                                │   │
│  │  • 离散特征: 随机替换为其他值                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              特征嵌入 (Feature Tokenizer)                │   │
│  │                                                          │   │
│  │  x ──▶ FT-Embedding ──▶ tokens ∈ R^(seq_len × d_model) │   │
│  │                                                          │   │
│  │  将15维特征映射为可学习的嵌入向量                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Transformer 编码器 (核心创新)                │   │
│  │                                                          │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  创新1: 可学习因果邻接矩阵 (Learnable Causal    │   │   │
│  │  │         Adjacency Matrix)                       │   │   │
│  │  │                                                 │   │   │
│  │  │  • 通过 NOTEARS 算法学习特征间的因果关系        │   │   │
│  │  │  • 生成因果掩码，约束注意力权重                 │   │   │
│  │  │  • DAG惩罚项: tr(e^(W⊙W)) - d                  │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                          │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  创新2: 行注意力 (Row Attention)                 │   │   │
│  │  │                                                 │   │   │
│  │  │  • 在batch维度上计算学生间相似度                │   │   │
│  │  │  • 融入学生社交拓扑信息                         │   │   │
│  │  │  • 仅在训练时使用，推理时关闭                   │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                          │   │
│  │  Output: 64维嵌入向量 z ∈ R^64                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              NNCLR 对比学习损失 (核心创新)                │   │
│  │                                                          │   │
│  │  Memory Bank (队列大小: 4096)                            │   │
│  │       │                                                  │   │
│  │       ├──▶ 存储历史批次的学生嵌入                        │   │
│  │       ├──▶ 为每个样本找最近邻 (Nearest Neighbor)        │   │
│  │       └──▶ 打破"自己跟自己比"的限制                     │   │
│  │                                                          │   │
│  │  Loss = -log[ exp(sim(z_i, nn_i)/τ) /                   │   │
│  │                Σ exp(sim(z_i, z_j)/τ) ]                 │   │
│  │                                                          │   │
│  │  其中 nn_i 是从Memory Bank中找到的最近邻                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.3.2 模型类图

```
┌─────────────────────────────────────────────────────────────────┐
│                         模型类图                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐         ┌─────────────────────┐       │
│  │   NTXentLoss        │         │ TransformerEncoder  │       │
│  │   ─────────────     │         │   ────────────────  │       │
│  │   NNCLR对比损失     │         │   Transformer编码器 │       │
│  │                     │         │                     │       │
│  │   + queue: Tensor   │         │   + adj_matrix: Param│      │
│  │   + queue_ptr: int  │         │   + row_attn: Module│       │
│  │                     │         │                     │       │
│  │   + forward(z_i, z_j)│        │   + forward(x)      │       │
│  │   + _dequeue_and_enqueue()│   │   + get_dag_penalty()│      │
│  └──────────┬──────────┘         └──────────┬──────────┘       │
│             │                               │                   │
│             │         ┌─────────────────────┘                   │
│             │         │                                         │
│             ▼         ▼                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │          SCARFLightning                 │                   │
│  │          ─────────────                  │                   │
│  │          SCARF Lightning模块            │                   │
│  │                                         │                   │
│  │  - contrastive_loss: NTXentLoss         │                   │
│  │  - model: SCARF                         │                   │
│  │                                         │                   │
│  │  + _get_first_phase_loss()              │                   │
│  │  + _get_second_phase_loss()             │                   │
│  │  + set_second_phase()                   │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 应用层 (Application Layer)

#### 3.4.1 学业风险预测

```python
# scripts/train/train_risk_model.py
class RiskPredictionPipeline:
    """
    学业风险预测流程：
    
    1. 特征提取
       Input: student_semester_base.csv
       └── SSL Encoder (frozen)
           └── 64-dim embeddings
    
    2. 分类器训练
       Model: LogisticRegression / XGBoost
       Input: 64-dim embeddings
       Output: risk_probability (0-1)
       
    3. 评估指标
       ├── AUC: 0.8373
       ├── Accuracy: 0.795
       ├── Precision: 0.12 (类别不平衡)
       └── Recall: 0.68
    """
```

#### 3.4.2 学生聚类分析

```python
# scripts/evaluate/analyze_clusters.py
class StudentClustering:
    """
    学生聚类分析流程：
    
    1. 降维
       Input: 64-dim embeddings
       ├── PCA (降维到10维)
       └── t-SNE (可视化到2维)
    
    2. 聚类
       Algorithm: HDBSCAN
       ├── min_cluster_size: 3% of data
       ├── min_samples: 5
       └── metric: euclidean
       
    3. 结果: 4类学生群体
    
    Cluster -1 (Outliers): 异常学生 (需紧急干预)
        ├── Count: 13 (0.5%)
        ├── Avg GPA: 1.44
        └── Fail Rate: 34%
        
    Cluster 0 (Low-profile Excellent): 低调学霸
        ├── Count: 1153 (46.1%)
        ├── Avg GPA: 3.42
        ├── Fail Rate: 0%
        └── Feature: 高网络使用但成绩优异
        
    Cluster 1 (Active Excellent): 活跃学霸
        ├── Count: 1212 (48.5%)
        ├── Avg GPA: 3.38
        ├── Fail Rate: 0%
        └── Feature: 传统优秀学生，低网络使用
        
    Cluster 2 (Medium Risk): 中等风险
        ├── Count: 122 (4.9%)
        ├── Avg GPA: 1.74
        └── Fail Rate: 16%
    """
```

## 4. 关键算法详解

### 4.1 因果邻接矩阵 (Causal Adjacency Matrix)

```python
# src/models/transformer.py

def get_dag_penalty(self, seq_len: int) -> torch.Tensor:
    """
    NOTEARS 算法的迹约束实现。
    
    目标: 学习特征间的因果关系，确保邻接矩阵是无环图(DAG)
    
    数学原理:
    1. W = softplus(adj_matrix)  # 保证非负
    2. W = W - diag(W)           # 消除自环
    3. M = W ⊙ W               # Hadamard积
    4. E = e^M                  # 矩阵指数
    5. penalty = tr(E) - d      # 迹约束
    
    当且仅当 W 是有向无环图时，penalty = 0
    """
    adj = self.adj_matrix[:seq_len, :seq_len]
    W = F.softplus(adj)
    W = W - torch.diag_embed(torch.diagonal(W))
    M = W * W
    E = torch.matrix_exp(M)
    trace = torch.trace(E)
    return trace - seq_len
```

### 4.2 NNCLR 对比学习

```python
# src/models/losses.py

def forward(self, z_i, z_j):
    """
    NNCLR (Nearest Neighbor Contrastive Learning)
    
    创新点: 使用Memory Bank中的最近邻作为正样本，
           而非传统的"自己跟自己比"
    
    Args:
        z_i: 原始样本嵌入 [batch_size, dim]
        z_j: 损坏视图嵌入 [batch_size, dim]
    
    Returns:
        loss: 对比损失
        
    流程:
    1. L2归一化 z_i
    2. 从Memory Bank中找最近邻 nn_i
    3. 计算 nn_i 与 z_j 的对比损失
    4. 更新Memory Bank
    """
    z_i_norm = F.normalize(z_i, dim=1)
    
    # 从队列中找最近邻
    if self.queue is not None:
        sim_with_queue = torch.matmul(z_i_norm, self.queue.T)
        _, nn_idx = torch.max(sim_with_queue, dim=1)
        nn_i = self.queue[nn_idx]
    else:
        nn_i = z_i
    
    # 计算对比损失...
    self._dequeue_and_enqueue(z_i_norm)
    return loss
```

## 5. 数据流详细说明

### 5.1 训练流程

```
阶段1: 自监督预训练 (SSL Pre-training)
========================================
输入: student_semester_base.csv (17k+ 无标签样本)
│
├─▶ DataModule
│   ├── 划分 train/val/test (70/15/15)
│   └── 创建 DAEDataset (连续特征 + 类别特征)
│
├─▶ Model
│   ├── DAELightning
│   │   ├── First Phase (SSL)
│   │   │   ├── 数据增强 (30% mask)
│   │   │   ├── FT-Embedding (15 → 64 dim)
│   │   │   ├── TransformerEncoder (3 layers)
│   │   │   ├── NNCLR Loss + DAG Penalty
│   │   │   └── 200 epochs
│   │   │
│   │   └── Second Phase (Fine-tuning)
│   │       ├── 冻结 encoder
│   │       ├── 训练分类头
│   │       └── 20 epochs
│
└─▶ Output
    ├── 训练好的模型权重 (.pt)
    ├── 学生嵌入向量 (64-dim)
    └── 训练日志


阶段2: 聚类分析 (Clustering)
=============================
输入: 64-dim embeddings (2500 students)
│
├─▶ Dimensionality Reduction
│   ├── PCA (64 → 10 dim)
│   └── t-SNE (10 → 2 dim for visualization)
│
├─▶ HDBSCAN Clustering
│   ├── min_cluster_size = 3% * n_samples
│   ├── min_samples = 5
│   └── metric = 'euclidean'
│
└─▶ Output
    ├── Cluster labels (-1, 0, 1, 2)
    ├── Cluster statistics
    └── Visualization (PCA/t-SNE plots)


阶段3: 下游任务 (Downstream Tasks)
===================================
输入: 64-dim embeddings + labels
│
├─▶ Risk Prediction
│   ├── LogisticRegression (AUC: 0.8373)
│   ├── XGBoost (AUC: 0.82)
│   └── RandomForest (AUC: 0.70)
│
├─▶ Intervention Strategy
│   ├── Outliers (Cluster -1): 紧急干预
│   ├── Cluster 2: 预警提示
│   └── Clusters 0,1: 保持现状
│
└─▶ Output
    ├── Risk scores per student
    ├── Intervention recommendations
    └── Performance reports
```

## 6. 项目模块依赖关系

```
src/
│
├── models/
│   ├── __init__.py  ─────────▶  exports: NTXentLoss, TransformerEncoder, etc.
│   ├── losses.py    ─────────▶  无外部依赖 (纯PyTorch)
│   ├── transformer.py ───────▶  无外部依赖 (纯PyTorch)
│   ├── scarf_lightning.py ───▶  depends on: losses.py, ts3l (external)
│   └── scarf.py     ─────────▶  depends on: ts3l (external)
│
├── data/
│   └── __init__.py  ─────────▶  预留数据模块
│
└── utils/
    └── __init__.py  ─────────▶  预留工具函数

scripts/
│
├── train/
│   ├── train_ssl_transformer.py ──▶ depends on: src/models/*
│   ├── train_ssl_dae.py ──────────▶ depends on: src/models/*
│   ├── train_baseline.py ─────────▶ depends on: pandas, sklearn
│   ├── train_risk_model.py ───────▶ depends on: src/models/scarf.py
│   └── train_ssl_cluster.py ──────▶ depends on: src/models/*
│
├── evaluate/
│   ├── compare_models.py ─────────▶ depends on: sklearn
│   └── analyze_clusters.py ───────▶ depends on: hdbscan, matplotlib
│
└── data/
    ├── prepare_data.py ───────────▶ depends on: pandas
    └── build_features.py ─────────▶ depends on: pandas, numpy
```

## 7. 配置文件说明

### 7.1 超参数配置

```python
# configs/default.yaml (推荐添加)

# 数据配置
data:
  input_dir: "prepared/06_tabulars3l"
  output_dir: "outputs/results"
  batch_size: 512
  num_workers: 4
  
# 模型配置
model:
  latent_dim: 64
  encoder_depth: 3
  n_head: 4
  hidden_dim: 128
  corruption_rate: 0.3
  
# 训练配置
training:
  pretrain_epochs: 200
  finetune_epochs: 20
  learning_rate: 1e-3
  temperature: 0.5  # for contrastive loss
  queue_size: 4096  # for NNCLR
  dag_penalty_weight: 0.01
  
# 聚类配置
clustering:
  algorithm: "HDBSCAN"
  min_cluster_ratio: 0.03
  min_samples: 5
  metric: "euclidean"
```

## 8. 扩展性设计

### 8.1 添加新的特征

```python
# src/data/features.py (新建)

def add_library_features(df):
    """
    添加图书馆特征示例
    """
    df['library_frequency'] = df['library_visits'] / df['days_in_semester']
    df['library_avg_duration'] = df['library_total_time'] / df['library_visits']
    return df
```

### 8.2 添加新的模型

```python
# src/models/new_model.py (新建)

import torch.nn as nn

class NewEncoder(nn.Module):
    """
    添加新的编码器架构示例
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)
```

### 8.3 添加新的下游任务

```python
# scripts/train/train_new_task.py (新建)

def train_new_task():
    """
    添加新的下游任务示例：学业成绩预测
    """
    # 1. 加载预训练模型
    model = load_pretrained_model()
    
    # 2. 提取特征
    embeddings = extract_embeddings(model, data)
    
    # 3. 训练新任务
    regressor = XGBRegressor()
    regressor.fit(embeddings, gpa_labels)
    
    # 4. 评估
    mse = mean_squared_error(test_labels, predictions)
```

## 9. 性能优化

### 9.1 训练速度优化

| 优化策略 | 实现方式 | 效果 |
|----------|----------|------|
| 增大batch size | BATCH_SIZE = 512 | 提升GPU利用率 |
| 多进程数据加载 | num_workers = 4 | 减少IO等待 |
| 混合精度训练 | torch.cuda.amp | 减少显存占用 |
| 提前停止 | EarlyStopping callback | 避免过拟合 |

### 9.2 内存优化

| 优化策略 | 实现方式 | 效果 |
|----------|----------|------|
| 梯度累积 | accumulate_grad_batches | 模拟大batch |
| 检查点保存 | save_top_k = 3 | 只保存最佳模型 |
| 数据分块 | chunked DataLoader | 减少内存峰值 |

## 10. 常见问题排查

### Q1: TabularS3L导入失败
```bash
# 解决方案
pip install -e ./TabularS3L-main/TabularS3L-main
```

### Q2: HDBSCAN安装失败
```bash
# Windows用户
conda install -c conda-forge hdbscan

# Linux/Mac
pip install hdbscan
```

### Q3: 显存不足
```python
# 减小batch size
BATCH_SIZE = 256  # 原为512

# 或使用CPU训练
 trainer = pl.Trainer(accelerator="cpu")
```

---

*文档版本: 1.0*  
*最后更新: 2026-03-18*  
*作者: PeiChen1215*
