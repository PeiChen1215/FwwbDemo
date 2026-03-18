# 基于多源数据的大学生行为分析与干预模型

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

本项目是第十七届中国大学生服务外包创新创业大赛A类赛题（赛题A14）的参赛作品，利用多源学生数据构建行为分析与学业风险预测模型，通过自监督学习技术从大规模无标签数据中学习学生行为表示。

---

## 核心亮点

### 技术创新
| 创新点 | 描述 | 效果 |
|--------|------|------|
| **自监督学习** | TabularS3L DAE + Transformer预训练 | 利用17k+无标签样本 |
| **因果Transformer** | 可学习因果邻接矩阵 + 行注意力 | 捕捉特征因果关系 |
| **NNCLR损失** | Memory Bank + 最近邻对比学习 | 提升表示质量 |
| **HDBSCAN聚类** | 密度聚类识别行为模式 | 发现4类学生群体 |

### 性能指标
- **AUC**: 0.8373 (超过赛题要求 ≥0.80)
- **准确率**: 79.5%
- **学生分群**: 4类群体精准识别
- **特征维度**: 64维SSL嵌入向量

---

## 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/PeiChen1215/FwwbDemo.git
cd FwwbDemo

# 创建虚拟环境
python -m venv fwwb_env

# 激活环境 (Windows)
.\fwwb_env\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 一键运行

```bash
# 快速测试 (~5分钟)
python run_quick_test.py

# 快速训练 (~15分钟, 10 epochs)
python run_full_pipeline.py --quick

# 完整训练 (~60分钟, 200 epochs)
python run_full_pipeline.py
```

### 3. 单独运行脚本

```bash
# 基线模型 (5-10分钟)
python scripts/train/train_baseline.py

# SSL训练 (30-60分钟)
python scripts/train/train_ssl_transformer.py

# 模型对比
python scripts/evaluate/compare_models.py
```

---

## 项目结构

```
FwwbDemo/
├── README.md                       # 项目说明
├── requirements.txt                # Python依赖
├── run_quick_test.py              # 快速测试脚本
├── run_full_pipeline.py           # 完整训练管道
├── train_all_quick.py             # 快速训练脚本
│
├── src/                           # 源代码包
│   ├── models/                    # 模型定义
│   │   ├── losses.py             # NNCLR对比损失
│   │   ├── transformer.py        # 因果Transformer
│   │   ├── scarf.py              # SCARF工具
│   │   └── scarf_lightning.py    # Lightning模块
│   ├── data/                      # 数据处理
│   └── utils/                     # 工具函数
│
├── scripts/                       # 可执行脚本
│   ├── train/                    # 训练脚本
│   │   ├── train_ssl_transformer.py    # 主SSL训练
│   │   ├── train_ssl_dae.py           # DAE版本
│   │   ├── train_baseline.py          # 基线模型
│   │   ├── train_risk_model.py        # 风险预测
│   │   └── train_ssl_cluster.py       # SSL+聚类
│   ├── evaluate/                 # 评估脚本
│   │   ├── compare_models.py         # 模型对比
│   │   └── analyze_clusters.py       # 聚类分析
│   └── data/                     # 数据处理
│       ├── prepare_data.py       # 数据准备
│       └── build_features.py     # 特征工程
│
├── prepared/                      # 数据目录
│   ├── 03_datasets/              # 原始数据
│   └── 06_tabulars3l/            # 处理后的数据
│
├── outputs/                       # 输出目录
│   ├── models/                   # 模型权重
│   ├── results/                  # 结果文件
│   └── figures/                  # 可视化图表
│
├── docs/                          # 文档
│   ├── ARCHITECTURE.md           # 架构文档
│   └── ...                       # 其他文档
│
└── tests/                         # 测试代码
```

---

## 核心结果

### 模型性能对比

| 排名 | 模型 | AUC | 状态 |
|:----:|------|:-----:|:----:|
| 🥇 | **SSL + LogisticRegression** | **0.8373** | ✅ 达标 |
| 🥈 | Raw + LogisticRegression | 0.7441 | ⚠️ |
| 🥉 | Raw + RandomForest | 0.7243 | ⚠️ |
| 4 | Combined + RandomForest | 0.7229 | ⚠️ |
| 5 | SSL + RandomForest | 0.6984 | ⚠️ |

**结论**: SSL嵌入显著提升模型性能，AUC从0.74提升至0.84。

### 学生聚类结果 (HDBSCAN)

通过SSL嵌入向量进行密度聚类，识别出4类学生群体：

| 群体 | 数量 | 占比 | GPA | 挂科率 | 特征描述 | 干预策略 |
|:----:|:----:|:----:|:---:|:------:|----------|----------|
| -1 | 13 | 0.5% | 1.44 | 34% | 异常行为，严重偏离 | 🔴 紧急干预 |
| 0 | 1153 | 46.1% | 3.42 | 0% | 低调学霸，高网络使用但成绩优异 | 🟢 保持现状 |
| 1 | 1212 | 48.5% | 3.38 | 0% | 活跃学霸，传统好学生 | 🟢 保持现状 |
| 2 | 122 | 4.9% | 1.74 | 16% | 中等风险，学习困难 | 🟡 预警关注 |

**干预建议**:
- **Outliers (13人)**: 立即约谈，了解异常情况
- **Cluster 2 (122人)**: 学习预警，提供辅导资源
- **Clusters 0,1**: 维持现有支持体系

---

## 技术架构

### 自监督学习流程

```
原始特征 (15维)
    │
    ▼ 数据增强 (30% mask)
损坏视图
    │
    ▼ FT-Embedding
特征令牌 (seq_len × 64)
    │
    ▼ Transformer Encoder
    │   ├── 因果邻接矩阵 (DAG约束)
    │   └── 行注意力 (学生间关系)
    │
64维嵌入向量
    │
    ▼ NNCLR Loss
    │   ├── Memory Bank (4096)
    │   └── 最近邻正样本
    │
对比学习训练
```

### 关键技术组件

| 组件 | 文件 | 说明 |
|------|------|------|
| **NNCLR损失** | `src/models/losses.py` | Memory Bank + 最近邻对比学习 |
| **因果Transformer** | `src/models/transformer.py` | 可学习因果邻接矩阵 |
| **SCARF Lightning** | `src/models/scarf_lightning.py` | PyTorch Lightning模块 |

---

## 训练脚本说明

### 主要脚本

| 脚本 | 用途 | 运行时间 | 输出 |
|------|------|---------|------|
| `train_ssl_transformer.py` | 主SSL训练 | 30-60分钟 | 64维嵌入、聚类结果 |
| `train_baseline.py` | 基线模型 | 5-10分钟 | LR/RF模型、评估指标 |
| `train_risk_model.py` | 风险预测 | 10-15分钟 | 预测结果、AUC报告 |
| `compare_models.py` | 模型对比 | 2-5分钟 | 对比表格、可视化 |
| `analyze_clusters.py` | 聚类分析 | 3-5分钟 | 群体画像、统计报告 |

### 超参数配置

```python
# train_ssl_transformer.py
PRETRAIN_EPOCHS = 200      # 预训练轮数 (quick模式: 10)
BATCH_SIZE = 512           # 批次大小
LATENT_DIM = 64            # 嵌入维度
ENCODER_DEPTH = 3          # Transformer层数
N_HEAD = 4                 # 注意力头数
CORRUPTION_RATE = 0.3      # 数据损坏率
QUEUE_SIZE = 4096          # Memory Bank大小
```

---

## 常见问题

### Q1: 如何快速验证项目能运行？
```bash
python run_quick_test.py
# 预期输出: AUC ~0.83，说明环境正常
```

### Q2: 没有GPU可以运行吗？
可以！项目已配置CPU优化：
```python
os.environ["OMP_NUM_THREADS"] = "8"
BATCH_SIZE = 512
```

### Q3: TabularS3L安装失败？
```bash
# 手动安装本地版本
cd TabularS3L-main/TabularS3L-main
pip install -e .
```

### Q4: 训练时间太长？
使用快速模式：
```bash
python run_full_pipeline.py --quick
# 或修改脚本中的 PRETRAIN_EPOCHS = 10
```

---

## 贡献指南

欢迎提交Issue和PR！

### 添加新特征
```python
# scripts/data/build_features.py
def add_new_features(df):
    df['new_feature'] = df['col1'] / df['col2']
    return df
```

### 添加新模型
```python
# src/models/new_model.py
class NewEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
```

---

## 许可证

MIT License - 详见 [LICENSE](docs/LICENSE)

---

## 联系方式

- 作者: PeiChen1215
- 项目地址: https://github.com/PeiChen1215/FwwbDemo
- 赛题: 第十七届中国大学生服务外包创新创业大赛 A14

---

## 致谢

- [TabularS3L](https://github.com/kimjw2003/TabularS3L) - 自监督学习框架
- [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) - 密度聚类算法
- [PyTorch Lightning](https://lightning.ai/) - 深度学习框架
