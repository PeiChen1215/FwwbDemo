# 基于多源数据的大学生行为分析与干预模型

本项目是第十七届中国大学生服务外包创新创业大赛A类赛题（赛题A14）的参赛作品，利用多源学生数据构建行为分析与学业风险预测模型。

## 项目亮点

- **自监督学习 (SSL)**: 使用 TabularS3L 的 DAE + Transformer 进行预训练
- **创新架构**: Transformer编码器结合可学习因果邻接矩阵和行注意力
- **NNCLR损失**: 引入Memory Bank的最近邻对比学习
- **HDBSCAN聚类**: 识别学生行为模式，发现4类学生群体
- **AUC 0.8373**: 超过赛题要求的0.80基准

## 项目结构

```
.
├── README.md                 # 本文件
├── requirements.txt          # Python依赖
├── .gitignore               # Git忽略配置
├── src/                     # 源代码包
│   ├── models/             # 模型定义
│   │   ├── losses.py       # NNCLR对比损失
│   │   ├── scarf.py        # SCARF模型工具
│   │   ├── transformer.py  # 因果Transformer编码器
│   │   └── scarf_lightning.py  # Lightning模块
│   ├── data/               # 数据处理
│   └── utils/              # 工具函数
├── scripts/                # 可执行脚本
│   ├── train/             # 训练脚本
│   │   ├── train_ssl_transformer.py
│   │   ├── train_ssl_dae.py
│   │   ├── train_baseline.py
│   │   └── train_risk_model.py
│   ├── evaluate/          # 评估脚本
│   │   ├── compare_models.py
│   │   └── analyze_clusters.py
│   └── data/              # 数据处理脚本
│       ├── prepare_data.py
│       └── build_features.py
├── configs/               # 配置文件
├── outputs/               # 输出目录
│   ├── models/           # 模型权重
│   ├── results/          # 结果数据
│   └── figures/          # 可视化图表
├── tests/                 # 测试代码
└── docs/                  # 文档
    ├── 技术栈方案.md
    └── 项目评估报告.md
```

## 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv fwwb_env

# 激活环境
.\fwwb_env\Scripts\activate  # Windows
source fwwb_env/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

```bash
python scripts/data/prepare_data.py
python scripts/data/build_features.py
```

### 训练SSL模型

```bash
# 训练DAE+Transformer模型
python scripts/train/train_ssl_transformer.py
```

### 评估模型

```bash
# 对比不同模型性能
python scripts/evaluate/compare_models.py

# 分析聚类结果
python scripts/evaluate/analyze_clusters.py
```

## 核心结果

### 聚类分析（4类学生群体）

| 群体 | 数量 | 占比 | GPA | 挂科率 | 特征描述 |
|------|------|------|-----|--------|----------|
| Outliers | 13 | 0.5% | 1.44 | 34% | 异常行为，需紧急干预 |
| Cluster 0 | 1153 | 46.1% | 3.42 | 0% | 低调学霸，高网络使用但成绩优异 |
| Cluster 1 | 1212 | 48.5% | 3.38 | 0% | 活跃学霸，低网络使用 |
| Cluster 2 | 122 | 4.9% | 1.74 | 16% | 中等风险群体 |

### 模型性能对比

| 方法 | AUC | F1 |
|------|-----|-----|
| Raw Features + LogisticRegression | 0.7972 | 0.04 |
| **SSL Embeddings + LogisticRegression** | **0.8373** | **0.04** |

## 技术栈

- **深度学习**: PyTorch, PyTorch Lightning
- **自监督学习**: TabularS3L (DAE, SCARF, SubTab)
- **聚类**: HDBSCAN
- **机器学习**: scikit-learn, XGBoost
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn

## 作者

PeiChen1215

## 许可证

MIT License
