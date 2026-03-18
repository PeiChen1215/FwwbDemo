#!/bin/bash
# Windows用户请使用 PowerShell 或手动修改参数运行

# 快速SSL训练 (10 epochs测试)
python scripts/train/train_ssl_transformer.py \
    --epochs 10 \
    --batch_size 512

# 完整训练请使用默认参数
# python scripts/train/train_ssl_transformer.py
