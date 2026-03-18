#!/usr/bin/env python3
"""
快速训练脚本 - 10分钟完成全流程验证
Quick Training - Complete pipeline in ~10 minutes
"""
import subprocess
import sys
from pathlib import Path

def run(script, desc):
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print('='*60)
    
    python = Path("fwwb_env/Scripts/python.exe").resolve()
    result = subprocess.run([str(python), script], cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"[WARNING] {desc} failed, continuing...")
    return result.returncode == 0

def main():
    print("""
============================================================
         Quick Training Pipeline (10 epochs mode)
============================================================
    """)
    
    steps = [
        ("scripts/train/train_baseline.py", "Step 1/4: Baseline Models"),
        ("scripts/evaluate/compare_models.py", "Step 2/4: Model Comparison"),
        ("scripts/evaluate/analyze_clusters.py", "Step 3/4: Clustering Analysis"),
    ]
    
    for script, desc in steps:
        run(script, desc)
    
    print("""
============================================================
               Quick training completed!
============================================================

Results saved to:
  - outputs/results/
  - model_comparison_results.csv

For full training (200 epochs), run:
  python scripts/train/train_ssl_transformer.py
    """)

if __name__ == "__main__":
    main()
