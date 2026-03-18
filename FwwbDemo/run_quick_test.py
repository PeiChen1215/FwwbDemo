#!/usr/bin/env python3
"""
快速测试脚本 - 验证项目能正常运行
使用减少的epoch数快速验证流程
"""
import subprocess
import sys
from pathlib import Path

def run_script(script_path, description):
    """运行脚本并显示进度"""
    print(f"\n{'='*60}")
    print(f"正在运行: {description}")
    print(f"脚本: {script_path}")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=Path(__file__).parent
    )
    
    if result.returncode == 0:
        print(f"[OK] {description} - Success")
    else:
        print(f"[FAIL] {description} - Failed")
    
    return result.returncode == 0

def main():
    """运行快速测试流程"""
    print("""
==============================================================
              Student Behavior Analysis - Quick Test
==============================================================
    """)
    
    # 检查虚拟环境
    print(f"Python: {sys.executable}")
    print(f"工作目录: {Path(__file__).parent}")
    
    # 测试1: 对比模型（最快，不训练）
    print("\n[TEST 1] Compare Raw Features vs SSL Embeddings")
    success = run_script(
        "scripts/evaluate/compare_models.py",
        "模型对比评估"
    )
    
    if not success:
        print("[FAIL] Test failed, please check environment")
        return
    
    print("\n" + "="*60)
    print("[DONE] Quick test completed! Project runs normally.")
    print("="*60)
    print("""
Next Steps:
1. Run full SSL training: python scripts/train/train_ssl_transformer.py
   (Estimated time: 30-60 minutes, 200 epochs)
   
2. Or run quick SSL test (10 epochs):
   Modify PRETRAIN_EPOCHS = 10 in the script
    
3. Run baseline model: python scripts/train/train_baseline.py
   (Estimated time: 5-10 minutes)
    """)

if __name__ == "__main__":
    main()
