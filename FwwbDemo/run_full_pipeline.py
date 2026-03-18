#!/usr/bin/env python3
"""
完整训练管道 - 一键运行所有模型
Full Training Pipeline

执行顺序:
1. Data Preparation (if needed)
2. SSL Pre-training (DAE + Transformer)
3. Clustering Analysis (HDBSCAN)
4. Risk Prediction Model
5. Evaluation & Comparison

Usage:
    python run_full_pipeline.py [--quick] [--skip-ssl]
    
Options:
    --quick      Use 10 epochs for quick test (default: 200)
    --skip-ssl   Skip SSL training, use existing embeddings
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


class PipelineRunner:
    """训练管道运行器"""
    
    def __init__(self, quick_mode=False, skip_ssl=False):
        self.quick_mode = quick_mode
        self.skip_ssl = skip_ssl
        self.base_dir = Path(__file__).resolve().parent
        self.venv_python = self.base_dir / "fwwb_env" / "Scripts" / "python.exe"
        self.results = []
        
        # 检查虚拟环境
        if not self.venv_python.exists():
            print(f"[ERROR] Virtual environment not found: {self.venv_python}")
            print("Please create venv: python -m venv fwwb_env")
            sys.exit(1)
    
    def run_step(self, step_num, description, script_path, timeout=3600):
        """运行单个步骤"""
        print(f"\n{'='*70}")
        print(f"[{step_num}/5] {description}")
        print(f"Script: {script_path}")
        print(f"Time limit: {timeout//60} minutes")
        print('='*70)
        
        start_time = time.time()
        
        try:
            # 修改脚本中的epochs（如果是快速模式）
            if self.quick_mode and 'train_ssl' in script_path:
                # 读取脚本内容并临时修改epochs
                script_content = Path(script_path).read_text(encoding='utf-8')
                original_content = script_content
                script_content = script_content.replace(
                    'PRETRAIN_EPOCHS = 200', 
                    'PRETRAIN_EPOCHS = 10'
                )
                script_content = script_content.replace(
                    'PRETRAIN_EPOCHS = 100',
                    'PRETRAIN_EPOCHS = 10'
                )
                
                # 写回临时修改
                if script_content != original_content:
                    Path(script_path).write_text(script_content, encoding='utf-8')
                    print("[INFO] Quick mode: Set epochs to 10")
            
            # 运行脚本
            result = subprocess.run(
                [str(self.venv_python), str(script_path)],
                cwd=self.base_dir,
                capture_output=False,
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"\n[SUCCESS] Step {step_num} completed in {elapsed:.1f}s")
                self.results.append((step_num, description, "SUCCESS", elapsed))
                return True
            else:
                print(f"\n[FAILED] Step {step_num} failed with code {result.returncode}")
                self.results.append((step_num, description, "FAILED", elapsed))
                return False
                
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"\n[TIMEOUT] Step {step_num} timed out after {timeout//60} minutes")
            self.results.append((step_num, description, "TIMEOUT", elapsed))
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n[ERROR] Step {step_num} error: {e}")
            self.results.append((step_num, description, "ERROR", elapsed))
            return False
    
    def check_existing_data(self):
        """检查是否已有处理好的数据"""
        required_files = [
            self.base_dir / "prepared" / "03_datasets" / "student_semester_base.csv",
            self.base_dir / "prepared" / "06_tabulars3l" / "a14_tabulars3l_metadata.json",
        ]
        
        missing = [f.name for f in required_files if not f.exists()]
        
        if missing:
            print(f"[WARNING] Missing data files: {missing}")
            return False
        return True
    
    def check_existing_embeddings(self):
        """检查是否已有SSL嵌入"""
        embedding_files = [
            self.base_dir / "outputs" / "results" / "transformer_semester_embeddings.csv",
            self.base_dir / "prepared" / "06_tabulars3l" / "tabulars3l_semester_embeddings.csv",
        ]
        
        for f in embedding_files:
            if f.exists():
                print(f"[INFO] Found existing embeddings: {f}")
                return True
        return False
    
    def run_full_pipeline(self):
        """运行完整管道"""
        print("""
=================================================================
           Student Behavior Analysis - Full Training Pipeline
=================================================================
        """)
        
        print(f"Working directory: {self.base_dir}")
        print(f"Python: {self.venv_python}")
        print(f"Quick mode: {self.quick_mode}")
        print(f"Skip SSL: {self.skip_ssl}")
        
        # 检查数据
        if not self.check_existing_data():
            print("\n[ERROR] Data not prepared. Please run data preparation first.")
            return False
        
        # 检查是否已有嵌入（如果skip_ssl）
        if self.skip_ssl and not self.check_existing_embeddings():
            print("\n[WARNING] --skip-ssl specified but no embeddings found.")
            print("Will run SSL training anyway.")
            self.skip_ssl = False
        
        total_start = time.time()
        
        # Step 1: 基线模型（快速验证）
        self.run_step(
            1, 
            "Baseline Models (LR/RF)",
            "scripts/train/train_baseline.py",
            timeout=600  # 10分钟
        )
        
        # Step 2: SSL预训练（主训练）
        if not self.skip_ssl:
            ssl_timeout = 600 if self.quick_mode else 3600  # 10分钟或60分钟
            ssl_success = self.run_step(
                2,
                "SSL Pre-training (DAE + Transformer)",
                "scripts/train/train_ssl_transformer.py",
                timeout=ssl_timeout
            )
            
            if not ssl_success:
                print("\n[WARNING] SSL training failed or timed out.")
                print("Will try to use existing embeddings if available.")
        
        # Step 3: 模型对比
        self.run_step(
            3,
            "Model Comparison (Raw vs SSL)",
            "scripts/evaluate/compare_models.py",
            timeout=300  # 5分钟
        )
        
        # Step 4: 聚类分析
        self.run_step(
            4,
            "Clustering Analysis (HDBSCAN)",
            "scripts/evaluate/analyze_clusters.py",
            timeout=300  # 5分钟
        )
        
        # Step 5: 风险预测
        self.run_step(
            5,
            "Risk Prediction Model",
            "scripts/train/train_risk_model.py",
            timeout=600  # 10分钟
        )
        
        # 总结
        total_elapsed = time.time() - total_start
        self.print_summary(total_elapsed)
        
        return True
    
    def print_summary(self, total_time):
        """打印总结报告"""
        print(f"\n{'='*70}")
        print("                         PIPELINE SUMMARY")
        print('='*70)
        
        for step_num, description, status, elapsed in self.results:
            status_icon = "[OK]" if status == "SUCCESS" else f"[{status}]"
            print(f"  Step {step_num}: {description:40s} {status_icon:8s} ({elapsed:.1f}s)")
        
        print(f"\n  Total time: {total_time//60:.0f}m {total_time%60:.0f}s")
        
        # 检查结果文件
        print(f"\n{'='*70}")
        print("                         OUTPUT FILES")
        print('='*70)
        
        output_files = [
            ("Baseline metrics", "outputs/results/baseline_metrics.csv"),
            ("Model comparison", "model_comparison_results.csv"),
            ("SSL embeddings", "outputs/results/transformer_semester_embeddings.csv"),
            ("Cluster summary", "outputs/results/transformer_student_cluster_summary.csv"),
        ]
        
        for name, path in output_files:
            full_path = self.base_dir / path
            status = "EXISTS" if full_path.exists() else "NOT FOUND"
            print(f"  {name:25s}: {path:50s} [{status}]")
        
        print(f"\n{'='*70}")
        print("Pipeline completed!")
        print('='*70)


def main():
    parser = argparse.ArgumentParser(
        description="Run full training pipeline for student behavior analysis"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Quick mode: use 10 epochs instead of 200"
    )
    parser.add_argument(
        "--skip-ssl",
        action="store_true", 
        help="Skip SSL training (use existing embeddings)"
    )
    
    args = parser.parse_args()
    
    runner = PipelineRunner(quick_mode=args.quick, skip_ssl=args.skip_ssl)
    runner.run_full_pipeline()


if __name__ == "__main__":
    main()
