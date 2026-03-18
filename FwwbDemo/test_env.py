"""
环境验证脚本 - 检查所有依赖是否正确安装
"""
import sys

def check_package(name, import_name=None):
    """检查包是否安装"""
    if import_name is None:
        import_name = name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  [OK] {name:20s} - {version}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {name:20s} - not installed")
        return False

print("="*60)
print("A14 Environment Check")
print("="*60)

print("\n[1] Python:")
print(f"  Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print(f"  Path: {sys.executable}")

print("\n[2] Data Science Packages:")
all_ok = True
all_ok &= check_package("numpy")
all_ok &= check_package("pandas")
all_ok &= check_package("scikit-learn", "sklearn")
all_ok &= check_package("matplotlib")
all_ok &= check_package("seaborn")

print("\n[3] Deep Learning:")
all_ok &= check_package("torch")
all_ok &= check_package("pytorch-lightning", "pytorch_lightning")
all_ok &= check_package("torchmetrics")

print("\n[4] Project Libs:")
all_ok &= check_package("ts3l")

print("\n[5] Utils:")
check_package("tqdm")
check_package("openpyxl")

print("\n" + "="*60)
if all_ok:
    print("[SUCCESS] All core packages installed!")
    print("="*60)
    print("\nReady to run:")
    print("  1. python run_baseline_simple.py")
    print("  2. python prepare_tabulars3l_inputs.py")
    print("  3. python train_tabulars3l_dae_cluster_transformer.py")
else:
    print("[WARNING] Some packages missing")
print("="*60)
