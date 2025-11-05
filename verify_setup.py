#!/usr/bin/env python3
"""
Verification script to check if everything is set up correctly for PPO training.
Run this before starting training to ensure all dependencies and files are in place.
"""

import os
import sys
from pathlib import Path


def print_section(title):
    """Print formatted section title."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check_mark(condition, message):
    """Print check mark or X based on condition."""
    status = "✓" if condition else "✗"
    color = "\033[92m" if condition else "\033[91m"
    end_color = "\033[0m"
    print(f"{color}[{status}]{end_color} {message}")
    return condition


def verify_setup():
    """Verify all setup requirements."""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "PPO Training Setup Verification" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    
    all_ok = True
    
    # ========================================================================
    # 1. Check Python Version
    # ========================================================================
    print_section("1. Python Environment")
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    ok = check_mark(
        sys.version_info >= (3, 10),
        f"Python version: {python_version} (required: >= 3.10)"
    )
    all_ok = all_ok and ok
    
    # ========================================================================
    # 2. Check Required Packages
    # ========================================================================
    print_section("2. Required Packages")
    
    packages = {
        "ray": "ray",
        "gymnasium": "gymnasium",
        "numpy": "numpy",
        "torch": "torch",
        "pettingzoo": "pettingzoo",
    }
    
    for package_name, import_name in packages.items():
        try:
            __import__(import_name)
            ok = check_mark(True, f"{package_name}: installed")
        except ImportError:
            ok = check_mark(False, f"{package_name}: NOT installed")
            all_ok = False
    
    # ========================================================================
    # 3. Check SUMO Setup
    # ========================================================================
    print_section("3. SUMO Configuration")
    
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        ok = check_mark(True, f"SUMO_HOME: {sumo_home}")
        sumo_path = Path(sumo_home) / "bin" / "sumo"
        ok2 = check_mark(sumo_path.exists(), f"SUMO binary: {sumo_path}")
        all_ok = all_ok and ok and ok2
    else:
        ok = check_mark(False, "SUMO_HOME: NOT set")
        all_ok = False
    
    # ========================================================================
    # 4. Check Project Structure
    # ========================================================================
    print_section("4. Project Structure")
    
    base_path = Path(__file__).parent
    
    required_dirs = {
        "scripts": "scripts directory",
        "src/environment/drl_algo": "environment module",
        "network": "network files",
    }
    
    for dir_path, description in required_dirs.items():
        full_path = base_path / dir_path
        ok = check_mark(full_path.exists(), f"{description}: {full_path}")
        all_ok = all_ok and ok
    
    # ========================================================================
    # 5. Check Required Files
    # ========================================================================
    print_section("5. Required Python Files")
    
    required_files = {
        "scripts/train_ppo.py": "PPO training script",
        "scripts/eval_ppo.py": "PPO evaluation script",
        "src/environment/drl_algo/env.py": "SUMO environment",
        "src/environment/drl_algo/traffic_signal.py": "Traffic signal class",
        "pyproject.toml": "Project configuration",
    }
    
    for file_path, description in required_files.items():
        full_path = base_path / file_path
        ok = check_mark(full_path.exists(), f"{description}: {file_path}")
        all_ok = all_ok and ok
    
    # ========================================================================
    # 6. Check Network Files
    # ========================================================================
    print_section("6. Network Files")
    
    networks = ["grid4x4", "4x4loop", "zurich", "PhuQuoc"]
    network_path = base_path / "network"
    
    for network in networks:
        net_file = network_path / network / f"{network}.net.xml"
        rou_file = network_path / network / f"{network}.rou.xml"
        ok1 = check_mark(net_file.exists(), f"{network}: network file (.net.xml)")
        ok2 = check_mark(rou_file.exists(), f"{network}: route file (.rou.xml)")
        all_ok = all_ok and ok1 and ok2
    
    # ========================================================================
    # 7. Check Output Directory
    # ========================================================================
    print_section("7. Output Directory")
    
    results_path = base_path / "results"
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
        check_mark(True, f"Results directory created: {results_path}")
    else:
        check_mark(True, f"Results directory exists: {results_path}")
    
    # ========================================================================
    # 8. Summary and Recommendations
    # ========================================================================
    print_section("8. Summary")
    
    if all_ok:
        print("\n✓ All checks passed! You're ready to start training.\n")
        print("Quick start commands:")
        print("  1. Quick test:   python scripts/train_ppo.py --iterations 10 --workers 1")
        print("  2. Normal train: python scripts/train_ppo.py --iterations 100 --workers 2")
        print("  3. Full train:   python scripts/train_ppo.py --iterations 500 --workers 4 --gpu")
        print("\nFor more options:")
        print("  python scripts/train_ppo.py --help")
        print("\nFor detailed guide:")
        print("  cat TRAINING_GUIDE.md")
        print("")
    else:
        print("\n✗ Some checks failed. Please fix the issues above before training.\n")
        print("Common fixes:")
        print("  1. SUMO_HOME not set:")
        print("     export SUMO_HOME=/usr/share/sumo")
        print("  2. Missing Python packages:")
        print("     pip install -e .")
        print("  3. Missing project files:")
        print("     Check if you're in the correct directory")
        print("")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(verify_setup())
