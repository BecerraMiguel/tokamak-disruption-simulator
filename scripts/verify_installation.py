#!/usr/bin/env python3
"""
Installation verification script for the Tokamak Disruption Simulator.

This script verifies that all required dependencies are installed and
that the project structure is correct.

Usage:
    python scripts/verify_installation.py
"""

import sys
from pathlib import Path


def check_python_version():
    """Verify Python version is 3.10 or higher."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"  [FAIL] Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    print(f"  [OK] Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_core_dependencies():
    """Verify core Python dependencies are installed."""
    print("\nChecking core dependencies...")
    
    dependencies = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("h5py", "h5py"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("sklearn", "scikit-learn"),
        ("tqdm", "tqdm"),
        ("yaml", "PyYAML"),
    ]
    
    all_ok = True
    for module_name, package_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  [OK] {package_name} ({version})")
        except ImportError:
            print(f"  [FAIL] {package_name} not installed")
            all_ok = False
    
    return all_ok


def check_dream_installation():
    """Check if DREAM is installed (optional)."""
    print("\nChecking DREAM installation (optional)...")
    
    try:
        import DREAM
        print("  [OK] DREAM Python interface available")
        return True
    except ImportError:
        print("  [SKIP] DREAM not installed (optional for phenomenological mode)")
        return None


def check_project_structure():
    """Verify project directory structure."""
    print("\nChecking project structure...")
    
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        "configs",
        "configs/dina",
        "configs/dream",
        "src",
        "src/dina",
        "src/dream",
        "src/pipeline",
        "src/utils",
        "scripts",
        "notebooks",
        "docs",
        "data",
        "results",
        "tests",
    ]
    
    required_files = [
        "README.md",
        "LICENSE",
        "requirements.txt",
        "environment.yml",
        ".gitignore",
        "src/__init__.py",
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  [OK] {dir_path}/")
        else:
            print(f"  [FAIL] {dir_path}/ missing")
            all_ok = False
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [FAIL] {file_path} missing")
            all_ok = False
    
    return all_ok


def check_src_imports():
    """Verify source modules can be imported."""
    print("\nChecking source module imports...")
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    modules = [
        "src",
        "src.dina",
        "src.dream",
        "src.pipeline",
        "src.utils",
    ]
    
    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  [OK] {module_name}")
        except ImportError as e:
            print(f"  [FAIL] {module_name}: {e}")
            all_ok = False
    
    return all_ok


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Tokamak Disruption Simulator - Installation Verification")
    print("=" * 60)
    
    results = {
        "Python version": check_python_version(),
        "Core dependencies": check_core_dependencies(),
        "DREAM (optional)": check_dream_installation(),
        "Project structure": check_project_structure(),
        "Source imports": check_src_imports(),
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in results.items():
        if result is True:
            status = "PASS"
        elif result is False:
            status = "FAIL"
            all_passed = False
        else:
            status = "SKIP"
        print(f"  {check_name}: {status}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("Installation verification PASSED")
        print("The basic installation is complete.")
        print("See docs/INSTALLATION.md for DREAM and DINA setup.")
        return 0
    else:
        print("Installation verification FAILED")
        print("Please review the errors above and fix missing components.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
