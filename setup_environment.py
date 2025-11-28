"""
Complete Environment Setup for LSTM Anomaly Detection System
============================================================
This script sets up the complete environment with all dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"[OK] {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"[ERROR] Failed to install {package}")
        return False

def setup_environment():
    """Setup complete environment"""
    print("=" * 60)
    print("SETTING UP ANOMALY DETECTION ENVIRONMENT")
    print("=" * 60)
    
    # Required packages
    packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "flask>=2.0.0",
        "waitress>=2.1.0",
        "openai>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]
    
    print("\nInstalling required packages...")
    failed_packages = []
    
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n[WARNING] Failed to install: {', '.join(failed_packages)}")
        print("Please install manually using: pip install <package_name>")
    else:
        print("\n[OK] All packages installed successfully!")
    
    # Create necessary directories
    print("\nCreating project directories...")
    directories = [
        "templates",
        "static",
        "logs",
        "exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"[OK] Created directory: {directory}")
    
    # Check for required data files
    print("\nChecking required data files...")
    required_files = [
        "X_train.npy",
        "X_test.npy", 
        "y_test.npy",
        "reconstruction_errors.npy",
        "threshold.npy",
        "lstm_autoencoder_final.h5"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] Found: {file}")
        else:
            print(f"[MISSING] {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n[WARNING] Missing data files: {', '.join(missing_files)}")
        print("Run the preprocessing and training scripts first!")
    
    print("\n" + "=" * 60)
    print("ENVIRONMENT SETUP COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    setup_environment()