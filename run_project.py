"""
Project Runner - Complete Anomaly Detection System
==================================================
This script runs the complete project with all components
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_environment():
    """Check if environment is properly set up"""
    print("Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check required files
    required_files = [
        "X_train.npy", "X_test.npy", "y_test.npy",
        "reconstruction_errors.npy", "threshold.npy"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        return False
    print("✓ All data files present")
    
    # Check key modules
    try:
        import numpy, pandas, sklearn, tensorflow, flask
        print("✓ All packages available")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def run_flask_app():
    """Run the Flask web application"""
    print("\n" + "="*50)
    print("STARTING ANOMALY DETECTION SYSTEM")
    print("="*50)
    
    print("\nStarting Flask server...")
    print("🌐 Web Interface will be available at:")
    print("   • Original Dashboard: http://localhost:5000/")
    print("   • Enhanced Dashboard: http://localhost:5000/enhanced")
    print("   • About Page: http://localhost:5000/about")
    
    print("\n📊 Available Features:")
    print("   • Real-time anomaly detection")
    print("   • XAI explanations")
    print("   • GenAI intelligence analysis")
    print("   • MITRE ATT&CK mapping")
    print("   • Interactive charts")
    print("   • Threat reports")
    
    print("\n🔧 API Endpoints:")
    print("   • /api/predict - Single sample analysis")
    print("   • /api/genai-analyze - GenAI analysis")
    print("   • /api/threat-report - Threat reports")
    print("   • /api/statistics - Performance metrics")
    
    print("\n" + "="*50)
    print("Press Ctrl+C to stop the server")
    print("="*50)
    
    try:
        # Import and run Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

def main():
    """Main function"""
    print("🚀 LSTM Anomaly Detection System")
    print("=" * 50)
    
    if not check_environment():
        print("\n❌ Environment check failed!")
        print("Run: python setup_environment.py")
        return
    
    print("\n✅ Environment ready!")
    
    # Ask user what to run
    print("\nWhat would you like to run?")
    print("1. Web Application (Flask)")
    print("2. Main Pipeline Analysis")
    print("3. Setup Environment")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        run_flask_app()
    elif choice == "2":
        print("\nRunning main pipeline...")
        try:
            from main_pipeline import main
            main()
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
    elif choice == "3":
        print("\nRunning environment setup...")
        try:
            from setup_environment import setup_environment
            setup_environment()
        except Exception as e:
            print(f"❌ Setup error: {e}")
    else:
        print("Invalid choice. Running web application...")
        run_flask_app()

if __name__ == "__main__":
    main()