#!/usr/bin/env python3
"""
Malaria Symptom-Based Detection GUI Application Launcher
"""

import sys
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'sklearn', 'pandas', 'numpy', 'joblib', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'pandas':
                import pandas
            elif package == 'numpy':
                import numpy
            elif package == 'joblib':
                import joblib
            elif package == 'matplotlib':
                import matplotlib
            elif package == 'seaborn':
                import seaborn
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall them using:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_dataset():
    """Check if the symptom dataset exists"""
    if not os.path.exists("mmc1.csv"):
        print("Dataset file 'mmc1.csv' not found.")
        print("Please ensure the malaria symptom dataset is available.")
        return False
    return True

def check_models():
    """Check if trained models exist"""
    model_files = [
        'malaria_symptom_decision_tree.joblib',
        'malaria_symptom_svm.joblib', 
        'malaria_symptom_logistic_regression.joblib',
        'malaria_symptom_random_forest.joblib',
        'malaria_symptom_scaler.joblib',
        'malaria_symptom_features.joblib'
    ]
    
    missing_files = []
    for file in model_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Trained model files not found:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run 'python malaria_symptom_classification.py' first to train the models.")
        return False
    return True

def train_models_if_needed():
    """Train models if they don't exist"""
    print("Training models...")
    try:
        from malaria_symptom_classification import main as train_main
        train_main()
        print("Models trained successfully!")
        return True
    except Exception as e:
        print(f"Failed to train models: {e}")
        return False

def main():
    """Main launcher function"""
    print("Malaria Symptom-Based Detection System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check dataset
    if not check_dataset():
        sys.exit(1)
    
    # Check models and train if needed
    if not check_models():
        print("\nTraining models...")
        if not train_models_if_needed():
            sys.exit(1)
    
    print("All checks passed. Launching symptom-based application...")
    
    # Import and run the GUI
    try:
        from malaria_symptom_gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
