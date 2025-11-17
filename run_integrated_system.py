#!/usr/bin/env python3
"""
Integrated Malaria Detection System Launcher
Supports both Image-Based and Symptom-Based Detection
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'opencv-python', 'scikit-learn', 'pandas', 'numpy', 
        'joblib', 'matplotlib', 'seaborn', 'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-learn':
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
            elif package == 'Pillow':
                from PIL import Image
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

def check_datasets():
    """Check if required datasets exist"""
    datasets = {
        'Image dataset': 'dataset.csv',
        'Symptom dataset': 'mmc1.csv'
    }
    
    missing_datasets = []
    for name, file in datasets.items():
        if not os.path.exists(file):
            missing_datasets.append(f"{name} ({file})")
    
    if missing_datasets:
        print("Missing datasets:")
        for dataset in missing_datasets:
            print(f"  - {dataset}")
        return False
    
    return True

def check_models():
    """Check if trained models exist"""
    # Image-based models
    image_models = ['rf_malaria_100_5']
    
    # Symptom-based models
    symptom_models = [
        'malaria_symptom_decision_tree.joblib',
        'malaria_symptom_svm.joblib',
        'malaria_symptom_logistic_regression.joblib',
        'malaria_symptom_random_forest.joblib',
        'malaria_symptom_scaler.joblib',
        'malaria_symptom_features.joblib'
    ]
    
    missing_models = []
    
    # Check image models
    for model in image_models:
        if not os.path.exists(model):
            missing_models.append(f"Image model: {model}")
    
    # Check symptom models
    for model in symptom_models:
        if not os.path.exists(model):
            missing_models.append(f"Symptom model: {model}")
    
    if missing_models:
        print("Missing trained models:")
        for model in missing_models:
            print(f"  - {model}")
        return False
    
    return True

def train_missing_models():
    """Train missing models"""
    print("\nTraining missing models...")
    
    # Train image-based model if needed
    if not os.path.exists('rf_malaria_100_5'):
        print("Training image-based model...")
        try:
            subprocess.run([sys.executable, 'malaria_classification.py'], check=True)
            print("Image-based model trained successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to train image-based model: {e}")
            return False
    
    # Train symptom-based models if needed
    missing_symptom_models = [
        'malaria_symptom_decision_tree.joblib',
        'malaria_symptom_svm.joblib',
        'malaria_symptom_logistic_regression.joblib',
        'malaria_symptom_random_forest.joblib',
        'malaria_symptom_scaler.joblib',
        'malaria_symptom_features.joblib'
    ]
    
    if any(not os.path.exists(model) for model in missing_symptom_models):
        print("Training symptom-based models...")
        try:
            subprocess.run([sys.executable, 'malaria_symptom_classification.py'], check=True)
            print("Symptom-based models trained successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to train symptom-based models: {e}")
            return False
    
    return True

def show_menu():
    """Show application menu"""
    print("\n" + "="*60)
    print("MALARIA DETECTION SYSTEM - MAIN MENU")
    print("="*60)
    print("1. 🖼️  Image-Based Detection (Blood Cell Analysis)")
    print("2. 🏥  Symptom-Based Detection (Clinical Symptoms)")
    print("3. 🔄  Integrated Detection (Combined Analysis)")
    print("4. 📊  Train Models")
    print("5. ❌  Exit")
    print("="*60)

def run_application(choice):
    """Run the selected application"""
    applications = {
        '1': {
            'name': 'Image-Based Detection',
            'script': 'malaria_gui.py',
            'description': 'Analyze blood cell images for malaria parasites'
        },
        '2': {
            'name': 'Symptom-Based Detection',
            'script': 'malaria_symptom_gui.py',
            'description': 'Analyze patient symptoms for malaria risk'
        },
        '3': {
            'name': 'Integrated Detection',
            'script': 'integrated_malaria_detection.py',
            'description': 'Combined image and symptom analysis'
        },
        '4': {
            'name': 'Train Models',
            'script': 'train_all_models.py',
            'description': 'Train all machine learning models'
        }
    }
    
    if choice in applications:
        app_info = applications[choice]
        print(f"\nLaunching {app_info['name']}...")
        print(f"Description: {app_info['description']}")
        
        try:
            subprocess.run([sys.executable, app_info['script']], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {app_info['name']}: {e}")
        except FileNotFoundError:
            print(f"Script not found: {app_info['script']}")
    else:
        print("Invalid choice. Please select 1-5.")

def train_all_models():
    """Train all models"""
    print("\nTraining all models...")
    
    scripts_to_run = [
        'malaria_classification.py',
        'malaria_symptom_classification.py'
    ]
    
    for script in scripts_to_run:
        if os.path.exists(script):
            print(f"Running {script}...")
            try:
                subprocess.run([sys.executable, script], check=True)
                print(f"{script} completed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Error running {script}: {e}")
        else:
            print(f"Script not found: {script}")
    
    print("\nModel training completed!")

def main():
    """Main launcher function"""
    print("Integrated Malaria Detection System")
    print("Combining Image-Based and Symptom-Based Analysis")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    # Check datasets
    if not check_datasets():
        print("\nPlease ensure all required datasets are available.")
        sys.exit(1)
    
    # Check and train models if needed
    if not check_models():
        print("\nSome models are missing. Training them now...")
        if not train_missing_models():
            print("Failed to train models. Please check your datasets and try again.")
            sys.exit(1)
    
    print("\n✅ All systems ready!")
    
    # Main application loop
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '5':
                print("\nThank you for using the Malaria Detection System!")
                break
            elif choice == '4':
                train_all_models()
            else:
                run_application(choice)
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
