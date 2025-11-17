#!/usr/bin/env python3
"""
Build run_symptom_app.py as executable with proper configuration
"""

import subprocess
import sys
import os
import shutil

def check_requirements():
    """Check if required files exist"""
    required_files = [
        'malaria_symptom_decision_tree.joblib',
        'malaria_symptom_svm.joblib',
        'malaria_symptom_logistic_regression.joblib',
        'malaria_symptom_random_forest.joblib',
        'malaria_symptom_scaler.joblib',
        'malaria_symptom_features.joblib',
        'mmc1.csv',
        'run_symptom_app.py',
        'malaria_symptom_gui.py'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print("[ERROR] Missing required files:")
        for file in missing:
            print(f"  - {file}")
        return False
    
    print("[OK] All required files present")
    return True

def clean_build():
    """Clean previous build artifacts"""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name}/...")
            shutil.rmtree(dir_name)
    
    print("[OK] Build directories cleaned")

def build_symptom_app():
    """Build the symptom app as executable"""
    print("\nBuilding MalariaSymptomApp.exe...")
    print("=" * 50)
    
    try:
        # Build using spec file
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            "MalariaSymptomApp.spec"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print("\nBuilding... (this may take a few minutes)")
        
        # Run the build
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("\nBuild Output:")
        print(result.stdout)
        
        if result.stderr:
            print("\nWarnings/Errors:")
            print(result.stderr)
        
        # Check if the executable was created
        exe_path = "dist/MalariaSymptomApp.exe"
        if os.path.exists(exe_path):
            file_size = os.path.getsize(exe_path)
            print(f"\n[SUCCESS] Executable created: {exe_path}")
            print(f"          Size: {file_size / (1024*1024):.2f} MB")
            return True
        else:
            print("\n[ERROR] Executable not found after build")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Build error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        return False

def main():
    """Main build process"""
    print("Malaria Symptom App - Executable Builder")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n[ERROR] Build aborted: Missing required files")
        sys.exit(1)
    
    # Clean previous builds
    clean_build()
    
    # Build
    success = build_symptom_app()
    
    if success:
        print("\n" + "=" * 50)
        print("[SUCCESS] BUILD COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nExecutable location: dist/MalariaSymptomApp.exe")
        print("\nNOTE: Console window enabled for debugging.")
        print("      If app works properly, edit MalariaSymptomApp.spec")
        print("      and change 'console=True' to 'console=False'")
        print("      then rebuild.")
    else:
        print("\n" + "=" * 50)
        print("[FAILED] BUILD FAILED!")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    main()
