# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

# Collect all model and data files
added_files = [
    ('malaria_symptom_decision_tree.joblib', '.'),
    ('malaria_symptom_svm.joblib', '.'),
    ('malaria_symptom_logistic_regression.joblib', '.'),
    ('malaria_symptom_random_forest.joblib', '.'),
    ('malaria_symptom_scaler.joblib', '.'),
    ('malaria_symptom_features.joblib', '.'),
    ('mmc1.csv', '.'),
]

# Hidden imports for all dependencies
hidden_imports = [
    'sklearn',
    'sklearn.tree',
    'sklearn.svm',
    'sklearn.linear_model',
    'sklearn.ensemble',
    'sklearn.preprocessing',
    'pandas',
    'numpy',
    'joblib',
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
]

a = Analysis(
    ['run_symptom_app.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MalariaSymptomApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Changed to True to see errors
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
