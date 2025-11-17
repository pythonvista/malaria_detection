# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['malaria_standalone_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[('malaria_symptom_*.joblib', '.'), ('mmc1.csv', '.')],
    hiddenimports=['scipy', 'scipy._lib', 'scipy._lib.array_api_compat', 'scipy._lib.array_api_compat.fft', 'scipy._lib._array_api_compat', 'scipy._lib._array_api_compat.fft', 'scipy.linalg', 'scipy.sparse', 'scipy.sparse.linalg', 'scipy.sparse.csgraph', 'scipy.sparse.csgraph._validation', 'sklearn', 'sklearn.ensemble', 'sklearn.tree', 'sklearn.svm', 'sklearn.linear_model', 'sklearn.model_selection', 'sklearn.preprocessing', 'sklearn.metrics', 'sklearn.utils', 'sklearn.utils._param_validation', 'sklearn.utils._tags', 'sklearn.utils.validation', 'joblib', 'cv2', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'PIL', 'tkinter', 'tkinter.ttk', 'tkinter.messagebox', 'tkinter.filedialog', 'threading', 'multiprocessing', 'concurrent', 'concurrent.futures'],
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
    name='MalariaDetectionSystem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
