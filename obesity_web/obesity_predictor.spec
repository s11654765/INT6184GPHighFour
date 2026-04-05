# -*- mode: python ; coding: utf-8 -*-
# 在 obesity_web 目录执行: py -m PyInstaller obesity_predictor.spec
from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = [
    ("static", "static"),
    ("xgb_model_bundle.joblib", "."),
]
binaries = []
hiddenimports = []

for pkg in ("xgboost", "sklearn"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports
    + [
        "feature_mappings",
        "model_final_encode",
        "student_advice_en",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._quad_tree",
        "sklearn.tree._utils",
    ],
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
    name="ObesityPredictor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
