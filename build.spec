# -*- mode: python ; coding: utf-8 -*-

import os
import sys

# Находим пути к данным
datas = []

# RapidOCR config
try:
    import rapidocr_onnxruntime
    rapidocr_dir = os.path.dirname(rapidocr_onnxruntime.__file__)
    config_path = os.path.join(rapidocr_dir, 'config.yaml')
    if os.path.exists(config_path):
        datas.append((config_path, 'rapidocr_onnxruntime'))
except ImportError:
    pass

# PyMuPDF данные
try:
    import pymupdf
    pymupdf_dir = os.path.dirname(pymupdf.__file__)
    if os.path.exists(pymupdf_dir):
        datas.append((pymupdf_dir, 'pymupdf'))
except ImportError:
    pass

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'cv2',
        'numpy',
        'PIL',
        'pymupdf',
        'fitz',
        'rapidocr_onnxruntime',
        'onnxruntime',
        'easyocr',
        'paddleocr',
        'paddle',
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
    name='DocumentOCR',
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
