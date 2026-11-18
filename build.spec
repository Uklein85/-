# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import glob

block_cipher = None

# ============================================================
# СБОР ДАННЫХ ДЛЯ ВКЛЮЧЕНИЯ В .EXE
# ============================================================

datas = []
binaries = []

# 1. Модели RapidOCR - ГЛАВНОЕ ИСПРАВЛЕНИЕ
try:
    import rapidocr_onnxruntime
    rapidocr_dir = os.path.dirname(rapidocr_onnxruntime.__file__)
    
    # Включаем config.yaml
    config_path = os.path.join(rapidocr_dir, 'config.yaml')
    if os.path.exists(config_path):
        datas.append((config_path, 'rapidocr_onnxruntime'))
        print(f"✅ Добавлен config.yaml")
    
    # Включаем ВСЕ модели ONNX (рекурсивно)
    models_dir = os.path.join(rapidocr_dir, 'models')
    if os.path.exists(models_dir):
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith('.onnx') or file.endswith('.yaml'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, rapidocr_dir)
                    dest_dir = os.path.join('rapidocr_onnxruntime', os.path.dirname(rel_path))
                    datas.append((full_path, dest_dir))
        print(f"✅ Добавлены модели ONNX из {models_dir}")
    
    # Включаем все .py файлы пакета
    for root, dirs, files in os.walk(rapidocr_dir):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, rapidocr_dir)
                dest_dir = os.path.join('rapidocr_onnxruntime', os.path.dirname(rel_path))
                datas.append((full_path, dest_dir))
    
except ImportError:
    print("⚠️ rapidocr_onnxruntime не найден")
except Exception as e:
    print(f"⚠️ Ошибка сбора RapidOCR: {e}")

# 2. PyMuPDF данные
try:
    import pymupdf
    pymupdf_dir = os.path.dirname(pymupdf.__file__)
    # Включаем ресурсы PyMuPDF
    for item in os.listdir(pymupdf_dir):
        full_path = os.path.join(pymupdf_dir, item)
        if os.path.isfile(full_path) and not item.endswith('.py'):
            datas.append((full_path, 'pymupdf'))
    print(f"✅ Добавлены данные PyMuPDF")
except:
    pass

# 3. OpenCV DLL (важно для Windows)
try:
    import cv2
    cv2_dir = os.path.dirname(cv2.__file__)
    for item in os.listdir(cv2_dir):
        if item.endswith('.dll') or item.endswith('.pyd'):
            full_path = os.path.join(cv2_dir, item)
            binaries.append((full_path, '.'))
    print(f"✅ Добавлены DLL OpenCV")
except:
    pass

print(f"\n📦 Всего данных: {len(datas)}")
print(f"📦 Всего бинарников: {len(binaries)}")

# ============================================================
# АНАЛИЗ
# ============================================================

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'cv2',
        'numpy',
        'PIL',
        'PIL.Image',
        'pymupdf',
        'fitz',
        'rapidocr_onnxruntime',
        'rapidocr_onnxruntime.utils',
        'onnxruntime',
        'onnxruntime.capi',
        'easyocr',
        'paddleocr',
        'paddle',
        'torchvision',
        'torchvision.ops',
        'yaml',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'torch',  # Исключаем torch чтобы уменьшить размер (EasyOCR опционален)
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ============================================================
# СБОРКА
# ============================================================

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
    console=True,  # ВАЖНО: True чтобы видеть ошибки
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
