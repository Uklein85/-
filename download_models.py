"""
Скрипт для предварительного скачивания моделей OCR перед сборкой в .exe
Запустить один раз: python download_models.py
"""
import os
import requests
from pathlib import Path

def download_models():
    models_dir = Path(__file__).parent / 'ocr_models' / 'rapidocr'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Рабочие URL моделей RapidOCR (ch_PP-OCRv4)
    models = {
        'ch_PP-OCRv4_det_infer.onnx': 'https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_PP-OCRv4_det_infer.onnx',
        'ch_PP-OCRv4_rec_infer.onnx': 'https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_PP-OCRv4_rec_infer.onnx',
        'ch_ppocr_mobile_v2.0_cls_infer.onnx': 'https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_ppocr_mobile_v2.0_cls_infer.onnx',
    }
    
    # GitHub требует User-Agent, иначе отдает 404
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for name, url in models.items():
        dest = models_dir / name
        if dest.exists() and dest.stat().st_size > 1000000: # > 1MB
            print(f"✅ {name} уже существует")
            continue
            
        print(f"⬇️ Скачивание {name}...")
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Успешно скачано: {name}")
        except Exception as e:
            print(f"❌ Ошибка скачивания {name}: {e}")
            if dest.exists(): dest.unlink()

if __name__ == "__main__":
    download_models()
    input("\nМодели скачаны. Теперь можно собирать .exe через build.bat")
