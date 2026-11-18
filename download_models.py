"""
Скрипт скачивания OCR моделей с retry logic для CI/CD
"""
import os
import sys
import time
import requests
from pathlib import Path

def download_with_retry(url: str, dest: Path, max_retries: int = 3):
    """Скачивание с повторными попытками"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'application/octet-stream'
    }
    
    for attempt in range(max_retries):
        try:
            print(f"⬇️ Попытка {attempt + 1}/{max_retries}: {dest.name}")
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Прогресс: {percent:.1f}%", end='', flush=True)
            
            print(f"\n✅ Скачано: {dest.name} ({downloaded / 1024 / 1024:.1f} МБ)")
            return True
            
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            if dest.exists():
                dest.unlink()
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⏳ Ожидание {wait_time} сек перед повтором...")
                time.sleep(wait_time)
    
    return False

def download_models():
    models_dir = Path(__file__).parent / 'ocr_models' / 'rapidocr'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    models = {
        'ch_PP-OCRv4_det_infer.onnx': 'https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_PP-OCRv4_det_infer.onnx',
        'ch_PP-OCRv4_rec_infer.onnx': 'https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_PP-OCRv4_rec_infer.onnx',
        'ch_ppocr_mobile_v2.0_cls_infer.onnx': 'https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_ppocr_mobile_v2.0_cls_infer.onnx',
    }
    
    success_count = 0
    for name, url in models.items():
        dest = models_dir / name
        if dest.exists() and dest.stat().st_size > 1000000:
            print(f"✓ {name} уже существует")
            success_count += 1
            continue
        
        if download_with_retry(url, dest):
            success_count += 1
        else:
            print(f"✗ Не удалось скачать {name}")
    
    if success_count == len(models):
        print("\n🎉 Все модели скачаны успешно!")
        return 0
    else:
        print(f"\n⚠️ Скачано {success_count}/{len(models)} моделей")
        return 1

if __name__ == "__main__":
    sys.exit(download_models())
