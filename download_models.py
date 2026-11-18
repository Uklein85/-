"""
Скрипт скачивания OCR моделей через встроенный urllib (без внешних зависимостей)
Совместим с GitHub Actions (Windows runner) и локальной сборкой
"""
import os
import sys
import io
import time
import urllib.request
import ssl
from pathlib import Path

# === КРИТИЧНО: форсируем UTF-8 для stdout/stderr ===
# Это решает UnicodeEncodeError на Windows-консоли GitHub Actions
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Также устанавливаем переменную окружения для дочерних процессов
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'


def download_with_retry(url: str, dest: Path, max_retries: int = 3):
    """Скачивание с повторными попытками и обходом 404"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/octet-stream'
    }
    
    ctx = ssl.create_default_context()
    # Не отключаем проверку SSL — GitHub требует валидный сертификат
    # ctx.check_hostname = False
    # ctx.verify_mode = ssl.CERT_NONE
    
    for attempt in range(max_retries):
        try:
            print(f"[INFO] Attempt {attempt + 1}/{max_retries}: {dest.name}")
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=120, context=ctx) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                
                with open(dest, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            # Используем \r без эмодзи — безопасно для любой кодировки
                            print(f"\r   Progress: {percent:.1f}% ({downloaded // 1024} KB)", 
                                  end='', flush=True)
            
            print(f"\n[OK] Downloaded: {dest.name} ({downloaded / 1024 / 1024:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            if dest.exists():
                dest.unlink()
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"[INFO] Waiting {wait_time}s before retry...")
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
            print(f"[OK] {name} already exists ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
            success_count += 1
            continue
        
        if download_with_retry(url, dest):
            success_count += 1
        else:
            print(f"[FAIL] Could not download {name}")
    
    print(f"\n{'='*50}")
    if success_count == len(models):
        print(f"[SUCCESS] All models downloaded ({success_count}/{len(models)})")
        return 0
    else:
        print(f"[WARNING] Downloaded {success_count}/{len(models)} models")
        return 1


if __name__ == "__main__":
    sys.exit(download_models())
