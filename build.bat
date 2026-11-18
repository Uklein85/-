@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ============================================================
echo   Сборка DocumentOCR для АО "СТНГ"
echo   Разработчик: Уклеин И.В. | г. Новосибирск | 2026
echo ============================================================
echo.

REM === ПРОВЕРКА PYTHON ===
python --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден в PATH!
    echo Установите Python 3.10+ с python.org и отметьте "Add to PATH"
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYVER=%%i
echo [1/6] Python найден: %PYVER%

REM === ОБНОВЛЕНИЕ PIP ===
echo [2/6] Обновление pip и setuptools...
python -m pip install --upgrade pip setuptools wheel --quiet

REM === УСТАНОВКА ЗАВИСИМОСТЕЙ ===
echo [3/6] Установка зависимостей из requirements.txt...
if not exist "requirements (2).txt" (
    echo [ОШИБКА] Файл requirements (2).txt не найден!
    pause
    exit /b 1
)
pip install -r "requirements (2).txt" --quiet
if errorlevel 1 (
    echo [ОШИБКА] Не удалось установить зависимости!
    pause
    exit /b 1
)
echo       ✓ Зависимости установлены

REM === УСТАНОВКА PYINSTALLER ===
pip install pyinstaller --quiet

REM === СКАЧИВАНИЕ OCR МОДЕЛЕЙ (обход ошибки 404) ===
echo [4/6] Скачивание OCR моделей RapidOCR...
if not exist "ocr_models\rapidocr" mkdir "ocr_models\rapidocr"

set MODELS_DIR=ocr_models\rapidocr
set DET_URL=https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_PP-OCRv4_det_infer.onnx
set REC_URL=https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_PP-OCRv4_rec_infer.onnx
set CLS_URL=https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_ppocr_mobile_v2.0_cls_infer.onnx

REM Скачивание через PowerShell с User-Agent (иначе GitHub отдаёт 404)
powershell -Command "& { $headers = @{'User-Agent'='Mozilla/5.0'}; if (-not (Test-Path '%MODELS_DIR%\ch_PP-OCRv4_det_infer.onnx') -or (Get-Item '%MODELS_DIR%\ch_PP-OCRv4_det_infer.onnx').Length -lt 1000000) { Write-Host '  ⬇ ch_PP-OCRv4_det_infer.onnx...'; Invoke-WebRequest -Uri '%DET_URL%' -OutFile '%MODELS_DIR%\ch_PP-OCRv4_det_infer.onnx' -Headers $headers } else { Write-Host '  ✓ det уже есть' } }"

powershell -Command "& { $headers = @{'User-Agent'='Mozilla/5.0'}; if (-not (Test-Path '%MODELS_DIR%\ch_PP-OCRv4_rec_infer.onnx') -or (Get-Item '%MODELS_DIR%\ch_PP-OCRv4_rec_infer.onnx').Length -lt 1000000) { Write-Host '  ⬇ ch_PP-OCRv4_rec_infer.onnx...'; Invoke-WebRequest -Uri '%REC_URL%' -OutFile '%MODELS_DIR%\ch_PP-OCRv4_rec_infer.onnx' -Headers $headers } else { Write-Host '  ✓ rec уже есть' } }"

powershell -Command "& { $headers = @{'User-Agent'='Mozilla/5.0'}; if (-not (Test-Path '%MODELS_DIR%\ch_ppocr_mobile_v2.0_cls_infer.onnx') -or (Get-Item '%MODELS_DIR%\ch_ppocr_mobile_v2.0_cls_infer.onnx').Length -lt 1000000) { Write-Host '  ⬇ ch_ppocr_mobile_v2.0_cls_infer.onnx...'; Invoke-WebRequest -Uri '%CLS_URL%' -OutFile '%MODELS_DIR%\ch_ppocr_mobile_v2.0_cls_infer.onnx' -Headers $headers } else { Write-Host '  ✓ cls уже есть' } }"

REM Проверка что модели скачались
if not exist "%MODELS_DIR%\ch_PP-OCRv4_det_infer.onnx" (
    echo [ОШИБКА] Не удалось скачать модели! Проверьте интернет-соединение.
    pause
    exit /b 1
)
echo       ✓ Модели OCR готовы

REM === ОЧИСТКА СТАРОЙ СБОРКИ ===
echo [5/6] Очистка предыдущих сборок...
if exist "build" rmdir /s /q "build"
if exist "dist\DocumentOCR.exe" del /f /q "dist\DocumentOCR.exe"

REM === СБОРКА .EXE ЧЕРЕЗ PYINSTALLER ===
echo [6/6] Сборка DocumentOCR.exe (это займёт 2-5 минут)...
echo.

pyinstaller --noconfirm --onefile --windowed ^
    --name "DocumentOCR" ^
    --add-data "ocr_models;ocr_models" ^
    --add-data "README.md;." ^
    --hidden-import "rapidocr_onnxruntime" ^
    --hidden-import "rapidocr_onnxruntime.ch_ppocr_mobile_v2_cls" ^
    --hidden-import "onnxruntime" ^
    --hidden-import "onnxruntime.capi" ^
    --hidden-import "onnxruntime.capi._pybind_state" ^
    --hidden-import "cv2" ^
    --hidden-import "cv2.cv2" ^
    --hidden-import "numpy" ^
    --hidden-import "numpy.core" ^
    --hidden-import "numpy.core.multiarray" ^
    --hidden-import "PIL" ^
    --hidden-import "PIL._imaging" ^
    --hidden-import "pymupdf" ^
    --hidden-import "fitz" ^
    --hidden-import "yaml" ^
    --hidden-import "openpyxl" ^
    --hidden-import "pandas" ^
    --collect-all PyQt6 ^
    --collect-submodules onnxruntime ^
    --collect-submodules cv2 ^
    main.py

if errorlevel 1 (
    echo.
    echo [ОШИБКА] Сборка не удалась! Проверьте вывод выше.
    pause
    exit /b 1
)

REM === ФИНАЛЬНАЯ ПРОВЕРКА ===
echo.
echo ============================================================
if exist "dist\DocumentOCR.exe" (
    echo   ✓ СБОРКА УСПЕШНА!
    echo   Файл: dist\DocumentOCR.exe
    for %%A in ("dist\DocumentOCR.exe") do echo   Размер: %%~zA байт
) else (
    echo   ✗ Файл DocumentOCR.exe не найден в папке dist
)
echo ============================================================
echo.
echo Готово! Запустите dist\DocumentOCR.exe для проверки.
echo Модели OCR встроены в .exe — интернет больше не нужен.
echo.
pause
