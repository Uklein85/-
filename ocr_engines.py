"""
OCR Engines Module - Исправленная версия для .exe сборки
Разработчик: Уклеин И.В. | АО "СТНГ" | 2026
"""
import os
import sys
import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Настройка логирования (ASCII только - для совместимости с Windows cp1252)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================
# УТИЛИТА: ПОЛУЧЕНИЕ ПУТИ К МОДЕЛЯМ
# ============================================================
def get_models_dir() -> Path:
    """
    Получить папку для хранения моделей.
    Приоритет: _MEIPASS (PyInstaller) -> рядом с exe -> папка проекта
    """
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            bundled = Path(sys._MEIPASS) / 'ocr_models'
            if bundled.exists():
                return bundled
        return Path(sys.executable).parent / 'ocr_models'
    else:
        return Path(__file__).parent / 'ocr_models'


def check_models_available() -> Dict[str, bool]:
    """Проверить наличие всех необходимых моделей."""
    models_dir = get_models_dir() / 'rapidocr'
    required = [
        'ch_PP-OCRv4_det_infer.onnx',
        'ch_PP-OCRv4_rec_infer.onnx',
        'ch_ppocr_mobile_v2.0_cls_infer.onnx'
    ]
    result = {}
    for name in required:
        path = models_dir / name
        result[name] = path.exists() and path.stat().st_size > 1000000
    return result


# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================
@dataclass
class TextBlock:
    text: str
    confidence: float
    bbox: List[List[float]] = field(default_factory=list)
    font_size: float = 0.0
    is_bold: bool = False
    language: str = ''


@dataclass
class ImageRegion:
    bbox: List[List[float]]
    image: np.ndarray
    confidence: float
    region_type: str


@dataclass
class OCRResult:
    text: str
    blocks: List[TextBlock]
    images: List[ImageRegion]
    confidence: float
    engine: str
    language: str
    metadata: Dict = field(default_factory=dict)


# ============================================================
# БАЗОВЫЙ КЛАСС
# ============================================================
class OCREngine(ABC):
    def __init__(self, languages: List[str] = None):
        self.languages = languages or ['ru', 'en']
        self.name = "BaseEngine"
        self._is_available = False

    @property
    def is_available(self) -> bool:
        return self._is_available

    @abstractmethod
    def recognize(self, image: np.ndarray) -> OCRResult:
        pass

    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        return self.recognize(image)


# ============================================================
# ДВИЖОК 1: RAPIDOCR (ОСНОВНОЙ)
# ============================================================
class RapidOCREngine(OCREngine):
    def __init__(self, languages: List[str] = None):
        super().__init__(languages)
        self.name = "RapidOCR"
        self._engine = None
        try:
            models_dir = get_models_dir() / 'rapidocr'
            models_status = check_models_available()
            missing = [name for name, exists in models_status.items() if not exists]
            
            if missing:
                raise FileNotFoundError(
                    f"Отсутствуют OCR модели: {', '.join(missing)}\n"
                    f"Ожидаемая папка: {models_dir}\n"
                    f"Запустите download_models.py перед сборкой .exe"
                )
            
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR(
                det_model_path=str(models_dir / 'ch_PP-OCRv4_det_infer.onnx'),
                rec_model_path=str(models_dir / 'ch_PP-OCRv4_rec_infer.onnx'),
                cls_model_path=str(models_dir / 'ch_ppocr_mobile_v2.0_cls_infer.onnx'),
            )
            self._is_available = True
            logger.info(f"[OK] RapidOCR инициализирован (модели: {models_dir})")
        except ImportError as e:
            logger.warning(f"[WARN] RapidOCR не установлен: {e}")
            self._is_available = False
        except FileNotFoundError as e:
            logger.error(f"[ERROR] {e}")
            self._is_available = False
        except Exception as e:
            logger.error(f"[ERROR] Ошибка инициализации RapidOCR: {type(e).__name__}: {e}")
            self._is_available = False

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available or self._engine is None:
            raise RuntimeError("RapidOCR недоступен")
        result, _ = self._engine(image)
        if not result:
            return OCRResult(text='', blocks=[], images=[], confidence=0.0, engine=self.name, language='+'.join(self.languages))
        
        blocks, texts, confidences = [], [], []
        for detection in result:
            bbox, text, confidence = detection
            blocks.append(TextBlock(text=text, confidence=confidence, bbox=bbox, language=self._detect_language(text)))
            texts.append(text)
            confidences.append(confidence)
        return OCRResult(text='\n'.join(texts), blocks=blocks, images=[], confidence=float(np.mean(confidences)) if confidences else 0.0, engine=self.name, language='+'.join(self.languages))

    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        result = self.recognize(image)
        for block in result.blocks:
            if block.bbox and len(block.bbox) >= 2:
                height = abs(block.bbox[2][1] - block.bbox[0][1])
                block.font_size = height
        return result

    def _detect_language(self, text: str) -> str:
        cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        latin = sum(1 for c in text if '\u0041' <= c <= '\u005A' or '\u0061' <= c <= '\u007A')
        if cyrillic > latin: return 'ru'
        elif latin > cyrillic: return 'en'
        return 'unknown'


# ============================================================
# ДВИЖОК 2: EASYOCR (ОПЦИОНАЛЬНЫЙ)
# ============================================================
class EasyOCREngine(OCREngine):
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        super().__init__(languages)
        self.name = "EasyOCR"
        self._reader = None
        try:
            import easyocr
            lang_map = {'ru': 'ru', 'en': 'en', 'rus': 'ru', 'eng': 'en'}
            langs = [lang_map.get(l, l) for l in self.languages]
            self._reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
            self._is_available = True
            logger.info("[OK] EasyOCR инициализирован")
        except ImportError as e:
            logger.warning(f"[WARN] EasyOCR не установлен: {e}")
            self._is_available = False
        except Exception as e:
            logger.error(f"[ERROR] Ошибка EasyOCR: {type(e).__name__}: {e}")
            self._is_available = False

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available or self._reader is None:
            raise RuntimeError("EasyOCR недоступен")
        results = self._reader.readtext(image)
        blocks, texts, confidences = [], [], []
        for detection in results:
            bbox, text, confidence = detection
            blocks.append(TextBlock(text=text, confidence=confidence, bbox=bbox, language='ru' if any('\u0400' <= c <= '\u04FF' for c in text) else 'en'))
            texts.append(text)
            confidences.append(confidence)
        return OCRResult(text='\n'.join(texts), blocks=blocks, images=[], confidence=float(np.mean(confidences)) if confidences else 0.0, engine=self.name, language='+'.join(self.languages))


# ============================================================
# ДВИЖОК 3: PADDLEOCR (ОПЦИОНАЛЬНЫЙ)
# ============================================================
class PaddleOCREngine(OCREngine):
    def __init__(self, languages: List[str] = None, use_gpu: bool = False):
        super().__init__(languages)
        self.name = "PaddleOCR"
        self._ocr = None
        try:
            from paddleocr import PaddleOCR
            lang_map = {'ru': 'ru', 'en': 'en', 'rus': 'ru', 'eng': 'en'}
            primary_lang = lang_map.get(self.languages[0], 'ru')
            try:
                self._ocr = PaddleOCR(use_angle_cls=True, lang=primary_lang, use_gpu=use_gpu)
            except TypeError:
                self._ocr = PaddleOCR(use_angle_cls=True, lang=primary_lang)
            self._is_available = True
            logger.info(f"[OK] PaddleOCR инициализирован (язык: {primary_lang})")
        except ImportError as e:
            logger.warning(f"[WARN] PaddleOCR не установлен: {e}")
            self._is_available = False
        except Exception as e:
            logger.error(f"[ERROR] Ошибка PaddleOCR: {type(e).__name__}: {e}")
            self._is_available = False

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available or self._ocr is None:
            raise RuntimeError("PaddleOCR недоступен")
        result = self._ocr.ocr(image, cls=True)
        if not result or not result[0]:
            return OCRResult(text='', blocks=[], images=[], confidence=0.0, engine=self.name, language='+'.join(self.languages))
        blocks, texts, confidences = [], [], []
        for line in result[0]:
            bbox, (text, confidence) = line
            blocks.append(TextBlock(text=text, confidence=confidence, bbox=bbox, language='ru' if any('\u0400' <= c <= '\u04FF' for c in text) else 'en'))
            texts.append(text)
            confidences.append(confidence)
        return OCRResult(text='\n'.join(texts), blocks=blocks, images=[], confidence=float(np.mean(confidences)) if confidences else 0.0, engine=self.name, language='+'.join(self.languages))


# ============================================================
# МЕНЕДЖЕР ДВИЖКОВ
# ============================================================
class OCREngineManager:
    PRIORITY = ['RapidOCR', 'PaddleOCR', 'EasyOCR']

    def __init__(self, preferred_engine: str = None, languages: List[str] = None):
        self.languages = languages or ['ru', 'en']
        # ЯВНАЯ ИНИЦИАЛИЗАЦИЯ СЛОВАРЯ (решает unhashable type: 'list')
        self.engines: Dict[str, OCREngine] = {}
        self.current_engine: Optional[OCREngine] = None
        self._init_all_engines(preferred_engine)

    def _init_all_engines(self, preferred: str = None):
        logger.info("Инициализация OCR движков...")
        
        engine_factories = [
            ('RapidOCR', lambda: RapidOCREngine(self.languages)),
            ('EasyOCR', lambda: EasyOCREngine(self.languages)),
            ('PaddleOCR', lambda: PaddleOCREngine(self.languages)),
        ]
        
        for name, factory in engine_factories:
            try:
                engine = factory()
                self.engines[name] = engine
                status = "OK" if engine.is_available else "UNAVAILABLE"
                logger.info(f"  {name}: {status}")
            except Exception as e:
                logger.warning(f"Ошибка инициализации {name}: {e}")
        
        # Выбор движка
        if preferred and preferred in self.engines and self.engines[preferred].is_available:
            self.current_engine = self.engines[preferred]
        else:
            for engine_name in self.PRIORITY:
                if engine_name in self.engines and self.engines[engine_name].is_available:
                    self.current_engine = self.engines[engine_name]
                    break
        
        if self.current_engine:
            logger.info(f"[OK] Выбран движок: {self.current_engine.name}")
        else:
            logger.error("[ERROR] Ни один OCR движок не доступен!")
            logger.error("Решение: pip install rapidocr-onnxruntime")

    def get_available_engines(self) -> List[str]:
        return [name for name, engine in self.engines.items() if engine.is_available]

    def switch_engine(self, engine_name: str) -> bool:
        if engine_name in self.engines and self.engines[engine_name].is_available:
            self.current_engine = self.engines[engine_name]
            logger.info(f"Переключено на: {engine_name}")
            return True
        return False

    def recognize(self, image: np.ndarray, with_details: bool = True) -> OCRResult:
        if not self.current_engine:
            raise RuntimeError("Нет доступных OCR движков")
        try:
            if with_details:
                return self.current_engine.recognize_with_details(image)
            else:
                return self.current_engine.recognize(image)
        except Exception as e:
            logger.error(f"Ошибка распознавания: {e}")
            return self._try_fallback(image, with_details)

    def _try_fallback(self, image: np.ndarray, with_details: bool) -> OCRResult:
        for name, engine in self.engines.items():
            if engine.is_available and engine != self.current_engine:
                logger.info(f"Попытка fallback: {name}")
                try:
                    if with_details:
                        return engine.recognize_with_details(image)
                    else:
                        return engine.recognize(image)
                except Exception:
                    continue
        raise RuntimeError("Все OCR движки не смогли обработать изображение")
