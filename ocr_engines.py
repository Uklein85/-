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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# УТИЛИТА: ПОЛУЧЕНИЕ ПУТИ К МОДЕЛЯМ
# ============================================================
def get_models_dir() -> Path:
    """Получить папку для хранения моделей"""
    if getattr(sys, 'frozen', False):
        # В .exe сборке - рядом с exe файлом
        return Path(sys.executable).parent / 'ocr_models'
    else:
        # При запуске из Python - в папке проекта
        return Path(__file__).parent / 'ocr_models'


def ensure_rapidocr_models():
    """Убедиться что модели RapidOCR доступны"""
    models_dir = get_models_dir() / 'rapidocr'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    required_models = [
        'ch_PP-OCRv4_det_infer.onnx',
        'ch_PP-OCRv4_rec_infer.onnx',
        'ch_ppocr_mobile_v2.0_cls_infer.onnx'
    ]
    
    missing = []
    for name in required_models:
        dest = models_dir / name
        if not dest.exists() or dest.stat().st_size < 1000000:
            missing.append(name)
    
    if missing:
        raise FileNotFoundError(
            f"Отсутствуют OCR модели: {', '.join(missing)}\n"
            f"Ожидаемая папка: {models_dir}\n"
            f"Запустите download_models.py перед сборкой .exe"
        )
    
    return models_dir


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
            models_dir = ensure_rapidocr_models()
            os.environ['RAPIDOCR_MODELS_DIR'] = str(models_dir)
            
            from rapidocr_onnxruntime import RapidOCR
            
            self._engine = RapidOCR(
                det_model_path=str(models_dir / 'ch_PP-OCRv4_det_infer.onnx'),
                rec_model_path=str(models_dir / 'ch_PP-OCRv4_rec_infer.onnx'),
                cls_model_path=str(models_dir / 'ch_ppocr_mobile_v2.0_cls_infer.onnx'),
            )
            
            self._is_available = True
            logger.info(f"RapidOCR инициализирован (модели: {models_dir})")
            
        except ImportError as e:
            logger.warning(f"RapidOCR не установлен: {e}")
            logger.warning("   Решение: pip install rapidocr-onnxruntime")
            self._is_available = False
            
        except Exception as e:
            logger.error(f"Ошибка RapidOCR: {type(e).__name__}: {e}")
            self._is_available = False

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available or self._engine is None:
            raise RuntimeError("RapidOCR недоступен")
        
        result, _ = self._engine(image)
        
        if not result:
            return OCRResult(
                text='', blocks=[], images=[],
                confidence=0.0, engine=self.name,
                language='+'.join(self.languages)
            )
        
        blocks = []
        texts = []
        confidences = []
        
        for detection in result:
            bbox, text, confidence = detection
            blocks.append(TextBlock(
                text=text,
                confidence=confidence,
                bbox=bbox,
                language=self._detect_language(text)
            ))
            texts.append(text)
            confidences.append(confidence)
        
        return OCRResult(
            text='\n'.join(texts),
            blocks=blocks,
            images=[],
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            engine=self.name,
            language='+'.join(self.languages)
        )

    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        result = self.recognize(image)
        
        for block in result.blocks:
            if block.bbox and len(block.bbox) >= 2:
                height = abs(block.bbox[2][1] - block.bbox[0][1])
                block.font_size = height
                block.is_bold = self._estimate_bold(image, block.bbox)
        
        result.images = self._detect_images(image)
        return result

    def _detect_language(self, text: str) -> str:
        cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        latin = sum(1 for c in text if '\u0041' <= c <= '\u005A' or '\u0061' <= c <= '\u007A')
        
        if cyrillic > latin:
            return 'ru'
        elif latin > cyrillic:
            return 'en'
        return 'unknown'

    def _estimate_bold(self, image: np.ndarray, bbox: List) -> bool:
        try:
            x_coords = [int(p[0]) for p in bbox]
            y_coords = [int(p[1]) for p in bbox]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            if x_max <= x_min or y_max <= y_min:
                return False
            
            roi = image[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                return False
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            density = np.sum(binary > 0) / binary.size
            
            return density > 0.35
        except:
            return False

    def _detect_images(self, image: np.ndarray) -> List[ImageRegion]:
        images = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                image_area = image.shape[0] * image.shape[1]
                
                if 10000 < area < image_area * 0.5:
                    roi = binary[y:y+h, x:x+w]
                    text_density = np.sum(roi > 0) / roi.size
                    
                    if text_density < 0.3:
                        roi_image = image[y:y+h, x:x+w]
                        images.append(ImageRegion(
                            bbox=[[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
                            image=roi_image,
                            confidence=0.8,
                            region_type='image'
                        ))
        except:
            pass
        
        return images


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
            
            try:
                import torchvision
                import torchvision.ops
                _ = torchvision.ops.nms
            except (ImportError, AttributeError, RuntimeError) as e:
                logger.warning(f"torchvision.nms недоступен: {e}")
                logger.warning("   EasyOCR может не работать")
            
            self._reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
            self._is_available = True
            logger.info("EasyOCR инициализирован")
            
        except ImportError as e:
            logger.warning(f"EasyOCR не установлен: {e}")
            self._is_available = False
            
        except Exception as e:
            logger.error(f"Ошибка EasyOCR: {type(e).__name__}: {e}")
            logger.warning("   EasyOCR недоступен. Используйте RapidOCR.")
            self._is_available = False

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available or self._reader is None:
            raise RuntimeError("EasyOCR недоступен")
        
        results = self._reader.readtext(image)
        blocks = []
        texts = []
        confidences = []
        
        for detection in results:
            bbox, text, confidence = detection
            blocks.append(TextBlock(
                text=text,
                confidence=confidence,
                bbox=bbox,
                language='ru' if any('\u0400' <= c <= '\u04FF' for c in text) else 'en'
            ))
            texts.append(text)
            confidences.append(confidence)
        
        return OCRResult(
            text='\n'.join(texts),
            blocks=blocks,
            images=[],
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            engine=self.name,
            language='+'.join(self.languages)
        )


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
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=primary_lang,
                    use_gpu=use_gpu,
                )
            except TypeError:
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=primary_lang,
                )
            
            self._is_available = True
            logger.info(f"PaddleOCR инициализирован (язык: {primary_lang})")
            
        except ImportError as e:
            logger.warning(f"PaddleOCR не установлен: {e}")
            self._is_available = False
            
        except Exception as e:
            logger.error(f"Ошибка PaddleOCR: {type(e).__name__}: {e}")
            self._is_available = False

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available or self._ocr is None:
            raise RuntimeError("PaddleOCR недоступен")
        
        result = self._ocr.ocr(image, cls=True)
        
        if not result or not result[0]:
            return OCRResult(
                text='', blocks=[], images=[],
                confidence=0.0, engine=self.name,
                language='+'.join(self.languages)
            )
        
        blocks = []
        texts = []
        confidences = []
        
        for line in result[0]:
            bbox, (text, confidence) = line
            blocks.append(TextBlock(
                text=text,
                confidence=confidence,
                bbox=bbox,
                language='ru' if any('\u0400' <= c <= '\u04FF' for c in text) else 'en'
            ))
            texts.append(text)
            confidences.append(confidence)
        
        return OCRResult(
            text='\n'.join(texts),
            blocks=blocks,
            images=[],
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            engine=self.name,
            language='+'.join(self.languages)
        )
# ============================================================
# МЕНЕДЖЕР ДВИЖКОВ
# ============================================================
class OCREngineManager:
    """
    Менеджер OCR движков.

    Допустимые вызовы (все эквивалентны и безопасны):
        OCREngineManager()
        OCREngineManager(['ru', 'en'])                 # позиционный список языков
        OCREngineManager(languages=['ru', 'en'])
        OCREngineManager('RapidOCR', ['ru', 'en'])
        OCREngineManager(preferred_engine='RapidOCR', languages=['ru', 'en'])
    """

    PRIORITY = ['RapidOCR', 'PaddleOCR', 'EasyOCR']

    def __init__(self, preferred_engine=None, languages=None):
        # ---------- Нормализация аргументов ----------
        # Список/кортеж на первой позиции = языки, а не имя движка.
        # Именно это раньше давало TypeError: unhashable type: 'list'.
        if isinstance(preferred_engine, (list, tuple, set)):
            if languages is None:
                languages = list(preferred_engine)
            preferred_engine = None
        elif preferred_engine is not None and not isinstance(preferred_engine, str):
            preferred_engine = str(preferred_engine)

        if languages is None:
            languages = ['ru', 'en']
        elif isinstance(languages, str):
            languages = [languages]
        else:
            languages = [str(x) for x in languages]
        if not languages:
            languages = ['ru', 'en']

        self.languages: List[str] = languages
        self.engines: Dict[str, OCREngine] = {}      # ключи — ТОЛЬКО строки
        self.current_engine: Optional[OCREngine] = None

        self._init_all_engines(preferred_engine)

    def _init_all_engines(self, preferred: Optional[str] = None):
        logger.info("Инициализация OCR движков...")
        logger.info("Языки: %s", ", ".join(self.languages))
        try:
            logger.info("Папка моделей: %s", get_models_dir())
        except Exception:
            pass

        # Опциональные тяжёлые движки можно отключить (ускоряет старт .exe):
        # set DOCOCR_DISABLE_OPTIONAL=1
        skip_optional = os.environ.get('DOCOCR_DISABLE_OPTIONAL', '0') == '1'

        factories = [
            ('RapidOCR', lambda: RapidOCREngine(list(self.languages))),
        ]
        if not skip_optional:
            factories += [
                ('PaddleOCR', lambda: PaddleOCREngine(list(self.languages))),
                ('EasyOCR', lambda: EasyOCREngine(list(self.languages))),
            ]

        for name, factory in factories:
            engine = None
            try:
                engine = factory()
            except Exception as e:
                logger.warning("  %s: ошибка инициализации (%s: %s)",
                               name, type(e).__name__, e)
            if engine is None:
                continue
            self.engines[str(name)] = engine
            logger.info("  %s: %s", name,
                        "доступен" if engine.is_available else "недоступен")

        # ---------- Выбор движка (только строковые ключи) ----------
        chosen: Optional[OCREngine] = None

        if isinstance(preferred, str) and preferred:
            candidate = self.engines.get(preferred)
            if candidate is not None and candidate.is_available:
                chosen = candidate
            else:
                logger.warning("Движок '%s' недоступен, выбираю автоматически", preferred)

        if chosen is None:
            for engine_name in self.PRIORITY:
                candidate = self.engines.get(engine_name)
                if candidate is not None and candidate.is_available:
                    chosen = candidate
                    break

        if chosen is None:
            for candidate in self.engines.values():
                if candidate.is_available:
                    chosen = candidate
                    break

        self.current_engine = chosen

        if self.current_engine is not None:
            logger.info("Выбран движок: %s", self.current_engine.name)
        else:
            logger.error("Ни один OCR движок не доступен")
            logger.error("Проверьте модели в папке ocr_models/rapidocr "
                         "и наличие пакета rapidocr-onnxruntime")

    # ---------- Публичные методы ----------
    def get_available_engines(self) -> List[str]:
        return [name for name, engine in self.engines.items() if engine.is_available]

    def switch_engine(self, engine_name) -> bool:
        if not isinstance(engine_name, str):
            return False
        engine = self.engines.get(engine_name)
        if engine is not None and engine.is_available:
            self.current_engine = engine
            logger.info("Переключено на: %s", engine_name)
            return True
        return False

    def recognize(self, image: np.ndarray, with_details: bool = True) -> OCRResult:
        if self.current_engine is None:
            raise RuntimeError("Нет доступных OCR движков")
        try:
            if with_details:
                return self.current_engine.recognize_with_details(image)
            return self.current_engine.recognize(image)
        except Exception as e:
            logger.error("Ошибка распознавания (%s): %s", type(e).__name__, e)
            return self._try_fallback(image, with_details)

    def _try_fallback(self, image: np.ndarray, with_details: bool) -> OCRResult:
        for name, engine in self.engines.items():
            if not engine.is_available or engine is self.current_engine:
                continue
            logger.info("Попытка fallback: %s", name)
            try:
                if with_details:
                    result = engine.recognize_with_details(image)
                else:
                    result = engine.recognize(image)
                self.current_engine = engine
                return result
            except Exception as e:
                logger.warning("Fallback %s не сработал: %s", name, e)
                continue
        raise RuntimeError("Все OCR движки не смогли обработать изображение")

