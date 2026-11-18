"""
OCR Engines Module
Все OCR движки для автономной работы без внешних зависимостей
"""

import os
import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================

@dataclass
class TextBlock:
    """Блок распознанного текста"""
    text: str
    confidence: float
    bbox: List[List[float]] = field(default_factory=list)
    font_size: float = 0.0
    is_bold: bool = False
    is_italic: bool = False
    language: str = ''


@dataclass
class ImageRegion:
    """Область изображения в документе"""
    bbox: List[List[float]]
    image: np.ndarray
    confidence: float
    region_type: str  # 'image', 'chart', 'diagram', 'photo'


@dataclass
class OCRResult:
    """Результат OCR обработки"""
    text: str
    blocks: List[TextBlock]
    images: List[ImageRegion]
    confidence: float
    engine: str
    language: str
    metadata: Dict = field(default_factory=dict)


# ============================================================
# АБСТРАКТНЫЙ БАЗОВЫЙ КЛАСС
# ============================================================

class OCREngine(ABC):
    """Абстрактный базовый класс для OCR движков"""
    
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
    
    @abstractmethod
    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        pass


# ============================================================
# ДВИЖОК 1: RAPIDOCR (ONNX-based, легкий)
# ============================================================

class RapidOCREngine(OCREngine):
    """
    RapidOCR - легковесный движок на базе ONNX Runtime
    Не требует внешних зависимостей
    Поддержка русского и английского языков
    """
    
    def __init__(self, languages: List[str] = None):
        super().__init__(languages)
        self.name = "RapidOCR"
        
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.engine = RapidOCR()
            self._is_available = True
            logger.info("✅ RapidOCR успешно инициализирован")
        except ImportError as e:
            logger.warning(f"❌ RapidOCR не установлен: {e}")
            logger.warning("   Решение: pip install rapidocr-onnxruntime")
            self._is_available = False
        except Exception as e:
            logger.error(f"❌ Ошибка RapidOCR: {type(e).__name__}: {e}")
            self._is_available = False
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        """Базовое распознавание"""
        if not self._is_available:
            raise RuntimeError("RapidOCR недоступен")
        
        result, _ = self.engine(image)
        
        if not result:
            return OCRResult(
                text='',
                blocks=[],
                images=[],
                confidence=0.0,
                engine=self.name,
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
        """Распознавание с детальной информацией"""
        result = self.recognize(image)
        
        # Анализ размеров шрифтов
        for block in result.blocks:
            if block.bbox and len(block.bbox) >= 2:
                height = abs(block.bbox[2][1] - block.bbox[0][1])
                block.font_size = height
                block.is_bold = self._estimate_bold(image, block.bbox)
        
        # Детекция изображений
        result.images = self._detect_images(image, result.blocks)
        
        return result
    
    def _detect_language(self, text: str) -> str:
        """Определение языка текста"""
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        latin_chars = sum(1 for c in text if '\u0041' <= c <= '\u005A' or '\u0061' <= c <= '\u007A')
        
        if cyrillic_chars > latin_chars:
            return 'ru'
        elif latin_chars > cyrillic_chars:
            return 'en'
        return 'unknown'
    
    def _estimate_bold(self, image: np.ndarray, bbox: List) -> bool:
        """Оценка жирности шрифта"""
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
            
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            density = np.sum(binary > 0) / binary.size
            return density > 0.35
        except Exception:
            return False
    
    def _detect_images(self, image: np.ndarray, blocks: List[TextBlock]) -> List[ImageRegion]:
        """Детекция изображений в документе"""
        images = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Поиск контуров
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                image_area = image.shape[0] * image.shape[1]
                
                # Фильтр: область должна быть достаточно большой
                if area > 10000 and area < image_area * 0.5:
                    # Проверка, что это не текст (мало текста внутри)
                    roi = binary[y:y+h, x:x+w]
                    text_density = np.sum(roi > 0) / roi.size
                    
                    if text_density < 0.3:  # Мало текста = изображение
                        roi_image = image[y:y+h, x:x+w]
                        images.append(ImageRegion(
                            bbox=[[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
                            image=roi_image,
                            confidence=0.8,
                            region_type='image'
                        ))
        except Exception as e:
            logger.warning(f"Ошибка детекции изображений: {e}")
        
        return images


# ============================================================
# ДВИЖОК 2: EASYOCR (PyTorch-based)
# ============================================================

class EasyOCREngine(OCREngine):
    """
    EasyOCR - движок на базе PyTorch
    Поддержка 80+ языков
    Высокое качество распознавания
    """
    
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        super().__init__(languages)
        self.name = "EasyOCR"
        self.gpu = gpu
        
        try:
            import easyocr
            lang_map = {'ru': 'ru', 'en': 'en', 'rus': 'ru', 'eng': 'en'}
            self.easyocr_langs = [lang_map.get(l, l) for l in self.languages]
            
            logger.info(f"Инициализация EasyOCR с языками: {self.easyocr_langs}")
            self.reader = easyocr.Reader(
                self.easyocr_langs, 
                gpu=self.gpu,
                verbose=False
            )
            self._is_available = True
            logger.info("✅ EasyOCR успешно инициализирован")
        except ImportError as e:
            logger.warning(f"❌ EasyOCR не установлен: {e}")
            logger.warning("   Решение: pip install easyocr")
            self._is_available = False
        except Exception as e:
            logger.error(f"❌ Ошибка EasyOCR: {type(e).__name__}: {e}")
            self._is_available = False
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available:
            raise RuntimeError("EasyOCR недоступен")
        
        results = self.reader.readtext(image)
        
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
    
    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        result = self.recognize(image)
        
        for block in result.blocks:
            if block.bbox and len(block.bbox) >= 2:
                height = abs(block.bbox[2][1] - block.bbox[0][1])
                block.font_size = height
        
        result.images = self._detect_images(image)
        
        return result
    
    def _detect_images(self, image: np.ndarray) -> List[ImageRegion]:
        """Детекция изображений"""
        images = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                image_area = image.shape[0] * image.shape[1]
                
                if area > 10000 and area < image_area * 0.5:
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
        except Exception:
            pass
        
        return images


# ============================================================
# ДВИЖОК 3: PADDLEOCR (лучшее качество)
# ============================================================

class PaddleOCREngine(OCREngine):
    """
    PaddleOCR от Baidu
    Лучшее качество для русского языка
    Поддержка таблиц и layout analysis
    """
    
    def __init__(self, languages: List[str] = None, use_gpu: bool = False):
        super().__init__(languages)
        self.name = "PaddleOCR"
        
        try:
            from paddleocr import PaddleOCR
            
            lang_map = {'ru': 'ru', 'en': 'en', 'rus': 'ru', 'eng': 'en'}
            primary_lang = lang_map.get(self.languages[0], 'ru')
            
            logger.info(f"Инициализация PaddleOCR с языком: {primary_lang}")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=primary_lang,
                use_gpu=use_gpu,
                show_log=False
            )
            self._is_available = True
            logger.info("✅ PaddleOCR успешно инициализирован")
        except ImportError as e:
            logger.warning(f"❌ PaddleOCR не установлен: {e}")
            logger.warning("   Решение: pip install paddlepaddle paddleocr")
            self._is_available = False
        except Exception as e:
            logger.error(f"❌ Ошибка PaddleOCR: {type(e).__name__}: {e}")
            self._is_available = False
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available:
            raise RuntimeError("PaddleOCR недоступен")
        
        result = self.ocr.ocr(image, cls=True)
        
        if not result or not result[0]:
            return OCRResult(
                text='',
                blocks=[],
                images=[],
                confidence=0.0,
                engine=self.name,
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
    
    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        result = self.recognize(image)
        
        for block in result.blocks:
            if block.bbox and len(block.bbox) >= 2:
                height = abs(block.bbox[2][1] - block.bbox[0][1])
                block.font_size = height
        
        return result


# ============================================================
# МЕНЕДЖЕР ДВИЖКОВ
# ============================================================

class OCREngineManager:
    """Менеджер OCR движков с автоматическим выбором лучшего"""
    
    PRIORITY = ['PaddleOCR', 'EasyOCR', 'RapidOCR']
    
    def __init__(self, preferred_engine: str = None, languages: List[str] = None):
        self.languages = languages or ['ru', 'en']
        self.engines: Dict[str, OCREngine] = {}
        self.current_engine: Optional[OCREngine] = None
        
        self._init_all_engines(preferred_engine)
    
    def _init_all_engines(self, preferred: str = None):
        """Инициализация всех движков"""
        logger.info("Инициализация OCR движков...")
        
        engine_classes = [
            ('RapidOCR', lambda: RapidOCREngine(self.languages)),
            ('EasyOCR', lambda: EasyOCREngine(self.languages)),
            ('PaddleOCR', lambda: PaddleOCREngine(self.languages)),
        ]
        
        for name, factory in engine_classes:
            try:
                engine = factory()
                self.engines[name] = engine
                status = "✓ доступен" if engine.is_available else "✗ недоступен"
                logger.info(f"  {name}: {status}")
            except Exception as e:
                logger.warning(f"Ошибка инициализации {name}: {e}")
        
        if preferred and preferred in self.engines and self.engines[preferred].is_available:
            self.current_engine = self.engines[preferred]
        else:
            for engine_name in self.PRIORITY:
                if engine_name in self.engines and self.engines[engine_name].is_available:
                    self.current_engine = self.engines[engine_name]
                    break
        
        if self.current_engine:
            logger.info(f"Выбран движок: {self.current_engine.name}")
        else:
            logger.error("Ни один OCR движок не доступен!")
    
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
                logger.info(f"Попытка использовать fallback: {name}")
                try:
                    if with_details:
                        return engine.recognize_with_details(image)
                    else:
                        return engine.recognize(image)
                except Exception:
                    continue
        raise RuntimeError("Все OCR движки не смогли обработать изображение")
