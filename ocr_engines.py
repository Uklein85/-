"""
OCR Engines Module - Автономная версия (модели вшиты в сборку)
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

def get_models_dir() -> Path:
    """Получить папку моделей (работает и в .exe, и в Python)"""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent / 'ocr_models'
    else:
        return Path(__file__).parent / 'ocr_models'

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

class RapidOCREngine(OCREngine):
    def __init__(self, languages: List[str] = None):
        super().__init__(languages)
        self.name = "RapidOCR"
        self._engine = None
        try:
            models_dir = get_models_dir() / 'rapidocr'
            if not models_dir.exists():
                raise FileNotFoundError(f"Папка моделей не найдена: {models_dir}. Запустите download_models.py")
            
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR(
                det_model_path=str(models_dir / 'ch_PP-OCRv4_det_infer.onnx'),
                rec_model_path=str(models_dir / 'ch_PP-OCRv4_rec_infer.onnx'),
                cls_model_path=str(models_dir / 'ch_ppocr_mobile_v2.0_cls_infer.onnx'),
            )
            self._is_available = True
            logger.info(f"✅ RapidOCR инициализирован (автономный режим)")
        except Exception as e:
            logger.error(f"❌ Ошибка RapidOCR: {e}")
            self._is_available = False

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available: raise RuntimeError("RapidOCR недоступен")
        result, _ = self._engine(image)
        if not result:
            return OCRResult(text='', blocks=[], images=[], confidence=0.0, engine=self.name, language='+'.join(self.languages))
        
        blocks, texts, confidences = [], [], []
        for detection in result:
            bbox, text, confidence = detection
            blocks.append(TextBlock(text=text, confidence=confidence, bbox=bbox, language='ru' if any('\u0400' <= c <= '\u04FF' for c in text) else 'en'))
            texts.append(text)
            confidences.append(confidence)
            
        return OCRResult(text='\n'.join(texts), blocks=blocks, images=[], 
                         confidence=float(np.mean(confidences)), engine=self.name, language='+'.join(self.languages))

class OCREngineManager:
    def __init__(self, languages: List[str] = None):
        self.languages = languages or ['ru', 'en']
        self.engines = {}
        self.current_engine = None
        self._init_engines()

    def _init_engines(self):
        engine = RapidOCREngine(self.languages)
        self.engines['RapidOCR'] = engine
        if engine.is_available:
            self.current_engine = engine
            logger.info("🎯 Выбран движок: RapidOCR")
        else:
            logger.error("❌ Нет доступных движков!")

    def get_available_engines(self) -> List[str]:
        return [n for n, e in self.engines.items() if e.is_available]

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self.current_engine: raise RuntimeError("Нет OCR движков")
        return self.current_engine.recognize(image)
