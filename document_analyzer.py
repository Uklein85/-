"""
Document Analyzer Module
Детальный анализ документов: таблицы, изображения, текст, структура
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TableData:
    """Данные таблицы"""
    table_id: int
    data: List[List[str]]
    bbox: List[List[float]]
    confidence: float
    page: int


@dataclass
class ImageData:
    """Данные изображения"""
    image_id: int
    image: np.ndarray
    bbox: List[List[float]]
    image_type: str  # 'photo', 'chart', 'diagram', 'logo'
    confidence: float


@dataclass
class DocumentStructure:
    """Структура документа"""
    text_blocks: List[Dict]
    tables: List[TableData]
    images: List[ImageData]
    headings: List[Dict]
    paragraphs: List[Dict]
    metadata: Dict


class DocumentAnalyzer:
    """Анализатор документов для извлечения всех элементов"""
    
    def __init__(self):
        self.table_extractor = TableExtractor()
        self.image_detector = ImageDetector()
        self.structure_analyzer = StructureAnalyzer()
    
    def analyze_page(self, image: np.ndarray, ocr_result) -> DocumentStructure:
        """Полный анализ страницы"""
        
        # Извлечение таблиц
        tables = self.table_extractor.extract_tables(image)
        
        # Детекция изображений
        images = self.image_detector.detect_images(image, ocr_result.blocks)
        
        # Анализ структуры
        structure = self.structure_analyzer.analyze(ocr_result.blocks)
        
        return DocumentStructure(
            text_blocks=structure['text_blocks'],
            tables=tables,
            images=images,
            headings=structure['headings'],
            paragraphs=structure['paragraphs'],
            metadata=structure['metadata']
        )


class TableExtractor:
    """Извлечение таблиц из документов"""
    
    def extract_tables(self, image: np.ndarray) -> List[TableData]:
        """Извлечение таблиц с помощью OpenCV"""
        tables = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Бинаризация
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Детекция горизонтальных линий
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Детекция вертикальных линий
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            # Объединение
            grid = cv2.add(horizontal, vertical)
            
            # Поиск контуров
            contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            table_id = 1
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Фильтр размера таблицы
                if w > 100 and h > 50:
                    # Извлечение ячеек
                    cells = self._extract_cells(binary, x, y, w, h)
                    
                    tables.append(TableData(
                        table_id=table_id,
                        data=cells,
                        bbox=[[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
                        confidence=0.85,
                        page=1
                    ))
                    table_id += 1
        except Exception as e:
            logger.error(f"Ошибка извлечения таблиц: {e}")
        
        return tables
    
    def _extract_cells(self, binary: np.ndarray, x: int, y: int, w: int, h: int) -> List[List[str]]:
        """Извлечение ячеек таблицы"""
        cells = []
        
        try:
            roi = binary[y:y+h, x:x+w]
            
            # Разделение на строки
            row_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, 1))
            rows = cv2.morphologyEx(roi, cv2.MORPH_OPEN, row_kernel)
            
            # Поиск строк
            row_contours, _ = cv2.findContours(rows, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            row_contours = sorted(row_contours, key=lambda c: cv2.boundingRect(c)[1])
            
            for row_contour in row_contours:
                rx, ry, rw, rh = cv2.boundingRect(row_contour)
                row_roi = roi[ry:ry+rh, rx:rx+rw]
                
                # Разделение на ячейки
                col_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rh))
                cols = cv2.morphologyEx(row_roi, cv2.MORPH_OPEN, col_kernel)
                
                col_contours, _ = cv2.findContours(cols, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                col_contours = sorted(col_contours, key=lambda c: cv2.boundingRect(c)[0])
                
                row_data = []
                for col_contour in col_contours:
                    cx, cy, cw, ch = cv2.boundingRect(col_contour)
                    cell_roi = row_roi[cy:cy+ch, cx:cx+cw]
                    
                    # OCR ячейки (упрощенно)
                    cell_text = self._ocr_cell(cell_roi)
                    row_data.append(cell_text)
                
                if row_data:
                    cells.append(row_data)
        except Exception as e:
            logger.warning(f"Ошибка извлечения ячеек: {e}")
        
        return cells
    
    def _ocr_cell(self, cell_image: np.ndarray) -> str:
        """OCR одной ячейки (упрощенно)"""
        # Здесь можно вызвать основной OCR движок
        # Для простоты возвращаем пустую строку
        return ''


class ImageDetector:
    """Детектор изображений в документах"""
    
    def detect_images(self, image: np.ndarray, text_blocks: List) -> List[ImageData]:
        """Детекция изображений"""
        images = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            image_id = 1
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                image_area = image.shape[0] * image.shape[1]
                
                if area > 10000 and area < image_area * 0.5:
                    roi = binary[y:y+h, x:x+w]
                    text_density = np.sum(roi > 0) / roi.size
                    
                    if text_density < 0.3:
                        roi_image = image[y:y+h, x:x+w]
                        img_type = self._classify_image(roi_image)
                        
                        images.append(ImageData(
                            image_id=image_id,
                            image=roi_image,
                            bbox=[[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
                            image_type=img_type,
                            confidence=0.8
                        ))
                        image_id += 1
        except Exception as e:
            logger.error(f"Ошибка детекции изображений: {e}")
        
        return images
    
    def _classify_image(self, image: np.ndarray) -> str:
        """Классификация типа изображения"""
        # Упрощенная классификация
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Проверка на график (много линий)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.1:
            return 'chart'
        elif edge_density > 0.05:
            return 'diagram'
        else:
            return 'photo'


class StructureAnalyzer:
    """Анализатор структуры документа"""
    
    def analyze(self, blocks: List) -> Dict:
        """Анализ структуры документа"""
        text_blocks = []
        headings = []
        paragraphs = []
        
        for block in blocks:
            block_data = {
                'text': block.text,
                'confidence': block.confidence,
                'bbox': block.bbox,
                'font_size': block.font_size,
                'is_bold': block.is_bold
            }
            text_blocks.append(block_data)
            
            # Определение заголовков (крупный шрифт или жирный)
            if block.font_size > 20 or block.is_bold:
                headings.append(block_data)
            else:
                paragraphs.append(block_data)
        
        return {
            'text_blocks': text_blocks,
            'headings': headings,
            'paragraphs': paragraphs,
            'metadata': {
                'total_blocks': len(blocks),
                'headings_count': len(headings),
                'paragraphs_count': len(paragraphs)
            }
        }
