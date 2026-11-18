"""
DocumentOCR - Программа для распознавания сканированных документов
Разработчик: Уклеин И.В.
Организация: АО "СТНГ"
Город: г. Новосибирск
Год: 2026

Автономная работа: не требует внешних зависимостей (Tesseract, Poppler)
Поддержка: PDF, JPG, PNG, TIFF, BMP, WEBP
OCR движки: RapidOCR (основной), EasyOCR, PaddleOCR (опционально)
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QTabWidget,
    QProgressBar, QComboBox, QCheckBox, QGroupBox, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QStatusBar,
    QMessageBox, QFrame, QScrollArea, QSizePolicy, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor
from PIL import Image
import json
import csv
from datetime import datetime

# ============================================================
# ИСПРАВЛЕНИЕ ПУТЕЙ ДЛЯ .EXE СБОРКИ
# ============================================================

if getattr(sys, 'frozen', False):
    # Запущено как .exe
    base_path = sys._MEIPASS
    exe_dir = os.path.dirname(sys.executable)
    
    # Настройка путей для OCR моделей
    models_dir = os.path.join(exe_dir, 'ocr_models')
    os.environ['RAPIDOCR_MODELS_DIR'] = models_dir
    
    # Настройка TESSDATA если есть локальная папка
    local_tessdata = os.path.join(exe_dir, 'tessdata')
    if os.path.exists(local_tessdata):
        os.environ['TESSDATA_PREFIX'] = local_tessdata
    
    print(f"📁 Режим: .exe сборка")
    print(f"📁 Базовый путь: {base_path}")
    print(f"📁 Папка exe: {exe_dir}")
    print(f"📁 Папка моделей: {models_dir}")
else:
    # Запущено как Python скрипт
    base_path = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Режим: Python скрипт")
    print(f"📁 Базовый путь: {base_path}")

# Импорт OCR движков
from ocr_engines import OCREngineManager


# ============================================================
# ПОТОК ОБРАБОТКИ
# ============================================================

class ProcessingThread(QThread):
    """Поток для обработки документа без блокировки интерфейса"""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    page_processed = pyqtSignal(int, int)
    
    def __init__(self, file_path, ocr_manager, preprocess=True, dpi=300):
        super().__init__()
        self.file_path = file_path
        self.ocr_manager = ocr_manager
        self.preprocess = preprocess
        self.dpi = dpi
        self._is_running = True
    
    def stop(self):
        self._is_running = False
    
    def run(self):
        try:
            ext = Path(self.file_path).suffix.lower()
            
            if ext == '.pdf':
                result = self._process_pdf()
            elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']:
                result = self._process_image()
            else:
                raise ValueError(f"Неподдерживаемый формат: {ext}")
            
            if self._is_running:
                self.finished.emit(result)
                
        except Exception as e:
            self.error.emit(str(e))
    
    def _process_image(self):
        """Обработка изображения"""
        self.progress.emit(10, "Загрузка изображения...")
        image = cv2.imread(self.file_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить: {self.file_path}")
        
        if self.preprocess:
            self.progress.emit(30, "Предобработка...")
            image = self._preprocess_image(image)
        
        self.progress.emit(50, "Распознавание текста...")
        ocr_result = self.ocr_manager.recognize(image, with_details=True)
        
        self.progress.emit(80, "Анализ структуры...")
        
        # Детекция таблиц
        tables = self._detect_tables(image)
        
        # Детекция изображений
        images = self._detect_images(image)
        
        self.progress.emit(100, "Готово")
        
        return {
            'file': self.file_path,
            'type': 'image',
            'engine_used': ocr_result.engine,
            'text': ocr_result.text,
            'confidence': ocr_result.confidence,
            'language': ocr_result.language,
            'blocks': [
                {
                    'text': b.text,
                    'confidence': b.confidence,
                    'font_size': b.font_size,
                    'is_bold': b.is_bold,
                    'language': b.language
                }
                for b in ocr_result.blocks
            ],
            'tables': tables,
            'images': images,
            'blocks_count': len(ocr_result.blocks),
            'total_pages': 1,
            'pages': [{'text': ocr_result.text, 'confidence': ocr_result.confidence}]
        }
    
    def _process_pdf(self):
        """Обработка PDF через PyMuPDF"""
        # ИСПРАВЛЕНИЕ: правильный импорт PyMuPDF
        try:
            import pymupdf as fitz
        except ImportError:
            try:
                import fitz
            except ImportError:
                raise ImportError("Установите PyMuPDF: pip install PyMuPDF")
        
        self.progress.emit(5, "Открытие PDF...")
        
        doc = fitz.open(self.file_path)
        total = len(doc)
        
        result = {
            'file': self.file_path,
            'type': 'pdf',
            'total_pages': total,
            'pages': [],
            'full_text': '',
            'tables': [],
            'images': [],
            'overall_confidence': 0,
            'font_analysis': {}
        }
        
        all_blocks = []
        confidences = []
        
        for i in range(total):
            if not self._is_running:
                break
            
            progress = 10 + int(80 * (i + 1) / total)
            self.progress.emit(progress, f"Страница {i+1}/{total}...")
            self.page_processed.emit(i + 1, total)
            
            page = doc[i]
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)
            
            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
            img_array = img_array.reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            if self.preprocess:
                img_array = self._preprocess_image(img_array)
            
            ocr_result = self.ocr_manager.recognize(img_array, with_details=True)
            
            # Детекция таблиц и изображений для страницы
            tables = self._detect_tables(img_array)
            images = self._detect_images(img_array)
            
            page_data = {
                'page_number': i + 1,
                'text': ocr_result.text,
                'confidence': ocr_result.confidence,
                'blocks_count': len(ocr_result.blocks),
                'tables_count': len(tables),
                'images_count': len(images)
            }
            
            result['pages'].append(page_data)
            result['full_text'] += f"\n--- Страница {i+1} ---\n{ocr_result.text}\n"
            all_blocks.extend(ocr_result.blocks)
            confidences.append(ocr_result.confidence)
            
            result['tables'].extend(tables)
            result['images'].extend(images)
        
        doc.close()
        
        result['overall_confidence'] = float(np.mean(confidences)) if confidences else 0
        result['font_analysis'] = self._analyze_fonts(all_blocks)
        result['engine_used'] = self.ocr_manager.current_engine.name if self.ocr_manager.current_engine else 'Неизвестно'
        result['text'] = result['full_text']
        result['confidence'] = result['overall_confidence']
        result['blocks_count'] = len(all_blocks)
        
        self.progress.emit(100, "Готово")
        return result
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Уменьшение шума
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    def _detect_tables(self, image: np.ndarray) -> list:
        """Детекция таблиц"""
        tables = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            grid = cv2.add(horizontal, vertical)
            contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            table_id = 1
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 50:
                    tables.append({
                        'table_id': table_id,
                        'bbox': [x, y, w, h],
                        'data': [],
                        'note': 'Таблица обнаружена'
                    })
                    table_id += 1
        except Exception as e:
            print(f"⚠️ Ошибка детекции таблиц: {e}")
        
        return tables
    
    def _detect_images(self, image: np.ndarray) -> list:
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
                
                if 10000 < area < image_area * 0.5:
                    roi = binary[y:y+h, x:x+w]
                    text_density = np.sum(roi > 0) / roi.size
                    
                    if text_density < 0.3:
                        images.append({
                            'image_id': image_id,
                            'bbox': [x, y, w, h],
                            'type': 'image',
                            'confidence': 0.8
                        })
                        image_id += 1
        except Exception as e:
            print(f"⚠️ Ошибка детекции изображений: {e}")
        
        return images
    
    def _analyze_fonts(self, blocks: list) -> dict:
        """Анализ шрифтов"""
        if not blocks:
            return {'estimated_font': 'Unknown', 'average_size': 0}
        
        sizes = [b.font_size for b in blocks if b.font_size > 0]
        bold_count = sum(1 for b in blocks if b.is_bold)
        
        avg_size = float(np.mean(sizes)) if sizes else 0
        
        if avg_size < 12:
            font_name = "Мелкий шрифт"
        elif avg_size < 16:
            font_name = "Стандартный (12-14pt)"
        elif avg_size < 20:
            font_name = "Крупный (14-18pt)"
        else:
            font_name = "Заголовочный"
        
        return {
            'estimated_font': font_name,
            'average_size': avg_size,
            'size_range': [float(min(sizes)), float(max(sizes))] if sizes else [0, 0],
            'bold_percentage': (bold_count / len(blocks) * 100) if blocks else 0,
            'total_blocks': len(blocks)
        }


# ============================================================
# ГЛАВНОЕ ОКНО
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DocumentOCR — Распознавание документов")
        self.setMinimumSize(1100, 700)
        self.resize(1300, 800)
        
        self.ocr_manager = OCREngineManager(languages=['ru', 'en'])
        self.current_file = None
        self.current_result = None
        self.processing_thread = None
        
        self._init_ui()
        self._apply_styles()
        self._check_engines()
    
    def _init_ui(self):
        """Инициализация интерфейса"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # Верхняя панель
        main_layout.addWidget(self._create_top_panel())
        
        # Основной разделитель
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([400, 700])
        main_layout.addWidget(splitter, 1)
        
        # Нижняя панель
        main_layout.addWidget(self._create_bottom_panel())
        
        self.statusBar().showMessage("Готов к работе")
    
    # ========== ВЕРХНЯЯ ПАНЕЛЬ ==========
    def _create_top_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("topPanel")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        
        title = QLabel("📄 DocumentOCR")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)
        
        layout.addSpacing(20)
        
        layout.addWidget(QLabel("Движок:"))
        self.engine_combo = QComboBox()
        self.engine_combo.setMinimumWidth(140)
        self._populate_engines()
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        layout.addWidget(self.engine_combo)
        
        layout.addSpacing(10)
        
        self.chk_russian = QCheckBox("Русский")
        self.chk_russian.setChecked(True)
        layout.addWidget(self.chk_russian)
        
        self.chk_english = QCheckBox("Английский")
        self.chk_english.setChecked(True)
        layout.addWidget(self.chk_english)
        
        layout.addSpacing(10)
        
        self.chk_preprocess = QCheckBox("Предобработка")
        self.chk_preprocess.setChecked(True)
        layout.addWidget(self.chk_preprocess)
        
        layout.addSpacing(10)
        
        layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(150, 600)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSingleStep(50)
        layout.addWidget(self.dpi_spin)
        
        layout.addStretch()
        
        help_btn = QPushButton("❓")
        help_btn.setFixedSize(32, 32)
        help_btn.clicked.connect(self._show_help)
        layout.addWidget(help_btn)
        
        return panel
    
    # ========== ЛЕВАЯ ПАНЕЛЬ ==========
    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Компактная область загрузки
        drop_frame = QFrame()
        drop_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        drop_frame.setMinimumHeight(120)
        drop_frame.setMaximumHeight(150)
        drop_frame.setObjectName("dropArea")
        drop_frame.setAcceptDrops(True)
        
        drop_layout = QVBoxLayout(drop_frame)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_file = QLabel("📁 Перетащите файл сюда\nили нажмите «Открыть»")
        self.lbl_file.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_file.setStyleSheet("color: #666; font-size: 12px; padding: 20px;")
        drop_layout.addWidget(self.lbl_file)
        
        layout.addWidget(drop_frame)
        
        # Кнопки действий
        btn_layout = QHBoxLayout()
        
        self.btn_open = QPushButton("📂 Открыть")
        self.btn_open.clicked.connect(self._open_file)
        btn_layout.addWidget(self.btn_open)
        
        self.btn_process = QPushButton("▶ Распознать")
        self.btn_process.setObjectName("primaryButton")
        self.btn_process.clicked.connect(self._start_processing)
        self.btn_process.setEnabled(False)
        btn_layout.addWidget(self.btn_process)
        
        layout.addLayout(btn_layout)
        
        # Информация о файле
        info_group = QGroupBox("Информация о документе")
        info_layout = QVBoxLayout(info_group)
        self.lbl_filename = QLabel("Файл: не выбран")
        self.lbl_filesize = QLabel("Размер: —")
        self.lbl_filetype = QLabel("Тип: —")
        self.lbl_dimensions = QLabel("Разрешение: —")
        
        for lbl in [self.lbl_filename, self.lbl_filesize, 
                    self.lbl_filetype, self.lbl_dimensions]:
            lbl.setWordWrap(True)
            info_layout.addWidget(lbl)
        
        layout.addWidget(info_group)
        
        # Информация о разработчике
        dev_group = QGroupBox("О программе")
        dev_layout = QVBoxLayout(dev_group)
        
        dev_label = QLabel(
            "<b>Разработчик:</b> Уклеин И.В.<br>"
            "<b>Организация:</b> АО «СТНГ»<br>"
            "<b>Город:</b> г. Новосибирск<br>"
            "<b>Год:</b> 2026"
        )
        dev_label.setWordWrap(True)
        dev_label.setStyleSheet("color: #333; font-size: 11px; padding: 5px;")
        dev_layout.addWidget(dev_label)
        
        layout.addWidget(dev_group)
        layout.addStretch()
        
        return panel
    
    # ========== ПРАВАЯ ПАНЕЛЬ ==========
    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_text_tab(), "📝 Распознанный текст")
        self.tabs.addTab(self._create_tables_tab(), "📊 Таблицы")
        self.tabs.addTab(self._create_images_tab(), "🖼️ Изображения")
        self.tabs.addTab(self._create_analysis_tab(), "🔍 Анализ")
        self.tabs.addTab(self._create_pages_tab(), "📄 Страницы")
        
        layout.addWidget(self.tabs)
        
        # Кнопки экспорта
        export_layout = QHBoxLayout()
        
        self.btn_save_txt = QPushButton("💾 Сохранить TXT")
        self.btn_save_txt.clicked.connect(lambda: self._save_result('txt'))
        self.btn_save_txt.setEnabled(False)
        export_layout.addWidget(self.btn_save_txt)
        
        self.btn_save_json = QPushButton("💾 Сохранить JSON")
        self.btn_save_json.clicked.connect(lambda: self._save_result('json'))
        self.btn_save_json.setEnabled(False)
        export_layout.addWidget(self.btn_save_json)
        
        self.btn_copy = QPushButton("📋 Копировать")
        self.btn_copy.clicked.connect(self._copy_text)
        self.btn_copy.setEnabled(False)
        export_layout.addWidget(self.btn_copy)
        
        # Кнопка вывода готового файла
        self.btn_export_file = QPushButton("📤 Вывести готовый файл")
        self.btn_export_file.setObjectName("primaryButton")
        self.btn_export_file.clicked.connect(self._export_ready_file)
        self.btn_export_file.setEnabled(False)
        self.btn_export_file.setToolTip("Сохранить результат в выбранном формате")
        export_layout.addWidget(self.btn_export_file)
        
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        return panel
    
    def _create_text_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setFont(QFont("Consolas", 11))
        self.text_result.setPlaceholderText(
            "Здесь появится распознанный текст после обработки документа..."
        )
        layout.addWidget(self.text_result)
        
        return tab
    
    def _create_tables_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.tables_container = QVBoxLayout()
        self.tables_container.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.tables_container)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        
        layout.addWidget(scroll)
        
        self.lbl_no_tables = QLabel("📊 Таблицы не обнаружены")
        self.lbl_no_tables.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_no_tables.setStyleSheet("color: gray; font-size: 14px; padding: 40px;")
        self.tables_container.addWidget(self.lbl_no_tables)
        
        return tab
    
    def _create_images_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.images_container = QVBoxLayout()
        self.images_container.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.images_container)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        
        layout.addWidget(scroll)
        
        self.lbl_no_images = QLabel("🖼️ Изображения не обнаружены")
        self.lbl_no_images.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_no_images.setStyleSheet("color: gray; font-size: 14px; padding: 40px;")
        self.images_container.addWidget(self.lbl_no_images)
        
        return tab
    
    def _create_analysis_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        stats_group = QGroupBox("📈 Общая статистика")
        stats_layout = QVBoxLayout(stats_group)
        
        self.lbl_confidence = QLabel("Уверенность распознавания: —")
        self.lbl_engine = QLabel("Использованный движок: —")
        self.lbl_blocks = QLabel("Количество блоков: —")
        self.lbl_language = QLabel("Язык: —")
        
        for lbl in [self.lbl_confidence, self.lbl_engine, 
                    self.lbl_blocks, self.lbl_language]:
            stats_layout.addWidget(lbl)
        
        layout.addWidget(stats_group)
        
        font_group = QGroupBox("🔤 Анализ шрифтов")
        font_layout = QVBoxLayout(font_group)
        
        self.lbl_font_type = QLabel("Тип шрифта: —")
        self.lbl_font_size = QLabel("Средний размер: —")
        self.lbl_font_range = QLabel("Диапазон размеров: —")
        self.lbl_bold = QLabel("Жирный текст: —")
        
        for lbl in [self.lbl_font_type, self.lbl_font_size, 
                    self.lbl_font_range, self.lbl_bold]:
            font_layout.addWidget(lbl)
        
        layout.addWidget(font_group)
        layout.addStretch()
        
        return tab
    
    def _create_pages_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.pages_list = QComboBox()
        self.pages_list.currentIndexChanged.connect(self._on_page_changed)
        layout.addWidget(QLabel("Выберите страницу:"))
        layout.addWidget(self.pages_list)
        
        self.page_text = QTextEdit()
        self.page_text.setReadOnly(True)
        self.page_text.setFont(QFont("Consolas", 11))
        layout.addWidget(self.page_text)
        
        self.pages_data = []
        
        return tab
    
    # ========== НИЖНЯЯ ПАНЕЛЬ ==========
    def _create_bottom_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("bottomPanel")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar, 1)
        
        self.lbl_status = QLabel("Ожидание...")
        self.lbl_status.setMinimumWidth(200)
        layout.addWidget(self.lbl_status)
        
        self.btn_stop = QPushButton("⏹ Остановить")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_processing)
        layout.addWidget(self.btn_stop)
        
        return panel
    
    # ========== СТИЛИ ==========
    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
            }
            #topPanel {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            #bottomPanel {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            #dropArea {
                background-color: #fafafa;
                border: 2px dashed #c0c0c0;
                border-radius: 8px;
            }
            #dropArea:hover {
                background-color: #f0f7ff;
                border-color: #0078d4;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
                border-color: #0078d4;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #a0a0a0;
            }
            #primaryButton {
                background-color: #0078d4;
                color: white;
                border: none;
                font-weight: bold;
            }
            #primaryButton:hover {
                background-color: #106ebe;
            }
            #primaryButton:disabled {
                background-color: #c0c0c0;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                padding: 10px;
            }
            QComboBox, QSpinBox {
                padding: 5px 10px;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                background-color: white;
            }
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                text-align: center;
                height: 22px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
            QLabel {
                color: #333333;
            }
        """)
    
    # ========== МЕТОДЫ ==========
    def _populate_engines(self):
        """Заполнение списка доступных движков"""
        self.engine_combo.clear()
        available = self.ocr_manager.get_available_engines()
        if available:
            self.engine_combo.addItems(available)
            current = self.ocr_manager.current_engine.name
            idx = self.engine_combo.findText(current)
            if idx >= 0:
                self.engine_combo.setCurrentIndex(idx)
        else:
            self.engine_combo.addItem("Нет доступных движков")
    
    def _check_engines(self):
        """Проверка доступных движков"""
        available = self.ocr_manager.get_available_engines()
        if not available:
            QMessageBox.warning(
                self, "Внимание",
                "Не найдено ни одного OCR движка!\n\n"
                "Установите хотя бы один:\n"
                "• pip install rapidocr-onnxruntime (рекомендуется)\n"
                "• pip install easyocr\n"
                "• pip install paddlepaddle paddleocr"
            )
    
    def _on_engine_changed(self, engine_name):
        """Смена OCR движка"""
        if self.ocr_manager.switch_engine(engine_name):
            self.statusBar().showMessage(f"Переключено на {engine_name}", 3000)
    
    def _open_file(self):
        """Открытие файла через диалог"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите документ",
            "",
            "Документы (*.pdf *.jpg *.jpeg *.png *.tiff *.bmp *.webp);;Все файлы (*.*)"
        )
        if file_path:
            self._load_file(file_path)
    
    def _load_file(self, file_path):
        """Загрузка информации о файле"""
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Ошибка", f"Файл не найден:\n{file_path}")
            return
        
        self.current_file = file_path
        filename = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        ext = Path(file_path).suffix.lower()
        
        self.lbl_filename.setText(f"Файл: {filename}")
        self.lbl_filesize.setText(f"Размер: {self._format_size(size)}")
        self.lbl_filetype.setText(f"Тип: {ext.upper().replace('.', '')}")
        self.lbl_file.setText(f"✓ {filename}")
        
        if ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']:
            try:
                img = Image.open(file_path)
                self.lbl_dimensions.setText(f"Разрешение: {img.size[0]}×{img.size[1]}")
            except:
                self.lbl_dimensions.setText("Разрешение: —")
        else:
            self.lbl_dimensions.setText("Разрешение: — (PDF)")
        
        self.btn_process.setEnabled(True)
        self.statusBar().showMessage(f"Загружен: {filename}", 3000)
    
    def _format_size(self, size):
        """Форматирование размера файла"""
        for unit in ['Б', 'КБ', 'МБ', 'ГБ']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} ТБ"
    
    def _start_processing(self):
        """Запуск обработки документа"""
        if not self.current_file:
            return
        
        self.processing_thread = ProcessingThread(
            self.current_file,
            self.ocr_manager,
            self.chk_preprocess.isChecked(),
            self.dpi_spin.value()
        )
        self.processing_thread.progress.connect(self._on_progress)
        self.processing_thread.finished.connect(self._on_finished)
        self.processing_thread.error.connect(self._on_error)
        self.processing_thread.page_processed.connect(self._on_page_processed)
        
        self.btn_process.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Запуск обработки...")
        
        self.processing_thread.start()
    
    def _stop_processing(self):
        """Остановка обработки"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.lbl_status.setText("Остановка...")
    
    def _on_progress(self, value, message):
        """Обновление прогресса"""
        self.progress_bar.setValue(value)
        self.lbl_status.setText(message)
    
    def _on_page_processed(self, current, total):
        """Обработка страницы"""
        self.statusBar().showMessage(f"Страница {current} из {total}", 2000)
    
    def _on_finished(self, result):
        """Обработка завершена"""
        self.current_result = result
        self._display_result(result)
        
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save_txt.setEnabled(True)
        self.btn_save_json.setEnabled(True)
        self.btn_copy.setEnabled(True)
        self.btn_export_file.setEnabled(True)
        
        self.progress_bar.setValue(100)
        self.lbl_status.setText("✓ Обработка завершена")
        self.statusBar().showMessage("Готово", 5000)
    
    def _on_error(self, error_msg):
        """Обработка ошибки"""
        QMessageBox.critical(self, "Ошибка обработки", error_msg)
        
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText(f"✗ Ошибка: {error_msg[:50]}")
    
    def _display_result(self, result):
        """Отображение результатов"""
        # Текст
        text = result.get('full_text', result.get('text', ''))
        self.text_result.setPlainText(text)
        
        # Статистика
        conf = result.get('overall_confidence', result.get('confidence', 0))
        self.lbl_confidence.setText(f"Уверенность распознавания: {conf:.1%}")
        self.lbl_engine.setText(f"Использованный движок: {result.get('engine_used', '—')}")
        self.lbl_blocks.setText(f"Количество блоков: {result.get('blocks_count', '—')}")
        self.lbl_language.setText(f"Язык: {result.get('language', '—')}")
        
        # Анализ шрифтов
        font = result.get('font_analysis', {})
        if font:
            self.lbl_font_type.setText(f"Тип шрифта: {font.get('estimated_font', '—')}")
            self.lbl_font_size.setText(f"Средний размер: {font.get('average_size', 0):.1f} px")
            sr = font.get('size_range', [0, 0])
            self.lbl_font_range.setText(f"Диапазон размеров: {sr[0]:.1f} — {sr[1]:.1f} px")
            self.lbl_bold.setText(f"Жирный текст: {font.get('bold_percentage', 0):.1f}%")
        
        # Таблицы
        self._display_tables(result.get('tables', []))
        
        # Изображения
        self._display_images(result.get('images', []))
        
        # Страницы
        self._display_pages(result.get('pages', []))
    
    def _display_tables(self, tables):
        """Отображение таблиц"""
        while self.tables_container.count():
            item = self.tables_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not tables:
            lbl = QLabel("📊 Таблицы не обнаружены")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: gray; font-size: 14px; padding: 40px;")
            self.tables_container.addWidget(lbl)
            return
        
        for i, table in enumerate(tables, 1):
            group = QGroupBox(f"Таблица {i}")
            layout = QVBoxLayout(group)
            
            if table.get('data'):
                data = table['data']
                if data:
                    rows = len(data)
                    cols = max(len(row) for row in data) if data else 0
                    
                    table_widget = QTableWidget(rows, cols)
                    table_widget.horizontalHeader().setSectionResizeMode(
                        QHeaderView.ResizeMode.Stretch
                    )
                    
                    for r, row in enumerate(data):
                        for c, cell in enumerate(row):
                            item = QTableWidgetItem(str(cell) if cell else "")
                            table_widget.setItem(r, c, item)
                    
                    table_widget.setMaximumHeight(min(400, 50 + rows * 30))
                    layout.addWidget(table_widget)
            else:
                layout.addWidget(QLabel(table.get('note', 'Таблица обнаружена')))
            
            self.tables_container.addWidget(group)
    
    def _display_images(self, images):
        """Отображение изображений"""
        while self.images_container.count():
            item = self.images_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not images:
            lbl = QLabel("🖼️ Изображения не обнаружены")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: gray; font-size: 14px; padding: 40px;")
            self.images_container.addWidget(lbl)
            return
        
        for i, img in enumerate(images, 1):
            label = QLabel(f"🖼️ Изображение {i} (тип: {img.get('type', 'image')}, уверенность: {img.get('confidence', 0):.1%})")
            self.images_container.addWidget(label)
    
    def _display_pages(self, pages):
        """Отображение страниц"""
        self.pages_data = pages
        self.pages_list.clear()
        
        if not pages:
            self.page_text.setPlainText("Нет данных о страницах")
            return
        
        for i, page in enumerate(pages, 1):
            conf = page.get('confidence', 0)
            self.pages_list.addItem(f"Страница {i} ({conf:.1%})")
        
        self.pages_list.setCurrentIndex(0)
    
    def _on_page_changed(self, index):
        """Смена страницы"""
        if 0 <= index < len(self.pages_data):
            page = self.pages_data[index]
            self.page_text.setPlainText(page.get('text', ''))
    
    def _save_result(self, format_type):
        """Сохранение результата"""
        if not self.current_result:
            return
        
        if format_type == 'txt':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить текст", "result.txt", "Текстовые файлы (*.txt)"
            )
            if file_path:
                text = self.current_result.get('full_text', 
                                               self.current_result.get('text', ''))
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.statusBar().showMessage(f"Сохранено: {file_path}", 5000)
        
        elif format_type == 'json':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить JSON", "result.json", "JSON файлы (*.json)"
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_result, f, ensure_ascii=False, 
                             indent=2, default=str)
                self.statusBar().showMessage(f"Сохранено: {file_path}", 5000)
    
    def _export_ready_file(self):
        """Кнопка вывода готового файла"""
        if not self.current_result:
            QMessageBox.warning(self, "Внимание", "Сначала обработайте документ")
            return
        
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Вывод готового файла")
        dialog.setText("Выберите формат сохранения:")
        dialog.setInformativeText(
            f"Документ: {os.path.basename(self.current_file)}\n"
            f"Страниц: {self.current_result.get('total_pages', 1)}\n"
            f"Уверенность: {self.current_result.get('confidence', 0):.1%}"
        )
        
        btn_txt = dialog.addButton("📄 TXT (текст)", QMessageBox.ButtonRole.AcceptRole)
        btn_json = dialog.addButton("📋 JSON (полные данные)", QMessageBox.ButtonRole.AcceptRole)
        btn_csv = dialog.addButton("📊 CSV (таблицы)", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
        
        dialog.exec()
        
        clicked = dialog.clickedButton()
        
        if clicked == btn_txt:
            self._save_result('txt')
        elif clicked == btn_json:
            self._save_result('json')
        elif clicked == btn_csv:
            self._export_all_tables()
    
    def _export_all_tables(self):
        """Экспорт всех таблиц в один CSV"""
        if not self.current_result or not self.current_result.get('tables'):
            QMessageBox.warning(self, "Внимание", "Таблицы не найдены")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт всех таблиц", "all_tables.csv", "CSV файлы (*.csv)"
        )
        if file_path:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for i, table in enumerate(self.current_result['tables'], 1):
                    if table.get('data'):
                        writer.writerow([f"=== ТАБЛИЦА {i} ==="])
                        writer.writerows(table['data'])
                        writer.writerow([])
            self.statusBar().showMessage(f"Все таблицы сохранены: {file_path}", 5000)
    
    def _copy_text(self):
        """Копирование текста в буфер"""
        if self.current_result:
            text = self.current_result.get('full_text', 
                                           self.current_result.get('text', ''))
            QApplication.clipboard().setText(text)
            self.statusBar().showMessage("✓ Текст скопирован в буфер обмена", 3000)
    
    def _show_help(self):
        """Показать справку"""
        QMessageBox.information(
            self, "Справка",
            "<h3>DocumentOCR — Распознавание документов</h3>"
            "<p><b>Разработчик:</b> Уклеин И.В.</p>"
            "<p><b>Организация:</b> АО «СТНГ»</p>"
            "<p><b>Город:</b> г. Новосибирск, 2026</p>"
            "<hr>"
            "<p><b>Возможности:</b></p>"
            "<ul>"
            "<li>Распознавание текста из PDF и изображений</li>"
            "<li>Поддержка русского и английского языков</li>"
            "<li>Извлечение таблиц и изображений</li>"
            "<li>Анализ шрифтов и структуры</li>"
            "<li>Несколько OCR движков на выбор</li>"
            "<li>Автономная работа без внешних зависимостей</li>"
            "</ul>"
            "<p><b>Использование:</b></p>"
            "<ol>"
            "<li>Перетащите файл в область загрузки или нажмите «Открыть»</li>"
            "<li>Выберите OCR движок и настройки</li>"
            "<li>Нажмите «Распознать»</li>"
            "<li>Нажмите «Вывести готовый файл» для сохранения</li>"
            "</ol>"
        )


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

def main():
    app = QApplication(sys.argv)
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
