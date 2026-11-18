"""
DocumentOCR - Программа для распознавания сканированных документов
Разработчик: Уклеин И.В.
Организация: АО "СТНГ"
Город: г. Новосибирск
Год: 2026
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
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PIL import Image
import json
import csv
from datetime import datetime

from ocr_engines import OCREngineManager
from document_analyzer import DocumentAnalyzer


class ProcessingThread(QThread):
    """Поток обработки документа"""
    
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
        self.analyzer = DocumentAnalyzer()
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
        self.progress.emit(10, "Загрузка изображения...")
        image = cv2.imread(self.file_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить: {self.file_path}")
        
        if self.preprocess:
            self.progress.emit(30, "Предобработка...")
            image = self._preprocess(image)
        
        self.progress.emit(50, "Распознавание текста...")
        ocr_result = self.ocr_manager.recognize(image, with_details=True)
        
        self.progress.emit(70, "Анализ структуры...")
        structure = self.analyzer.analyze_page(image, ocr_result)
        
        self.progress.emit(90, "Формирование результатов...")
        
        result = {
            'file': self.file_path,
            'type': 'image',
            'engine_used': ocr_result.engine,
            'text': ocr_result.text,
            'confidence': ocr_result.confidence,
            'language': ocr_result.language,
            'blocks': [{'text': b.text, 'confidence': b.confidence, 'font_size': b.font_size} for b in ocr_result.blocks],
            'tables': [{'table_id': t.table_id, 'data': t.data, 'confidence': t.confidence} for t in structure.tables],
            'images': [{'image_id': i.image_id, 'type': i.image_type, 'confidence': i.confidence} for i in structure.images],
            'headings': structure.headings,
            'metadata': structure.metadata,
            'total_pages': 1,
            'pages': [{'text': ocr_result.text, 'confidence': ocr_result.confidence}]
        }
        
        self.progress.emit(100, "Готово")
        return result
    
    def _process_pdf(self):
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        
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
            'engine_used': self.ocr_manager.current_engine.name if self.ocr_manager.current_engine else 'Неизвестно'
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
                img_array = self._preprocess(img_array)
            
            ocr_result = self.ocr_manager.recognize(img_array, with_details=True)
            structure = self.analyzer.analyze_page(img_array, ocr_result)
            
            page_data = {
                'page_number': i + 1,
                'text': ocr_result.text,
                'confidence': ocr_result.confidence,
                'blocks_count': len(ocr_result.blocks),
                'tables_count': len(structure.tables),
                'images_count': len(structure.images)
            }
            
            result['pages'].append(page_data)
            result['full_text'] += f"\n--- Страница {i+1} ---\n{ocr_result.text}\n"
            all_blocks.extend(ocr_result.blocks)
            confidences.append(ocr_result.confidence)
            
            result['tables'].extend([{'table_id': t.table_id, 'data': t.data, 'page': i+1} for t in structure.tables])
            result['images'].extend([{'image_id': img.image_id, 'type': img.image_type, 'page': i+1} for img in structure.images])
        
        doc.close()
        
        result['overall_confidence'] = float(np.mean(confidences)) if confidences else 0
        result['text'] = result['full_text']
        result['confidence'] = result['overall_confidence']
        
        self.progress.emit(100, "Готово")
        return result
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Уменьшение шума
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


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
    
    def _init_ui(self):
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
        
        layout.addStretch()
        
        return panel
    
    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Область загрузки
        drop_frame = QFrame()
        drop_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        drop_frame.setMinimumHeight(120)
        drop_layout = QVBoxLayout(drop_frame)
        
        self.lbl_file = QLabel("📁 Перетащите файл сюда или нажмите «Открыть»")
        self.lbl_file.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_file.setStyleSheet("color: #666; font-size: 12px; padding: 20px;")
        drop_layout.addWidget(self.lbl_file)
        
        layout.addWidget(drop_frame)
        
        # Кнопки
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
    
    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_text_tab(), "📝 Текст")
        self.tabs.addTab(self._create_tables_tab(), "📊 Таблицы")
        self.tabs.addTab(self._create_images_tab(), "🖼️ Изображения")
        self.tabs.addTab(self._create_analysis_tab(), "🔍 Анализ")
        
        layout.addWidget(self.tabs)
        
        # Кнопки экспорта
        export_layout = QHBoxLayout()
        
        self.btn_export = QPushButton("📤 Вывести готовый файл")
        self.btn_export.setObjectName("primaryButton")
        self.btn_export.clicked.connect(self._export_ready_file)
        self.btn_export.setEnabled(False)
        export_layout.addWidget(self.btn_export)
        
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        return panel
    
    def _create_text_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setFont(QFont("Consolas", 11))
        layout.addWidget(self.text_result)
        
        return tab
    
    def _create_tables_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.tables_container = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.tables_container)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        return tab
    
    def _create_images_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.images_container = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.images_container)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        return tab
    
    def _create_analysis_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.lbl_stats = QLabel("Статистика будет здесь")
        layout.addWidget(self.lbl_stats)
        
        layout.addStretch()
        
        return tab
    
    def _create_bottom_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("bottomPanel")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar, 1)
        
        self.lbl_status = QLabel("Ожидание...")
        self.lbl_status.setMinimumWidth(200)
        layout.addWidget(self.lbl_status)
        
        return panel
    
    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f7; }
            #topPanel, #bottomPanel {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #f0f0f0; border-color: #0078d4; }
            #primaryButton {
                background-color: #0078d4;
                color: white;
                border: none;
                font-weight: bold;
            }
            #primaryButton:hover { background-color: #106ebe; }
            QTextEdit {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                padding: 10px;
            }
            QProgressBar {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                text-align: center;
                height: 22px;
            }
            QProgressBar::chunk { background-color: #0078d4; border-radius: 3px; }
        """)
    
    def _populate_engines(self):
        self.engine_combo.clear()
        available = self.ocr_manager.get_available_engines()
        if available:
            self.engine_combo.addItems(available)
            self.engine_combo.setCurrentIndex(0)
        else:
            self.engine_combo.addItem("Нет доступных движков")
            QMessageBox.warning(self, "Внимание", "Не найдено ни одного OCR движка!\n\nУстановите:\n• pip install rapidocr-onnxruntime")
    
    def _open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите документ", "",
            "Документы (*.pdf *.jpg *.jpeg *.png *.tiff *.bmp *.webp);;Все файлы (*.*)"
        )
        if file_path:
            self.current_file = file_path
            self.lbl_file.setText(f"✓ {Path(file_path).name}")
            self.btn_process.setEnabled(True)
    
    def _start_processing(self):
        if not self.current_file:
            return
        
        self.processing_thread = ProcessingThread(
            self.current_file, self.ocr_manager, self.chk_preprocess.isChecked()
        )
        self.processing_thread.progress.connect(self._on_progress)
        self.processing_thread.finished.connect(self._on_finished)
        self.processing_thread.error.connect(self._on_error)
        
        self.btn_process.setEnabled(False)
        self.progress_bar.setValue(0)
        self.processing_thread.start()
    
    def _on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.lbl_status.setText(message)
    
    def _on_finished(self, result):
        self.current_result = result
        self._display_result(result)
        
        self.btn_process.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.progress_bar.setValue(100)
        self.lbl_status.setText("✓ Готово")
    
    def _on_error(self, error_msg):
        QMessageBox.critical(self, "Ошибка", error_msg)
        self.btn_process.setEnabled(True)
        self.progress_bar.setValue(0)
    
    def _display_result(self, result):
        self.text_result.setPlainText(result.get('text', ''))
        
        # Таблицы
        while self.tables_container.count():
            item = self.tables_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for table in result.get('tables', []):
            group = QGroupBox(f"Таблица {table['table_id']}")
            layout = QVBoxLayout(group)
            
            if table['data']:
                rows = len(table['data'])
                cols = max(len(row) for row in table['data'])
                table_widget = QTableWidget(rows, cols)
                
                for r, row in enumerate(table['data']):
                    for c, cell in enumerate(row):
                        table_widget.setItem(r, c, QTableWidgetItem(str(cell)))
                
                layout.addWidget(table_widget)
            
            self.tables_container.addWidget(group)
        
        # Изображения
        while self.images_container.count():
            item = self.images_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for img in result.get('images', []):
            label = QLabel(f"🖼️ Изображение {img['image_id']} ({img['type']})")
            self.images_container.addWidget(label)
        
        # Статистика
        stats = result.get('metadata', {})
        self.lbl_stats.setText(
            f"Всего блоков: {stats.get('total_blocks', 0)}\n"
            f"Заголовков: {stats.get('headings_count', 0)}\n"
            f"Таблиц: {len(result.get('tables', []))}\n"
            f"Изображений: {len(result.get('images', []))}\n"
            f"Уверенность: {result.get('confidence', 0):.1%}"
        )
    
    def _export_ready_file(self):
        if not self.current_result:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", "result.json", "JSON (*.json);;Text (*.txt)"
        )
        
        if file_path:
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_result, f, ensure_ascii=False, indent=2, default=str)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.current_result.get('text', ''))
            
            self.statusBar().showMessage(f"Сохранено: {file_path}", 5000)


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
