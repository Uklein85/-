"""
DocumentOCR - Автономная система распознавания (АО "СТНГ", 2026)
"""
import sys, os, cv2, numpy as np, json, csv
from pathlib import Path
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage
from PIL import Image
import openpyxl
import pandas as pd
import pymupdf as fitz

from ocr_engines import OCREngineManager

# === ПОТОК ОБРАБОТКИ ===
class ProcessingThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, file_path, ocr_manager):
        super().__init__()
        self.file_path = file_path
        self.ocr_manager = ocr_manager

    def run(self):
        try:
            ext = Path(self.file_path).suffix.lower()
            if ext == '.pdf': result = self._process_pdf()
            elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']: result = self._process_image()
            elif ext in ['.xlsx', '.xls']: result = self._process_excel()
            else: raise ValueError(f"Неподдерживаемый формат: {ext}")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def _process_image(self):
        self.progress.emit(20, "Чтение изображения...")
        image = cv2.imread(self.file_path)
        if image is None: raise ValueError("Ошибка чтения файла")
        
        self.progress.emit(50, "Распознавание...")
        ocr_result = self.ocr_manager.recognize(image)
        self.progress.emit(100, "Готово")
        
        return {
            'file': self.file_path, 'type': 'image', 'text': ocr_result.text,
            'confidence': ocr_result.confidence, 'blocks': ocr_result.blocks,
            'total_pages': 1, 'pages': [{'text': ocr_result.text, 'confidence': ocr_result.confidence}]
        }

    def _process_pdf(self):
        self.progress.emit(10, "Открытие PDF...")
        doc = fitz.open(self.file_path)
        total = len(doc)
        result = {'file': self.file_path, 'type': 'pdf', 'total_pages': total, 'pages': [], 'full_text': '', 'tables': []}
        
        for i in range(total):
            self.progress.emit(10 + int(80 * (i + 1) / total), f"Страница {i+1}/{total}...")
            page = doc[i]
            # Рендер в 300 DPI для качественного OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            ocr_result = self.ocr_manager.recognize(img_array)
            result['pages'].append({'page_number': i+1, 'text': ocr_result.text, 'confidence': ocr_result.confidence})
            result['full_text'] += f"\n--- Страница {i+1} ---\n{ocr_result.text}\n"
            result['tables'].extend(self._detect_tables(img_array))
        doc.close()
        result['text'] = result['full_text']
        result['confidence'] = float(np.mean([p['confidence'] for p in result['pages']]))
        self.progress.emit(100, "Готово")
        return result

    def _process_excel(self):
        self.progress.emit(30, "Чтение Excel...")
        df = pd.read_excel(self.file_path, sheet_name=None)
        text_output = ""
        tables = []
        for sheet_name, data in df.items():
            text_output += f"\n=== Лист: {sheet_name} ===\n"
            text_output += data.to_string(index=False) + "\n"
            tables.append({'sheet': sheet_name, 'data': data.values.tolist()})
        self.progress.emit(100, "Готово")
        return {'file': self.file_path, 'type': 'excel', 'text': text_output, 'tables': tables, 'confidence': 1.0, 'pages': [{'text': text_output, 'confidence': 1.0}]}

    def _detect_tables(self, image):
        # Упрощенная детекция таблиц через OpenCV
        tables = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        grid = cv2.add(horizontal, vertical)
        contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 50:
                tables.append({'bbox': [x, y, w, h], 'data': []})
        return tables

# === ГЛАВНОЕ ОКНО ===
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DocumentOCR — АО «СТНГ» (Автономная версия)")
        self.setMinimumSize(1000, 700)
        self.ocr_manager = OCREngineManager(languages=['ru', 'en'])
        self.current_file = None
        self.current_result = None
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Верхняя панель
        top = QHBoxLayout()
        self.btn_open = QPushButton("📂 Открыть файл (PDF, Excel, IMG)")
        self.btn_open.clicked.connect(self._open_file)
        top.addWidget(self.btn_open)
        
        self.btn_process = QPushButton("▶ Распознать")
        self.btn_process.clicked.connect(self._start_processing)
        self.btn_process.setEnabled(False)
        top.addWidget(self.btn_process)
        top.addStretch()
        layout.addLayout(top)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.lbl_status = QLabel("Ожидание...")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.lbl_status)
        
        # Вкладки
        self.tabs = QTabWidget()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.tabs.addTab(self.text_edit, "📝 Текст")
        layout.addWidget(self.tabs, 1)
        
        # Кнопки экспорта
        export_layout = QHBoxLayout()
        self.btn_export = QPushButton("📤 Вывести готовый файл")
        self.btn_export.clicked.connect(self._export_menu)
        self.btn_export.setEnabled(False)
        export_layout.addWidget(self.btn_export)
        layout.addLayout(export_layout)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите документ", "", "Все файлы (*.pdf *.xlsx *.xls *.jpg *.png *.tiff *.bmp)")
        if path:
            self.current_file = path
            self.btn_process.setEnabled(True)
            self.lbl_status.setText(f"Выбран: {Path(path).name}")

    def _start_processing(self):
        self.thread = ProcessingThread(self.current_file, self.ocr_manager)
        self.thread.progress.connect(lambda v, m: (self.progress_bar.setValue(v), self.lbl_status.setText(m)))
        self.thread.finished.connect(self._on_finished)
        self.thread.error.connect(lambda e: QMessageBox.critical(self, "Ошибка", e))
        self.btn_process.setEnabled(False)
        self.thread.start()

    def _on_finished(self, result):
        self.current_result = result
        self.text_edit.setPlainText(result.get('text', ''))
        self.btn_process.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.lbl_status.setText("✅ Обработка завершена")

    def _export_menu(self):
        if not self.current_result: return
        fmt, _ = QInputDialog.getItem(self, "Экспорт", "Выберите формат:", 
                                      ["HTML", "Excel (XLSX)", "PDF (Сжатый)", "JPG", "PNG"], 0, False)
        if not fmt: return
        
        default_name = Path(self.current_file).stem
        if fmt == "HTML":
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить HTML", f"{default_name}.html", "HTML (*.html)")
            if path: self._save_html(path)
        elif fmt == "Excel (XLSX)":
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить Excel", f"{default_name}.xlsx", "Excel (*.xlsx)")
            if path: self._save_excel(path)
        elif fmt == "PDF (Сжатый)":
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить PDF", f"{default_name}_compressed.pdf", "PDF (*.pdf)")
            if path: self._save_compressed_pdf(path)
        elif fmt in ["JPG", "PNG"]:
            path, _ = QFileDialog.getSaveFileName(self, f"Сохранить {fmt}", f"{default_name}.{fmt.lower()}", f"{fmt} (*.{fmt.lower()})")
            if path: self._save_image(path, fmt.lower())

    def _save_html(self, path):
        text = self.current_result['text'].replace('\n', '<br>')
        html = f"<html><body><h1>Распознанный документ</h1><p>{text}</p></body></html>"
        with open(path, 'w', encoding='utf-8') as f: f.write(html)

    def _save_excel(self, path):
        # Сохраняем таблицы или текст построчно
        df = pd.DataFrame({'Текст': self.current_result['text'].split('\n')})
        df.to_excel(path, index=False)

    def _save_compressed_pdf(self, path):
        # Создаем новый PDF с текстовым слоем и сжатием
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), self.current_result['text'], fontsize=12)
        # Сжатие при сохранении
        doc.ez_save(path, garbage=4, deflate=True, deflate_images=True)

    def _save_image(self, path, fmt):
        # Сохраняем первую страницу как изображение
        if self.current_result['type'] == 'pdf':
            doc = fitz.open(self.current_file)
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
            pix.save(path)
        else:
            img = cv2.imread(self.current_file)
            cv2.imwrite(path, img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
