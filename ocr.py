import sys
import cv2
import numpy as np
import easyocr
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QCheckBox, QTextEdit, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# ==========================================
# Background Thread for Continuous OCR
# ==========================================
class OCRWorker(QThread):
    result_ready = pyqtSignal(list)

    def __init__(self, reader, allowed_chars):
        super().__init__()
        self.reader = reader
        self.allowed_chars = allowed_chars
        self.is_running = False
        self.current_target = None

    def run(self):
        self.is_running = True
        while self.is_running:
            if self.current_target is not None:
                frame_to_process = self.current_target.copy()
                results = self.reader.readtext(frame_to_process, allowlist=self.allowed_chars)
                self.result_ready.emit(results)
                
            time.sleep(1.0) 

    def update_target(self, img):
        self.current_target = img

    def stop(self):
        self.is_running = False
# ==========================================

class OCRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced EasyOCR Numeric Scanner")
        self.setGeometry(100, 100, 1100, 750)

        print("Loading EasyOCR Model...")
        self.reader = easyocr.Reader(['en'])
        
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.roi = None  
        
        self.look_alikes = {
            'O':'0', 'o':'0', 'l':'1', 'I':'1', 'i':'1', 
            'S':'5', 's':'5', 'Z':'2', 'z':'2', 'B':'8', 
            'b':'8', 'G':'6', 'g':'9', 'q':'9', 'A':'4'
        }
        
        allowed_chars = '0123456789.' + ''.join(self.look_alikes.keys())
        self.ocr_worker = OCRWorker(self.reader, allowed_chars)
        self.ocr_worker.result_ready.connect(self.handle_auto_results)
        
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        
        # --- Left Column (Video/Image Display) ---
        left_layout = QVBoxLayout()
        
        self.image_label = QLabel("Feed will appear here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000; color: #fff; border: 1px solid gray;")
        self.image_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.image_label)
        
        # Preprocessing Controls
        control_layout = QHBoxLayout()
        
        self.otsu_checkbox = QCheckBox("Auto Otsu")
        self.otsu_checkbox.setChecked(False)
        self.otsu_checkbox.stateChanged.connect(self.update_ui_state)
        control_layout.addWidget(self.otsu_checkbox)
        
        self.thresh_label = QLabel("Thresh: 127")
        control_layout.addWidget(self.thresh_label)
        
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setMinimum(0)
        self.thresh_slider.setMaximum(255)
        self.thresh_slider.setValue(127)
        self.thresh_slider.valueChanged.connect(self.update_ui_state)
        control_layout.addWidget(self.thresh_slider)

        self.dilate_label = QLabel("Dilation: 0")
        control_layout.addWidget(self.dilate_label)
        
        self.dilate_slider = QSlider(Qt.Horizontal)
        self.dilate_slider.setMinimum(0)
        self.dilate_slider.setMaximum(10) 
        self.dilate_slider.setValue(0)
        self.dilate_slider.valueChanged.connect(self.update_ui_state)
        control_layout.addWidget(self.dilate_slider)
        
        left_layout.addLayout(control_layout)
        main_layout.addLayout(left_layout, stretch=2)
        
        # --- Right Column (Controls & Output) ---
        right_layout = QVBoxLayout()
        
        self.btn_load_img = QPushButton("1. Upload Image")
        self.btn_load_img.clicked.connect(self.load_image)
        right_layout.addWidget(self.btn_load_img)
        
        self.btn_load_vid = QPushButton("2. Upload Video")
        self.btn_load_vid.clicked.connect(self.load_video)
        right_layout.addWidget(self.btn_load_vid)
        
        # --- NEW: Camera Selection Layout ---
        cam_layout = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.cam_combo.addItems([
            "Cam 0 (Default)", 
            "Cam 1 (USB)", 
            "Cam 2 (USB)", 
            "Cam 3 (USB)"
        ])
        cam_layout.addWidget(self.cam_combo)

        self.btn_start_cam = QPushButton("3. Live Camera")
        self.btn_start_cam.clicked.connect(self.start_camera)
        cam_layout.addWidget(self.btn_start_cam)
        
        right_layout.addLayout(cam_layout)
        # ------------------------------------

        self.btn_select_roi = QPushButton("4. Select ROI")
        self.btn_select_roi.clicked.connect(self.select_roi)
        right_layout.addWidget(self.btn_select_roi)
        
        self.btn_clear_roi = QPushButton("Clear ROI")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        right_layout.addWidget(self.btn_clear_roi)

        right_layout.addWidget(QLabel("--- OCR Settings ---"))
        self.conf_label = QLabel("Min Confidence: 15%")
        right_layout.addWidget(self.conf_label)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(15)
        self.conf_slider.valueChanged.connect(self.update_ui_state)
        right_layout.addWidget(self.conf_slider)

        # --- Action Buttons ---
        action_layout = QHBoxLayout()
        
        self.btn_run_ocr = QPushButton("RUN OCR\n(Single)")
        self.btn_run_ocr.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_run_ocr.clicked.connect(self.run_ocr)
        action_layout.addWidget(self.btn_run_ocr)
        
        self.btn_auto_update = QPushButton("Auto-Capture\n(Update)")
        self.btn_auto_update.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_auto_update.clicked.connect(self.start_auto_ocr)
        action_layout.addWidget(self.btn_auto_update)

        self.btn_stop_auto = QPushButton("Stop\nAuto")
        self.btn_stop_auto.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.btn_stop_auto.setEnabled(False)
        self.btn_stop_auto.clicked.connect(self.stop_auto_ocr)
        action_layout.addWidget(self.btn_stop_auto)

        right_layout.addLayout(action_layout)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("OCR Results will appear here...")
        right_layout.addWidget(self.result_text)
        
        main_layout.addLayout(right_layout, stretch=1)
        main_widget.setLayout(main_layout)

    # --- UI Logic ---
    def update_ui_state(self):
        self.thresh_slider.setEnabled(not self.otsu_checkbox.isChecked())
        self.thresh_label.setText(f"Thresh: {self.thresh_slider.value()}")
        self.dilate_label.setText(f"Dilation: {self.dilate_slider.value()}")
        self.conf_label.setText(f"Min Confidence: {self.conf_slider.value()}%")
        
        if not self.timer.isActive() and self.current_frame is not None:
            self.display_frame(self.current_frame)

    def clear_roi(self):
        self.roi = None
        if not self.timer.isActive() and self.current_frame is not None:
            self.display_frame(self.current_frame)

    # --- Media Loading ---
    def load_image(self):
        self.stop_media()
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg)")
        if fname:
            self.current_frame = cv2.imread(fname)
            self.roi = None
            self.display_frame(self.current_frame)

    def load_video(self):
        self.stop_media()
        fname, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if fname:
            self.capture = cv2.VideoCapture(fname)
            self.roi = None
            self.timer.start(30)

    def start_camera(self):
        self.stop_media()
        
        # --- NEW: Get index from dropdown ---
        cam_idx = self.cam_combo.currentIndex()
        self.capture = cv2.VideoCapture(cam_idx)
        
        # Check if the camera actually opened
        if not self.capture.isOpened():
            QMessageBox.warning(self, "Camera Error", f"Could not connect to Camera {cam_idx}. Make sure it is plugged in and not being used by another app.")
            self.capture = None
            return
            
        self.roi = None
        self.timer.start(30)

    def stop_media(self):
        self.timer.stop()
        self.stop_auto_ocr()
        if self.capture:
            self.capture.release()
            self.capture = None

    # --- Video Processing ---
    def update_frame(self):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
                
                if self.ocr_worker.is_running:
                    if self.roi:
                        x, y, w, h = self.roi
                        target_img = frame[y:y+h, x:x+w]
                    else:
                        target_img = frame
                    processed_img = self.apply_preprocessing(target_img)
                    self.ocr_worker.update_target(processed_img)
            else:
                self.stop_media()

    def apply_preprocessing(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        if self.otsu_checkbox.isChecked():
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            val = self.thresh_slider.value()
            _, thresh = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
            
        dilate_val = self.dilate_slider.value()
        if dilate_val > 0:
            kernel = np.ones((dilate_val, dilate_val), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            
        return thresh

    def display_frame(self, frame):
        processed = self.apply_preprocessing(frame)
        display_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        q_img = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    # --- ROI Selection ---
    def select_roi(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Error", "Load an image or start video first!")
            return
            
        was_playing = self.timer.isActive()
        if was_playing:
            self.timer.stop()

        h, w = self.current_frame.shape[:2]
        max_height = 800
        max_width = 1200
        scale = 1.0
        
        if h > max_height or w > max_width:
            scale = min(max_width / w, max_height / h)
            display_frame = cv2.resize(self.current_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            display_frame = self.current_frame.copy()

        QMessageBox.information(self, "Instructions", "Draw a rectangle and press ENTER or SPACE to confirm.")
        
        roi_window_name = "Select ROI (Press ENTER to confirm)"
        cv2.namedWindow(roi_window_name, cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(roi_window_name, display_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(roi_window_name)
        
        if roi[2] > 0 and roi[3] > 0: 
            self.roi = (
                int(roi[0] / scale), 
                int(roi[1] / scale), 
                int(roi[2] / scale), 
                int(roi[3] / scale)
            )
            
        if was_playing:
            self.timer.start(30)
        else:
            self.display_frame(self.current_frame)

    # --- Text Cleaning ---
    def clean_numeric_text(self, raw_text):
        cleaned = ""
        for char in raw_text:
            if char in self.look_alikes:
                cleaned += self.look_alikes[char]
            elif char.isdigit() or char == '.': 
                cleaned += char
        return cleaned

    # --- AUTO OCR LOGIC (Continuous) ---
    def start_auto_ocr(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Error", "Start a camera or video feed first.")
            return
        
        self.btn_auto_update.setEnabled(False)
        self.btn_run_ocr.setEnabled(False)
        self.btn_stop_auto.setEnabled(True)
        self.result_text.append("--- Auto-Capture Started ---")
        self.ocr_worker.start()

    def stop_auto_ocr(self):
        if self.ocr_worker.is_running:
            self.ocr_worker.stop()
            self.ocr_worker.wait() 
            self.btn_auto_update.setEnabled(True)
            self.btn_run_ocr.setEnabled(True)
            self.btn_stop_auto.setEnabled(False)
            self.result_text.append("--- Auto-Capture Stopped ---")

    def handle_auto_results(self, results):
        min_conf = self.conf_slider.value() / 100.0 
        found_any = False
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        output_lines = []
        for (bbox, text, prob) in results:
            if prob >= min_conf:
                final_text = self.clean_numeric_text(text)
                if final_text: 
                    output_lines.append(f"[{timestamp}] Auto-Scan: {final_text} (Conf: {prob:.2f})")
                    found_any = True
                    
        if found_any:
            for line in output_lines:
                self.result_text.append(line)
            self.result_text.verticalScrollBar().setValue(self.result_text.verticalScrollBar().maximum())

    # --- MANUAL OCR LOGIC (Single) ---
    def run_ocr(self):
        if self.current_frame is None:
            return

        if self.roi:
            x, y, w, h = self.roi
            target_img = self.current_frame[y:y+h, x:x+w]
        else:
            target_img = self.current_frame
            
        processed_img = self.apply_preprocessing(target_img)
        
        self.result_text.append("Scanning...")
        QApplication.processEvents()

        allowed_chars = '0123456789.' + ''.join(self.look_alikes.keys())
        results = self.reader.readtext(processed_img, allowlist=allowed_chars)
        
        min_conf = self.conf_slider.value() / 100.0 
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.result_text.append(f"--- Manual Results [{timestamp}] ---")
        found_any = False
        
        for (bbox, text, prob) in results:
            if prob >= min_conf:
                final_text = self.clean_numeric_text(text)
                if final_text: 
                    self.result_text.append(f"[{timestamp}] Found: {final_text} (Conf: {prob:.2f})")
                    found_any = True
                    
        if not found_any:
            self.result_text.append(f"No numerics found above {int(min_conf*100)}% confidence.")
            
        self.result_text.append("-" * 20)
        self.result_text.verticalScrollBar().setValue(self.result_text.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self.stop_media()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OCRApp()
    window.show()
    sys.exit(app.exec_())
