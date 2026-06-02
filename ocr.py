import sys
import cv2
import numpy as np
import easyocr
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QCheckBox, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class OCRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced EasyOCR Numeric Scanner")
        self.setGeometry(100, 100, 1000, 700)

        # Initialize EasyOCR (loads model into memory)
        print("Loading EasyOCR Model...")
        self.reader = easyocr.Reader(['en'])
        
        # State Variables
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.roi = None  # (x, y, w, h)
        
        # Look-alike mapping dictionary
        self.look_alikes = {
            'O':'0', 'o':'0', 'l':'1', 'I':'1', 'i':'1', 
            'S':'5', 's':'5', 'Z':'2', 'z':'2', 'B':'8', 
            'b':'8', 'G':'6', 'g':'9', 'q':'9', 'A':'4'
        }
        
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
        
        self.otsu_checkbox = QCheckBox("Use Auto Otsu Threshold")
        self.otsu_checkbox.setChecked(False)
        self.otsu_checkbox.stateChanged.connect(self.toggle_otsu)
        control_layout.addWidget(self.otsu_checkbox)
        
        self.thresh_label = QLabel("Manual Threshold: 127")
        control_layout.addWidget(self.thresh_label)
        
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setMinimum(0)
        self.thresh_slider.setMaximum(255)
        self.thresh_slider.setValue(127)
        self.thresh_slider.valueChanged.connect(self.update_slider_label)
        control_layout.addWidget(self.thresh_slider)
        
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
        
        self.btn_start_cam = QPushButton("3. Live Camera")
        self.btn_start_cam.clicked.connect(self.start_camera)
        right_layout.addWidget(self.btn_start_cam)
        
        self.btn_select_roi = QPushButton("4. Select ROI")
        self.btn_select_roi.clicked.connect(self.select_roi)
        right_layout.addWidget(self.btn_select_roi)
        
        self.btn_clear_roi = QPushButton("Clear ROI")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        right_layout.addWidget(self.btn_clear_roi)

        self.btn_run_ocr = QPushButton("5. RUN OCR (Numerics Only)")
        self.btn_run_ocr.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_run_ocr.clicked.connect(self.run_ocr)
        right_layout.addWidget(self.btn_run_ocr)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("OCR Results will appear here...")
        right_layout.addWidget(self.result_text)
        
        main_layout.addLayout(right_layout, stretch=1)
        main_widget.setLayout(main_layout)

    # --- UI Logic ---
    def update_slider_label(self):
        self.thresh_label.setText(f"Manual Threshold: {self.thresh_slider.value()}")
        if not self.timer.isActive() and self.current_frame is not None:
            self.display_frame(self.current_frame) # Update static image immediately

    def toggle_otsu(self):
        self.thresh_slider.setEnabled(not self.otsu_checkbox.isChecked())
        if not self.timer.isActive() and self.current_frame is not None:
            self.display_frame(self.current_frame) # Update static image immediately

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
            self.timer.start(30) # ~30 fps

    def start_camera(self):
        self.stop_media()
        self.capture = cv2.VideoCapture(0) # 0 is default webcam
        self.roi = None
        self.timer.start(30)

    def stop_media(self):
        self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None

    # --- Video & Image Processing ---
    def update_frame(self):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
            else:
                self.stop_media() # End of video

    def apply_preprocessing(self, frame):
        """Applies Grayscale and Thresholding based on UI controls"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.otsu_checkbox.isChecked():
            # Apply Otsu's Threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Apply Manual Threshold
            val = self.thresh_slider.value()
            _, thresh = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
            
        return thresh

    def display_frame(self, frame):
        # 1. Apply preprocessing to see what the OCR will see
        processed = self.apply_preprocessing(frame)
        
        # 2. Convert back to BGR for drawing colored ROI boxes
        display_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # 3. Draw ROI if it exists
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # 4. Convert to PyQt format
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
            self.timer.stop() # Pause feed while selecting

        # Use OpenCV's native ROI selector (pops up a temporary window)
        QMessageBox.information(self, "Instructions", "A window will open. Draw a rectangle and press ENTER or SPACE to confirm.")
        roi = cv2.selectROI("Select ROI (Press ENTER to confirm)", self.current_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI (Press ENTER to confirm)")
        
        if roi[2] > 0 and roi[3] > 0: # Ensure valid width/height
            self.roi = roi
            
        if was_playing:
            self.timer.start(30)
        else:
            self.display_frame(self.current_frame)

    # --- OCR Execution ---
    def clean_numeric_text(self, raw_text):
        """Replaces look-alikes with numbers and strips standard alphabets"""
        cleaned = ""
        for char in raw_text:
            if char in self.look_alikes:
                cleaned += self.look_alikes[char]
            elif char.isdigit():
                cleaned += char
        return cleaned

    def run_ocr(self):
        if self.current_frame is None:
            return

        # 1. Get the image area to process
        if self.roi:
            x, y, w, h = self.roi
            target_img = self.current_frame[y:y+h, x:x+w]
        else:
            target_img = self.current_frame
            
        # 2. Apply the exact same preprocessing we show on screen
        processed_img = self.apply_preprocessing(target_img)
        
        self.result_text.append("Scanning...")
        QApplication.processEvents() # Force UI update

        # 3. Run EasyOCR 
        # We pass an allowlist that includes digits AND our look-alikes to prevent it from guessing random symbols
        allowed_chars = '0123456789' + ''.join(self.look_alikes.keys())
        results = self.reader.readtext(processed_img, allowlist=allowed_chars)
        
        # 4. Parse and clean results
        self.result_text.append("--- OCR Results ---")
        for (bbox, text, prob) in results:
            final_text = self.clean_numeric_text(text)
            if final_text: # Only print if there's an actual number left
                self.result_text.append(f"Found: {final_text} (Confidence: {prob:.2f})")
        self.result_text.append("-" * 20)

    def closeEvent(self, event):
        self.stop_media()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OCRApp()
    window.show()
    sys.exit(app.exec_())