import sys
import cv2
import numpy as np
import easyocr
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSlider, QCheckBox, QTextEdit, QMessageBox, QComboBox,
                             QListWidget, QAbstractItemView, QListWidgetItem, QInputDialog,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QPolygonF, QPainter


# ==========================================
# HELPER: Crop Image based on ROI shape
# ==========================================
def crop_roi(frame, roi):
    """Crops a specific region out of the frame based on Rectangle or Polygon points."""
    if roi['type'] == 'rect':
        x, y, w, h = roi['points']
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w <= 0 or h <= 0:
            return None
        return frame[y:y + h, x:x + w]

    elif roi['type'] == 'poly':
        pts = np.array(roi['points'], np.int32)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        x, y, w, h = cv2.boundingRect(pts)
        if w <= 0 or h <= 0:
            return None
        return masked_frame[y:y + h, x:x + w]


def prepare_crop(img_crop):
    """Shared crop preparation: upscale tiny crops, downscale huge ones, pad edges.

    Small digits are a common cause of misreads, so we now upscale crops that are
    too small instead of only shrinking the big ones.
    """
    if img_crop is None or img_crop.size == 0:
        return None
    h, w = img_crop.shape[:2]
    target_max, target_min = 1000, 320
    if w > target_max:
        scale = target_max / w
        img_crop = cv2.resize(img_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif w < target_min:
        scale = target_min / w
        img_crop = cv2.resize(img_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(img_crop, 25, 25, 25, 25, cv2.BORDER_REPLICATE)


# ==========================================
# 1. CUSTOM GRAPHICS VIEW FOR ZOOM/PAN/DRAWING
# ==========================================
class ROICanvas(QGraphicsView):
    roi_finished = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)

        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.drawing_mode = None
        self.start_point = None
        self.current_shape = None
        self.poly_points = []

    def set_image(self, qpixmap):
        self.image_item.setPixmap(qpixmap)
        self.scene.setSceneRect(self.image_item.boundingRect())

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MidButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)
            return

        if self.drawing_mode is None:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())

        if event.button() == Qt.LeftButton:
            if self.drawing_mode == 'rect':
                self.start_point = scene_pos
                self.current_shape = self.scene.addRect(
                    scene_pos.x(), scene_pos.y(), 0, 0, QPen(QColor(0, 255, 0), 2)
                )
            elif self.drawing_mode == 'poly':
                self.poly_points.append(scene_pos)
                if not self.current_shape:
                    self.current_shape = self.scene.addPolygon(
                        QPolygonF(self.poly_points), QPen(QColor(0, 255, 0), 2)
                    )
                else:
                    self.current_shape.setPolygon(QPolygonF(self.poly_points))

        elif event.button() == Qt.RightButton and self.drawing_mode == 'poly':
            if len(self.poly_points) > 2:
                points_list = [(int(p.x()), int(p.y())) for p in self.poly_points]
                self.roi_finished.emit({'type': 'poly', 'points': points_list})

            if self.current_shape:
                self.scene.removeItem(self.current_shape)
            self.current_shape = None
            self.poly_points = []
            self.drawing_mode = None

    def mouseMoveEvent(self, event):
        if self.drawing_mode == 'rect' and self.start_point and self.current_shape:
            scene_pos = self.mapToScene(event.pos())
            x = min(self.start_point.x(), scene_pos.x())
            y = min(self.start_point.y(), scene_pos.y())
            w = abs(self.start_point.x() - scene_pos.x())
            h = abs(self.start_point.y() - scene_pos.y())
            self.current_shape.setRect(x, y, w, h)

        elif self.drawing_mode == 'poly' and self.current_shape:
            scene_pos = self.mapToScene(event.pos())
            temp_poly = QPolygonF(self.poly_points + [scene_pos])
            self.current_shape.setPolygon(temp_poly)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MidButton:
            self.setDragMode(QGraphicsView.NoDrag)

        elif event.button() == Qt.LeftButton and self.drawing_mode == 'rect':
            if self.start_point and self.current_shape:
                rect = self.current_shape.rect()
                if rect.width() > 5 and rect.height() > 5:
                    self.roi_finished.emit({
                        'type': 'rect',
                        'points': [int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())]
                    })
                self.scene.removeItem(self.current_shape)
                self.current_shape = None
                self.start_point = None
                self.drawing_mode = None

        super().mouseReleaseEvent(event)


# ==========================================
# 2. BACKGROUND WORKER (persistent: serves BOTH single-run and auto)
# ==========================================
class OCRWorker(QThread):
    result_ready = pyqtSignal(list)

    def __init__(self, reader, allowed_chars):
        super().__init__()
        self.reader = reader
        self.allowed_chars = allowed_chars
        self.is_running = False
        self.current_target = None
        self.current_rois = []
        self.new_frame_available = False

    def run(self):
        self.is_running = True
        while self.is_running:
            if self.new_frame_available and self.current_target is not None:
                self.new_frame_available = False
                base_frame = self.current_target.copy()
                rois_to_process = list(self.current_rois)

                all_results = []

                if not rois_to_process:
                    rois_to_process = [{'name': 'Full Image', 'type': 'rect',
                                        'points': [0, 0, base_frame.shape[1], base_frame.shape[0]]}]

                for roi in rois_to_process:
                    img_crop = prepare_crop(crop_roi(base_frame, roi))
                    if img_crop is None:
                        continue

                    results = self.reader.readtext(
                        img_crop, allowlist=self.allowed_chars,
                        mag_ratio=2.0, text_threshold=0.4, link_threshold=0.3, min_size=10
                    )

                    for (bbox, text, prob) in results:
                        all_results.append((roi['name'], text, prob))

                self.result_ready.emit(all_results)
            else:
                time.sleep(0.03)

    def update_target(self, img, rois):
        self.current_rois = rois          # set rois first so the worker never reads
        self.current_target = img         # a new frame against stale rois
        self.new_frame_available = True

    def stop(self):
        self.is_running = False


# ==========================================
# 3. MAIN UI APP
# ==========================================
class OCRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-ROI OCR Scanner (Fixed)")
        self.setGeometry(100, 100, 1400, 800)

        print("Loading EasyOCR Model...")
        self.reader = easyocr.Reader(['en'])

        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.roi_counter = 1
        self.auto_mode = False

        # Glyphs EasyOCR may emit for a digit. We ALLOW them so the model can
        # commit to its natural reading, then remap them back to digits in clean_text.
        self.look_alikes = {
            'O': '0', 'o': '0', 'D': '0', 'Q': '0',
            'l': '1', 'I': '1', 'i': '1', '|': '1', '!': '1', ']': '1', '[': '1', 't': '1', 'T': '1',
            'S': '5', 's': '5', 'Z': '2', 'z': '2', 'B': '8', 'b': '8',
            'G': '6', 'g': '9', 'q': '9', 'A': '4', 'F': '7', 'V': '7', 'v': '7', 'y': '7', '?': '7'
        }
        self.allowed_chars = '0123456789.' + ''.join(self.look_alikes.keys())

        self.ocr_worker = OCRWorker(self.reader, self.allowed_chars)
        self.ocr_worker.result_ready.connect(self.process_ocr_output)
        self.ocr_worker.start()  # persistent: idle-sleeps until a frame is pushed

        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()

        # --- Left Column (Video/Image Display) ---
        left_layout = QVBoxLayout()

        self.canvas = ROICanvas()
        self.canvas.roi_finished.connect(self.add_roi_to_list)
        left_layout.addWidget(self.canvas)

        # Preprocessing Controls (row 1: mode + invert + clahe)
        pre_row1 = QHBoxLayout()
        pre_row1.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Grayscale (no threshold)", "Otsu", "Adaptive", "Manual"])
        self.mode_combo.currentIndexChanged.connect(self.update_ui_state)
        pre_row1.addWidget(self.mode_combo)

        self.invert_checkbox = QCheckBox("Invert (light text on dark bg)")
        self.invert_checkbox.stateChanged.connect(self.update_ui_state)
        pre_row1.addWidget(self.invert_checkbox)

        self.clahe_checkbox = QCheckBox("CLAHE contrast")
        self.clahe_checkbox.stateChanged.connect(self.update_ui_state)
        pre_row1.addWidget(self.clahe_checkbox)
        left_layout.addLayout(pre_row1)

        # Preprocessing Controls (row 2: thresh + dilation sliders)
        pre_row2 = QHBoxLayout()
        self.thresh_label = QLabel("Thresh: 127")
        pre_row2.addWidget(self.thresh_label)
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 255)
        self.thresh_slider.setValue(127)
        self.thresh_slider.valueChanged.connect(self.update_ui_state)
        pre_row2.addWidget(self.thresh_slider)

        self.dilate_label = QLabel("Dilation: 0")
        pre_row2.addWidget(self.dilate_label)
        self.dilate_slider = QSlider(Qt.Horizontal)
        self.dilate_slider.setRange(0, 10)
        self.dilate_slider.setValue(0)
        self.dilate_slider.valueChanged.connect(self.update_ui_state)
        pre_row2.addWidget(self.dilate_slider)
        left_layout.addLayout(pre_row2)

        main_layout.addLayout(left_layout, stretch=3)

        # --- Right Column (Controls & ROI Manager) ---
        right_layout = QVBoxLayout()

        draw_layout = QHBoxLayout()
        self.btn_draw_rect = QPushButton("Draw Box")
        self.btn_draw_rect.clicked.connect(lambda: self.set_draw_mode('rect'))
        self.btn_draw_poly = QPushButton("Draw Polygon")
        self.btn_draw_poly.clicked.connect(lambda: self.set_draw_mode('poly'))
        draw_layout.addWidget(self.btn_draw_rect)
        draw_layout.addWidget(self.btn_draw_poly)
        right_layout.addLayout(draw_layout)

        right_layout.addWidget(QLabel("<b>ROI Manager (Drag to Reorder, Dbl-Click to Rename)</b>"))
        self.roi_list = QListWidget()
        self.roi_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.roi_list.itemDoubleClicked.connect(self.rename_roi)
        right_layout.addWidget(self.roi_list)

        roi_action_layout = QHBoxLayout()
        self.btn_delete_roi = QPushButton("Delete Selected")
        self.btn_delete_roi.clicked.connect(self.delete_roi)
        self.btn_refresh_roi = QPushButton("Renumber ROIs")
        self.btn_refresh_roi.clicked.connect(self.refresh_roi_names)
        roi_action_layout.addWidget(self.btn_delete_roi)
        roi_action_layout.addWidget(self.btn_refresh_roi)
        right_layout.addLayout(roi_action_layout)

        right_layout.addWidget(QLabel("<hr>"))
        self.btn_load_img = QPushButton("Load Image")
        self.btn_load_img.clicked.connect(self.load_image)
        right_layout.addWidget(self.btn_load_img)
        self.btn_load_vid = QPushButton("Load Video")
        self.btn_load_vid.clicked.connect(self.load_video)
        right_layout.addWidget(self.btn_load_vid)

        cam_layout = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(["Cam 0", "Cam 1", "Cam 2"])
        self.btn_start_cam = QPushButton("Live Camera")
        self.btn_start_cam.clicked.connect(self.start_camera)
        cam_layout.addWidget(self.cam_combo)
        cam_layout.addWidget(self.btn_start_cam)
        right_layout.addLayout(cam_layout)

        right_layout.addWidget(QLabel("<hr><b>OCR Settings</b>"))
        self.conf_label = QLabel("Min Confidence: 15%")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(15)
        self.conf_slider.valueChanged.connect(self.update_ui_state)
        right_layout.addWidget(self.conf_label)
        right_layout.addWidget(self.conf_slider)

        self.detail_checkbox = QCheckBox("Show per-ROI breakdown")
        right_layout.addWidget(self.detail_checkbox)

        action_layout = QHBoxLayout()
        self.btn_auto_update = QPushButton("Start Auto-OCR")
        self.btn_auto_update.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_auto_update.clicked.connect(self.start_auto_ocr)

        self.btn_stop_auto = QPushButton("Stop Auto")
        self.btn_stop_auto.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.btn_stop_auto.setEnabled(False)
        self.btn_stop_auto.clicked.connect(self.stop_auto_ocr)

        action_layout.addWidget(self.btn_auto_update)
        action_layout.addWidget(self.btn_stop_auto)
        right_layout.addLayout(action_layout)

        self.btn_single_run = QPushButton("Capture Once (Single Run)")
        self.btn_single_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_single_run.clicked.connect(self.run_single_ocr)
        right_layout.addWidget(self.btn_single_run)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        right_layout.addWidget(self.result_text)

        main_layout.addLayout(right_layout, stretch=1)
        main_widget.setLayout(main_layout)

        self.update_ui_state()

    # --- Drawing & ROI Logic ---
    def set_draw_mode(self, mode):
        self.canvas.drawing_mode = mode
        msg = "Click & Drag to draw Box." if mode == 'rect' else "Click points to draw Polygon. Right-Click to finish."
        self.result_text.append(f"<i>{msg}</i>")

    def add_roi_to_list(self, roi_data):
        roi_name = f"ROI {self.roi_counter}"
        self.roi_counter += 1
        roi_data['name'] = roi_name

        item = QListWidgetItem(f"{roi_name} ({roi_data['type']})")
        item.setData(Qt.UserRole, roi_data)
        self.roi_list.addItem(item)
        self.update_ui_state()

    def rename_roi(self, item):
        old_name = item.data(Qt.UserRole)['name']
        new_name, ok = QInputDialog.getText(self, "Rename ROI", "Enter new name:", text=old_name)
        if ok and new_name:
            roi_data = item.data(Qt.UserRole)
            roi_data['name'] = new_name
            item.setData(Qt.UserRole, roi_data)
            item.setText(f"{new_name} ({roi_data['type']})")
            self.update_ui_state()

    def refresh_roi_names(self):
        for i in range(self.roi_list.count()):
            item = self.roi_list.item(i)
            roi_data = item.data(Qt.UserRole)
            new_name = f"ROI {i + 1}"
            roi_data['name'] = new_name
            item.setData(Qt.UserRole, roi_data)
            item.setText(f"{new_name} ({roi_data['type']})")
        self.roi_counter = self.roi_list.count() + 1
        self.update_ui_state()

    def delete_roi(self):
        for item in self.roi_list.selectedItems():
            self.roi_list.takeItem(self.roi_list.row(item))
        self.update_ui_state()

    def get_all_rois(self):
        rois = []
        for i in range(self.roi_list.count()):
            rois.append(self.roi_list.item(i).data(Qt.UserRole))
        return rois

    # --- Video & Preprocessing Logic ---
    def update_ui_state(self):
        manual = self.mode_combo.currentText() == "Manual"
        self.thresh_slider.setEnabled(manual)
        self.thresh_label.setText(f"Thresh: {self.thresh_slider.value()}")
        self.dilate_label.setText(f"Dilation: {self.dilate_slider.value()}")
        self.conf_label.setText(f"Min Confidence: {self.conf_slider.value()}%")

        if not self.timer.isActive() and self.current_frame is not None:
            self.display_frame(self.current_frame)
            # If auto-OCR is on for a STILL image, re-scan as the user tunes settings.
            if self.auto_mode:
                self.push_frame_to_worker()

    def load_image(self):
        self.stop_media()
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.jpg *.png *.jpeg *.bmp)")
        if fname:
            self.current_frame = cv2.imread(fname)
            self.display_frame(self.current_frame)

    def load_video(self):
        self.stop_media()
        fname, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov)")
        if fname:
            self.capture = cv2.VideoCapture(fname)
            self.timer.start(30)

    def start_camera(self):
        self.stop_media()
        self.capture = cv2.VideoCapture(self.cam_combo.currentIndex())
        if self.capture.isOpened():
            self.timer.start(30)
        else:
            QMessageBox.warning(self, "Camera", "Could not open the selected camera.")

    def stop_media(self):
        self.timer.stop()
        self.stop_auto_ocr()
        if self.capture:
            self.capture.release()
            self.capture = None

    def update_frame(self):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
                if self.auto_mode:
                    self.push_frame_to_worker()
            else:
                self.stop_media()

    def apply_preprocessing(self, frame):
        """Builds the 3-channel image that EasyOCR actually reads.

        Supports raw grayscale (often the most accurate for printed digits),
        Otsu, adaptive, or manual thresholding, plus optional CLAHE and invert.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.clahe_checkbox.isChecked():
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        mode = self.mode_combo.currentText()
        if mode == "Grayscale (no threshold)":
            proc = gray
        elif mode == "Otsu":
            _, proc = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif mode == "Adaptive":
            proc = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 5)
        else:  # Manual
            _, proc = cv2.threshold(gray, self.thresh_slider.value(), 255, cv2.THRESH_BINARY)

        if self.invert_checkbox.isChecked():
            proc = cv2.bitwise_not(proc)

        d_val = self.dilate_slider.value()
        if d_val > 0:
            proc = cv2.dilate(proc, np.ones((d_val, d_val), np.uint8), iterations=1)

        # EasyOCR performs significantly better on 3-channel images.
        return cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)

    def display_frame(self, frame):
        if frame is None:
            return
        processed_bgr = self.apply_preprocessing(frame)
        display_img = processed_bgr.copy()

        for roi in self.get_all_rois():
            if roi['type'] == 'rect':
                x, y, w, h = roi['points']
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_img, roi['name'], (x, max(10, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif roi['type'] == 'poly':
                pts = np.array(roi['points'], np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(display_img, roi['name'], (pts[0][0][0], max(10, pts[0][0][1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        h, w, ch = display_img.shape
        q_img = QImage(display_img.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.canvas.set_image(QPixmap.fromImage(q_img))

    # --- OCR text cleanup ---
    def clean_text(self, text):
        """Maps look-alike glyphs to digits FIRST, then keeps digits and dots.

        The previous version filtered on the original char, so any digit read as a
        letter (O, l, S, B...) was dropped before it could be remapped. This is the
        main fix for number misinterpretation.
        """
        out = []
        for c in text:
            mapped = self.look_alikes.get(c, c)
            if mapped.isdigit() or mapped == '.':
                out.append(mapped)
        result = "".join(out)

        # Collapse accidental multiple dots (".." -> ".") and strip stray edge dots.
        while '..' in result:
            result = result.replace('..', '.')
        return result.strip('.')

    # --- OCR triggering (single + auto both go through the worker) ---
    def push_frame_to_worker(self):
        if self.current_frame is not None:
            processed = self.apply_preprocessing(self.current_frame)
            self.ocr_worker.update_target(processed, self.get_all_rois())

    def run_single_ocr(self):
        """Non-blocking single capture. Hands the frame to the worker thread."""
        if self.current_frame is None:
            return
        self.result_text.append(f"<i>Scanning {len(self.get_all_rois())} ROI(s)...</i>")
        self.push_frame_to_worker()

    def start_auto_ocr(self):
        if self.current_frame is None:
            return
        self.auto_mode = True
        self.btn_auto_update.setEnabled(False)
        self.btn_stop_auto.setEnabled(True)
        # Push immediately so STILL images scan too (timer isn't running for them).
        self.push_frame_to_worker()

    def stop_auto_ocr(self):
        self.auto_mode = False
        self.btn_auto_update.setEnabled(True)
        self.btn_stop_auto.setEnabled(False)

    def process_ocr_output(self, results):
        """Combines results in ROI order, with optional per-ROI breakdown."""
        min_conf = self.conf_slider.value() / 100.0
        ts = datetime.now().strftime("%H:%M:%S")

        combined_text = ""
        per_roi = {}  # preserves insertion order (ROI list order)

        for (roi_name, text, prob) in results:
            if prob >= min_conf:
                clean = self.clean_text(text)
                if clean:
                    combined_text += clean
                    per_roi.setdefault(roi_name, []).append((clean, prob))

        if combined_text:
            self.result_text.append(f"[{ts}] Result: <b>{combined_text}</b>")
            if self.detail_checkbox.isChecked():
                for name, vals in per_roi.items():
                    detail = ", ".join(f"{v} ({p * 100:.0f}%)" for v, p in vals)
                    self.result_text.append(f"&nbsp;&nbsp;&nbsp;&nbsp;{name}: {detail}")
            self.result_text.verticalScrollBar().setValue(
                self.result_text.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self.ocr_worker.stop()
        self.ocr_worker.wait()
        if self.capture:
            self.capture.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OCRApp()
    window.show()
    sys.exit(app.exec_())
