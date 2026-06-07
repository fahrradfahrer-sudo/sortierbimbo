import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QFrame, QPushButton, QComboBox, QScrollArea, QSplitter)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette, QPainter, QPen
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import cv2
import numpy as np
import pyzed.sl as sl
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ── NMS-Patch: ersetzt inkompatible torchvision C-Extension ──────────────────
import torch as _torch

def _nms_pure_torch(boxes, scores, iou_threshold):
    """Pure PyTorch NMS – läuft ohne torchvision C-Extension."""
    if boxes.numel() == 0:
        return _torch.tensor([], dtype=_torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = x1[rest].clamp(min=float(x1[i]))
        yy1 = y1[rest].clamp(min=float(y1[i]))
        xx2 = x2[rest].clamp(max=float(x2[i]))
        yy2 = y2[rest].clamp(max=float(y2[i]))
        inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_threshold]
    return _torch.tensor(keep, dtype=_torch.long, device=boxes.device)

try:
    import torchvision.ops.boxes as _tvb
    import torchvision.ops as _tvops
    _tvb.nms = _nms_pure_torch
    _tvops.nms = _nms_pure_torch
except Exception as _e:
    print(f"NMS-Patch Warnung: {_e}")
# ─────────────────────────────────────────────────────────────────────────────

from ultralytics import YOLO

cv2.ocl.setUseOpenCL(False)


class ZedCameraHandler(QThread):
    new_image_signal = pyqtSignal(QPixmap)
    new_depth_signal = pyqtSignal(QPixmap)
    log_message_signal = pyqtSignal(str)
    camera_stopped_signal = pyqtSignal()
    camera_started_successfully_signal = pyqtSignal()
    new_detections_signal = pyqtSignal(list)

    def __init__(self, parent=None, model_path: str = ""):
        super().__init__(parent)
        self.zed = sl.Camera()
        self.running = False
        self.init_params = None
        self.model_path = model_path
        self.yolo_model = None

    def start_camera(self, zed_sdk_params):
        self.init_params = zed_sdk_params
        self.log_message_signal.emit(
            f"Camera Handler: Starting camera — "
            f"Res={self.init_params.camera_resolution}, "
            f"FPS={self.init_params.camera_fps}, "
            f"Depth={self.init_params.depth_mode}"
        )
        self.start()

    def run(self):
        self.running = True

        if not self.init_params:
            self.log_message_signal.emit("Camera Handler Error: Keine Parameter gesetzt.")
            self.camera_stopped_signal.emit()
            self.running = False
            return

        if self.zed.is_opened():
            self.zed.close()

        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.log_message_signal.emit(f"Camera Handler Error: {err}")
            if self.zed.is_opened():
                self.zed.close()
            self.running = False
            self.camera_stopped_signal.emit()
            return

        self.log_message_signal.emit("Camera Handler: Kamera geöffnet.")
        self.camera_started_successfully_signal.emit()

        if self.model_path:
            try:
                self.yolo_model = YOLO(self.model_path)
                self.log_message_signal.emit(f"Camera Handler: YOLO geladen: {self.model_path}")
            except Exception as e:
                self.log_message_signal.emit(f"Camera Handler: YOLO Fehler: {e}")
                self.yolo_model = None
        else:
            self.log_message_signal.emit("Camera Handler: Kein Modellpfad angegeben.")

        image_sl = sl.Mat()
        depth_map_sl = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        frame_count = 0

        while self.running:
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image_sl, sl.VIEW.LEFT)
                image_data_np = image_sl.get_data()

                image_for_qt = None
                detections_list = []

                if image_data_np is not None and image_data_np.size > 0:
                    if image_data_np.ndim == 3 and image_data_np.shape[2] == 4:
                        image_for_qt = image_data_np[:, :, [2, 1, 0]]
                    elif image_data_np.ndim == 3 and image_data_np.shape[2] == 3:
                        image_for_qt = image_data_np[:, :, ::-1]
                    else:
                        continue
                else:
                    continue

                if image_for_qt is not None:
                    if not image_for_qt.flags.c_contiguous:
                        image_for_qt = np.ascontiguousarray(image_for_qt)

                    # YOLO alle 3 Frames (Performance)
                    frame_count += 1
                    if self.yolo_model and frame_count % 3 == 0:
                        try:
                            pil_image = Image.fromarray(image_for_qt)
                            results = self.yolo_model.predict(
                                source=pil_image,
                                verbose=False,
                                conf=0.6,   # Höherer Schwellwert → weniger Boxen → schnellerer NMS
                                device=0    # GPU
                            )
                            if results and results[0].boxes is not None:
                                for box_data in results[0].boxes:
                                    box_normalized = box_data.xyxyn[0].tolist()
                                    confidence = float(box_data.conf[0])
                                    class_id = int(box_data.cls[0])
                                    label = self.yolo_model.names[class_id]
                                    detections_list.append({
                                        'box_xyxyn': box_normalized,
                                        'confidence': confidence,
                                        'class_id': class_id,
                                        'label': label
                                    })
                        except Exception as e:
                            self.log_message_signal.emit(f"Camera Handler: YOLO Fehler: {e}")

                    # Nur senden wenn neue Detektionen vorhanden
                    if detections_list:
                        self.new_detections_signal.emit(detections_list)

                    h, w, ch = image_for_qt.shape
                    qt_image = QImage(image_for_qt, w, h, ch * w, QImage.Format_RGB888).copy()
                    self.new_image_signal.emit(QPixmap.fromImage(qt_image))

                # Tiefenkarte
                if self.init_params.depth_mode != sl.DEPTH_MODE.NONE:
                    self.zed.retrieve_measure(depth_map_sl, sl.MEASURE.DEPTH)
                    depth_map_ocv = depth_map_sl.get_data()
                    if depth_map_ocv is not None and depth_map_ocv.size > 0:
                        depth_display = np.nan_to_num(depth_map_ocv, nan=0.0, posinf=0.0, neginf=0.0)
                        min_val, max_val = np.min(depth_display), np.max(depth_display)
                        if max_val > min_val:
                            depth_norm = ((depth_display - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                        else:
                            depth_norm = np.zeros(depth_display.shape, dtype=np.uint8)
                        if not depth_norm.flags.c_contiguous:
                            depth_norm = np.ascontiguousarray(depth_norm)
                        h_d, w_d = depth_norm.shape
                        qt_depth = QImage(depth_norm, w_d, h_d, w_d, QImage.Format_Grayscale8).copy()
                        self.new_depth_signal.emit(QPixmap.fromImage(qt_depth))
                else:
                    empty_pixmap = QPixmap(320, 180)
                    empty_pixmap.fill(Qt.black)
                    self.new_depth_signal.emit(empty_pixmap)
            else:
                self.msleep(10)

        if self.zed.is_opened():
            self.zed.close()
        self.log_message_signal.emit("Camera Handler: Kamera geschlossen.")
        self.camera_stopped_signal.emit()

    def stop(self):
        self.log_message_signal.emit("Camera Handler: Stop angefordert.")
        self.running = False


class ZedAppGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZED Camera Viewer — Kartoffelsortierer")
        self.setGeometry(100, 100, 1200, 700)

        # ZED X unterstützt nur HD1200, HD1080 — kein HD720!
        self.current_resolution_enum = sl.RESOLUTION.HD1200
        self.current_fps_val = 30
        self.current_depth_mode_enum = sl.DEPTH_MODE.NEURAL

        # ── Modellpfad hier anpassen ─────────────────────────────────────────
        self.model_path_placeholder = (
            "/home/matthias/sortierbimbo/testmodell_facenet/"
            "Face-Recognition-using-YoloV8-and-FaceNet-main/detection/weights/best.engine"
        )
        # ─────────────────────────────────────────────────────────────────────

        self.is_camera_running = False
        self.pending_settings_change = False
        self.camera_handler = None
        self.last_image_pixmap = None
        self.last_detections = []   # Letzte bekannte Detektionen (bleibt bei Timeout erhalten)

        # ── UI ───────────────────────────────────────────────────────────────
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        left_panel_frame = QFrame()
        left_panel_frame.setFrameShape(QFrame.StyledPanel)
        left_panel_frame.setMinimumWidth(250)
        left_panel_frame.setMaximumWidth(400)
        left_panel_layout = QVBoxLayout(left_panel_frame)

        title = QLabel("Settings Panel", left_panel_frame)
        title.setAlignment(Qt.AlignCenter)
        left_panel_layout.addWidget(title)

        left_panel_layout.addWidget(QLabel("Resolution:", left_panel_frame))
        self.resolution_map = {
            "HD1200": sl.RESOLUTION.HD1200,
            "HD1080": sl.RESOLUTION.HD1080,
        }
        self.resolution_combo = QComboBox(left_panel_frame)
        self.resolution_combo.addItems(self.resolution_map.keys())
        self.resolution_combo.setCurrentText("HD1200")
        left_panel_layout.addWidget(self.resolution_combo)

        left_panel_layout.addWidget(QLabel("FPS:", left_panel_frame))
        self.fps_map = {"15": 15, "30": 30, "60": 60}
        self.fps_combo = QComboBox(left_panel_frame)
        self.fps_combo.addItems(self.fps_map.keys())
        self.fps_combo.setCurrentText("30")
        left_panel_layout.addWidget(self.fps_combo)

        left_panel_layout.addWidget(QLabel("Depth Mode:", left_panel_frame))
        self.depth_mode_map = {
            "NEURAL":      sl.DEPTH_MODE.NEURAL,
            "ULTRA":       sl.DEPTH_MODE.ULTRA,
            "QUALITY":     sl.DEPTH_MODE.QUALITY,
            "NONE":        sl.DEPTH_MODE.NONE,
        }
        self.depth_mode_combo = QComboBox(left_panel_frame)
        self.depth_mode_combo.addItems(self.depth_mode_map.keys())
        self.depth_mode_combo.setCurrentText("NEURAL")
        left_panel_layout.addWidget(self.depth_mode_combo)

        self.apply_settings_button = QPushButton("Apply Settings", left_panel_frame)
        self.apply_settings_button.clicked.connect(self.apply_settings)
        left_panel_layout.addWidget(self.apply_settings_button)
        left_panel_layout.addStretch(1)

        left_panel_layout.addWidget(QLabel("Camera Logs:", left_panel_frame))
        self.log_display = QLabel("Logs will appear here.", left_panel_frame)
        self.log_display.setWordWrap(True)
        self.log_display.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log_display.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px;"
        )
        self.scroll_area = QScrollArea(left_panel_frame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.log_display)
        self.scroll_area.setFixedHeight(200)
        left_panel_layout.addWidget(self.scroll_area)

        right_panel_frame = QFrame()
        right_panel_frame.setFrameShape(QFrame.StyledPanel)

        self.image_label = QLabel("Camera Feed")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 180)
        self.image_label.setFrameShape(QFrame.Box)
        img_palette = self.image_label.palette()
        img_palette.setColor(QPalette.Window, QColor("black"))
        self.image_label.setPalette(img_palette)
        self.image_label.setAutoFillBackground(True)

        self.depth_label = QLabel("Depth Map")
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setMinimumSize(320, 180)
        self.depth_label.setFrameShape(QFrame.Box)
        depth_palette = self.depth_label.palette()
        depth_palette.setColor(QPalette.Window, QColor("black"))
        self.depth_label.setPalette(depth_palette)
        self.depth_label.setAutoFillBackground(True)

        right_v_splitter = QSplitter(Qt.Vertical)
        right_v_splitter.addWidget(self.image_label)
        right_v_splitter.addWidget(self.depth_label)
        right_v_splitter.setSizes([400, 200])

        right_panel_layout = QVBoxLayout(right_panel_frame)
        right_panel_layout.addWidget(right_v_splitter)
        right_panel_frame.setLayout(right_panel_layout)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_panel_frame)
        main_splitter.addWidget(right_panel_frame)
        main_splitter.setSizes([300, 850])

        new_main_layout = QHBoxLayout(main_widget)
        new_main_layout.addWidget(main_splitter)
        main_widget.setLayout(new_main_layout)

        # ── Start ─────────────────────────────────────────────────────────────
        self.log_message("GUI: Kartoffelsortierer gestartet.")
        self.log_message(f"GUI: Modell: {self.model_path_placeholder}")
        self.show()
        self.start_camera_with_current_settings()

    # ── Kamerasteuerung ───────────────────────────────────────────────────────

    def apply_settings(self):
        if self.is_camera_running and self.camera_handler:
            self.apply_settings_button.setEnabled(False)
            self.pending_settings_change = True
            self.camera_handler.stop()
        else:
            self.pending_settings_change = False
            self.start_camera_with_current_settings()

    def start_camera_with_current_settings(self):
        self.apply_settings_button.setEnabled(False)

        selected_res_str = self.resolution_combo.currentText()
        self.current_resolution_enum = self.resolution_map.get(selected_res_str, self.current_resolution_enum)
        selected_fps_str = self.fps_combo.currentText()
        self.current_fps_val = self.fps_map.get(selected_fps_str, self.current_fps_val)
        selected_depth_str = self.depth_mode_combo.currentText()
        self.current_depth_mode_enum = self.depth_mode_map.get(selected_depth_str, self.current_depth_mode_enum)

        self.log_message(f"GUI: Res={selected_res_str}, FPS={self.current_fps_val}, Depth={selected_depth_str}")

        zed_sdk_params = sl.InitParameters()
        zed_sdk_params.camera_resolution = self.current_resolution_enum
        zed_sdk_params.camera_fps = self.current_fps_val
        zed_sdk_params.depth_mode = self.current_depth_mode_enum
        zed_sdk_params.coordinate_units = sl.UNIT.METER
        zed_sdk_params.sdk_verbose = 1

        if self.camera_handler:
            if self.camera_handler.isRunning():
                self.camera_handler.stop()
                self.camera_handler.wait(2000)
            try:
                self.camera_handler.new_image_signal.disconnect()
                self.camera_handler.new_depth_signal.disconnect()
                self.camera_handler.log_message_signal.disconnect()
                self.camera_handler.camera_stopped_signal.disconnect()
                self.camera_handler.camera_started_successfully_signal.disconnect()
                self.camera_handler.new_detections_signal.disconnect()
            except Exception:
                pass

        self.camera_handler = ZedCameraHandler(self, model_path=self.model_path_placeholder)
        self.camera_handler.new_image_signal.connect(self.update_image_label)
        self.camera_handler.new_depth_signal.connect(self.update_depth_label)
        self.camera_handler.log_message_signal.connect(self.log_message)
        self.camera_handler.camera_stopped_signal.connect(self.on_camera_handler_stopped)
        self.camera_handler.camera_started_successfully_signal.connect(self.on_camera_handler_started_successfully)
        self.camera_handler.new_detections_signal.connect(self.on_new_detections)
        self.camera_handler.start_camera(zed_sdk_params)

    # ── Slots ─────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_camera_handler_started_successfully(self):
        self.log_message("GUI: Kamera erfolgreich gestartet.")
        self.is_camera_running = True
        self.apply_settings_button.setEnabled(True)

    @pyqtSlot()
    def on_camera_handler_stopped(self):
        self.log_message("GUI: Kamera gestoppt.")
        self.is_camera_running = False
        self.apply_settings_button.setEnabled(True)
        if self.pending_settings_change:
            self.pending_settings_change = False
            self.start_camera_with_current_settings()

    @pyqtSlot(list)
    def on_new_detections(self, detections_list):
        # Letzte Detektionen nur überschreiben wenn neue vorhanden → kein Flackern
        if detections_list:
            self.last_detections = detections_list

    @pyqtSlot(QPixmap)
    def update_image_label(self, pixmap):
        if pixmap.isNull():
            return
        self.last_image_pixmap = pixmap

        if self.last_detections:
            draw_pixmap = pixmap.copy()
            painter = QPainter(draw_pixmap)
            pen = QPen(QColor(0, 255, 0), 3)
            painter.setPen(pen)
            w = draw_pixmap.width()
            h = draw_pixmap.height()
            for det in self.last_detections:
                x1n, y1n, x2n, y2n = det['box_xyxyn']
                x1, y1 = int(x1n * w), int(y1n * h)
                x2, y2 = int(x2n * w), int(y2n * h)
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                painter.drawText(x1, max(y1 - 4, 14), f"{det['label']} {det['confidence']:.2f}")
            painter.end()
            self.image_label.setPixmap(
                draw_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self.image_label.setPixmap(
                pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    @pyqtSlot(QPixmap)
    def update_depth_label(self, pixmap):
        if not pixmap.isNull():
            self.depth_label.setPixmap(
                pixmap.scaled(self.depth_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    @pyqtSlot(str)
    def log_message(self, message):
        if not hasattr(self, 'log_display'):
            print(message)
            return
        print(message)
        current_text = self.log_display.text()
        if "Logs will appear here." in current_text or not current_text.strip():
            self.log_display.setText(message)
        else:
            self.log_display.setText(f"{current_text}\n{message}")
        if hasattr(self, 'scroll_area') and self.scroll_area:
            sb = self.scroll_area.verticalScrollBar()
            if sb:
                sb.setValue(sb.maximum())

    def closeEvent(self, event):
        if self.camera_handler and self.camera_handler.isRunning():
            self.camera_handler.stop()
            self.camera_handler.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    gui = ZedAppGUI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
