import sys
import cv2
import numpy as np
import pyzed.sl as sl
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QFrame, QPushButton, QComboBox, QCheckBox, QScrollArea)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

class ZedCameraWorker(QThread):
    new_image_signal = pyqtSignal(QPixmap)
    new_depth_signal = pyqtSignal(QPixmap)
    log_message_signal = pyqtSignal(str)
    camera_stopped_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zed = sl.Camera()
        self.running = False
        self.init_params = None

    def start_camera(self, init_params):
        self.init_params = init_params
        self.log_message_signal.emit(f"Starting camera with: Res: {self.init_params.camera_resolution}, FPS: {self.init_params.camera_fps}, Depth: {self.init_params.depth_mode}")
        self.start()

    def run(self):
        self.running = True

        if not self.init_params:
            self.log_message_signal.emit("Error: Camera parameters not set before starting worker.")
            self.camera_stopped_signal.emit()
            return

        if self.zed.is_opened():
            self.zed.close()

        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.log_message_signal.emit(f"Error opening ZED: {err}")
            if self.zed.is_opened():
                self.zed.close()
            self.running = False
            self.camera_stopped_signal.emit()
            return

        self.log_message_signal.emit("ZED Camera opened successfully.")

        image_sl = sl.Mat()
        depth_map_sl = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()

        if self.init_params.depth_mode == sl.DEPTH_MODE.ULTRA or            self.init_params.depth_mode == sl.DEPTH_MODE.QUALITY:
             runtime_parameters.confidence_threshold = 80
             runtime_parameters.texture_confidence_threshold = 80 # Example for ULTRA
        elif self.init_params.depth_mode == sl.DEPTH_MODE.PERFORMANCE:
             runtime_parameters.confidence_threshold = 50
             runtime_parameters.texture_confidence_threshold = 50 # Example for PERFORMANCE
        # No specific runtime params for DEPTH_MODE.NONE related to depth retrieval

        while self.running:
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image_sl, sl.VIEW.LEFT)
                image_ocv_rgba = image_sl.get_data()
                image_ocv_rgb = cv2.cvtColor(image_ocv_rgba, cv2.COLOR_RGBA2RGB)

                h, w, ch = image_ocv_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(image_ocv_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.new_image_signal.emit(QPixmap.fromImage(qt_image))

                if self.init_params.depth_mode != sl.DEPTH_MODE.NONE:
                    self.zed.retrieve_measure(depth_map_sl, sl.MEASURE.DEPTH)
                    depth_map_ocv = depth_map_sl.get_data()

                    depth_map_display = np.nan_to_num(depth_map_ocv, nan=0.0, posinf=0.0, neginf=0.0)
                    min_val, max_val = np.min(depth_map_display), np.max(depth_map_display)
                    if max_val > min_val:
                        depth_map_normalized = ((depth_map_display - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                    else:
                        depth_map_normalized = np.zeros(depth_map_display.shape, dtype=np.uint8)

                    h_d, w_d = depth_map_normalized.shape
                    bytes_per_line_d = w_d
                    qt_depth_image = QImage(depth_map_normalized.data, w_d, h_d, bytes_per_line_d, QImage.Format_Grayscale8)
                    self.new_depth_signal.emit(QPixmap.fromImage(qt_depth_image))
                else:
                    # Emit an empty pixmap or clear depth label if depth is NONE
                    empty_pixmap = QPixmap(self.init_params.camera_resolution.width // 2, self.init_params.camera_resolution.height // 2) # Or some default size
                    empty_pixmap.fill(Qt.black)
                    self.new_depth_signal.emit(empty_pixmap)
            else:
                self.msleep(10)

        if self.zed.is_opened():
            self.zed.close()
        self.log_message_signal.emit("ZED Camera closed in worker thread.")
        self.camera_stopped_signal.emit()

    def stop(self):
        self.log_message_signal.emit("Attempting to stop camera worker...")
        self.running = False

class ZedAppGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZED Camera Viewer")
        self.setGeometry(100, 100, 1200, 700)

        self.current_resolution = sl.RESOLUTION.HD720
        self.current_fps = 30
        self.current_depth_mode = sl.DEPTH_MODE.PERFORMANCE # Default
        self.is_camera_running = False

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel_frame = QFrame(self)
        left_panel_frame.setFrameShape(QFrame.StyledPanel)
        left_panel_frame.setFixedWidth(300)
        left_panel_layout = QVBoxLayout(left_panel_frame)

        settings_title_label = QLabel("Settings Panel", left_panel_frame)
        settings_title_label.setAlignment(Qt.AlignCenter)
        left_panel_layout.addWidget(settings_title_label)

        self.resolution_map = {
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "VGA": sl.RESOLUTION.VGA
        }
        self.resolution_combo = QComboBox(left_panel_frame)
        self.resolution_combo.addItems(self.resolution_map.keys())
        self.resolution_combo.setCurrentText("HD720")
        left_panel_layout.addWidget(QLabel("Resolution:", left_panel_frame))
        left_panel_layout.addWidget(self.resolution_combo)

        self.depth_mode_checkbox = QCheckBox("Enable Ultra Depth Mode", left_panel_frame) # Changed label
        left_panel_layout.addWidget(self.depth_mode_checkbox)

        self.apply_settings_button = QPushButton("Apply Settings", left_panel_frame)
        self.apply_settings_button.clicked.connect(self.apply_settings)
        left_panel_layout.addWidget(self.apply_settings_button)

        left_panel_layout.addStretch(1)

        log_area_label = QLabel("Camera Logs:", left_panel_frame)
        left_panel_layout.addWidget(log_area_label)
        self.log_display = QLabel("Logs will appear here.", left_panel_frame)
        self.log_display.setWordWrap(True)
        self.log_display.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log_display.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px;")
        scroll_area = QScrollArea(left_panel_frame)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.log_display)
        scroll_area.setFixedHeight(200)
        left_panel_layout.addWidget(scroll_area)

        right_panel = QFrame(self)
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        self.image_label = QLabel("Press 'Apply Settings' to start Camera", self)
        # ... (image_label and depth_label setup remains the same as before)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setFrameShape(QFrame.Box)
        img_palette = self.image_label.palette()
        img_palette.setColor(QPalette.Window, QColor("black"))
        self.image_label.setPalette(img_palette)
        self.image_label.setAutoFillBackground(True)
        right_layout.addWidget(self.image_label, 2)

        self.depth_label = QLabel("Depth data will appear here", self) # Default text
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setMinimumSize(640, 180)
        self.depth_label.setFrameShape(QFrame.Box)
        depth_palette = self.depth_label.palette()
        depth_palette.setColor(QPalette.Window, QColor("black"))
        self.depth_label.setPalette(depth_palette)
        self.depth_label.setAutoFillBackground(True)
        right_layout.addWidget(self.depth_label, 1)


        main_layout.addWidget(left_panel_frame, 1)
        main_layout.addWidget(right_panel, 3)

        self.camera_worker = None
        self.log_message("GUI Initialized. Configure settings and press 'Apply Settings'.")
        self.show()

    def apply_settings(self):
        if self.camera_worker and self.camera_worker.isRunning():
            self.log_message("Stopping current camera stream to apply new settings...")
            self.apply_settings_button.setEnabled(False)
            self.camera_worker.stop()
            # Camera restart will be handled by on_camera_worker_stopped
            return

        # If camera is not running, or was stopped, directly start with new settings
        self.start_camera_with_current_settings()


    def start_camera_with_current_settings(self):
        self.log_message("Applying new settings and starting camera...")
        self.apply_settings_button.setEnabled(False)

        selected_res_str = self.resolution_combo.currentText()
        self.current_resolution = self.resolution_map.get(selected_res_str, sl.RESOLUTION.HD720)

        if self.depth_mode_checkbox.isChecked():
            self.current_depth_mode = sl.DEPTH_MODE.ULTRA
            self.log_message("Ultra Depth Mode selected.")
        else:
            self.current_depth_mode = sl.DEPTH_MODE.PERFORMANCE
            self.log_message("Performance Depth Mode selected.")

        new_params = sl.InitParameters()
        new_params.camera_resolution = self.current_resolution
        new_params.camera_fps = self.current_fps
        new_params.depth_mode = self.current_depth_mode
        new_params.coordinate_units = sl.UNIT.METER

        self.log_message(f"Config: Res={new_params.camera_resolution}, FPS={new_params.camera_fps}, Depth={new_params.depth_mode}")

        new_params.input = sl.InputType()
        new_params.input.set_from_camera_id(0)

        # Ensure previous worker is fully cleaned up if one existed
        if self.camera_worker:
            if self.camera_worker.isRunning(): # Should have been stopped by apply_settings
                self.camera_worker.stop()
                self.camera_worker.wait() # Wait for it to finish
            # Disconnect signals from old worker
            try:
                self.camera_worker.new_image_signal.disconnect()
                self.camera_worker.new_depth_signal.disconnect()
                self.camera_worker.log_message_signal.disconnect()
                self.camera_worker.camera_stopped_signal.disconnect()
            except TypeError: # Signals already disconnected
                pass


        self.camera_worker = ZedCameraWorker(self)
        self.camera_worker.new_image_signal.connect(self.update_image_label)
        self.camera_worker.new_depth_signal.connect(self.update_depth_label)
        self.camera_worker.log_message_signal.connect(self.log_message)
        self.camera_worker.camera_stopped_signal.connect(self.on_camera_worker_stopped)

        self.camera_worker.start_camera(new_params)
        self.is_camera_running = True
        # self.apply_settings_button.setEnabled(True) # Re-enabled in on_camera_worker_stopped

    def on_camera_worker_stopped(self):
        self.is_camera_running = False
        self.log_message("Camera worker confirmed stopped.")
        self.apply_settings_button.setEnabled(True)
        # Check if a settings change triggered this stop; if so, restart.
        # Current logic in apply_settings handles this: if worker was running, it's stopped.
        # Then, if apply_settings is called again (or if it was the first call), it starts.
        # This simple sequence should be okay.

    @pyqtSlot(QPixmap)
    def update_image_label(self, pixmap):
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @pyqtSlot(QPixmap)
    def update_depth_label(self, pixmap):
        self.depth_label.setPixmap(pixmap.scaled(self.depth_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @pyqtSlot(str)
    def log_message(self, message):
        current_text = self.log_display.text()
        if current_text == "Logs will appear here." or "Configure settings and press 'Apply Settings'." in current_text :
             self.log_display.setText(message)
        else:
             self.log_display.setText(f"{self.log_display.text()}\n{message}")
        try:
            self.log_display.parentWidget().verticalScrollBar().setValue(self.log_display.parentWidget().verticalScrollBar().maximum())
        except AttributeError:
            pass
        print(message)

    def closeEvent(self, event):
        self.log_message("Closing application...")
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.stop()
            self.camera_worker.wait(5000)
        event.accept()

def main():
    app = QApplication(sys.argv)
    gui = ZedAppGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
