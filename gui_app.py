import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QFrame, QPushButton, QComboBox, QCheckBox, QScrollArea)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot # Ensure pyqtSlot is imported
import cv2
import numpy as np
import pyzed.sl as sl

cv2.ocl.setUseOpenCL(False)

class ZedCameraWorker(QThread):
    new_image_signal = pyqtSignal(QPixmap)
    new_depth_signal = pyqtSignal(QPixmap)
    log_message_signal = pyqtSignal(str)
    camera_stopped_signal = pyqtSignal()
    camera_started_successfully_signal = pyqtSignal() # New signal

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zed = sl.Camera()
        self.running = False
        self.init_params = None # This will be sl.InitParameters()

    def start_camera(self, zed_sdk_params): # Renamed for clarity
        self.init_params = zed_sdk_params
        self.log_message_signal.emit(f"Worker: Starting camera with: Res: {self.init_params.camera_resolution}, FPS: {self.init_params.camera_fps}, Depth: {self.init_params.depth_mode}")
        self.start()

    def run(self):
        self.running = True

        if not self.init_params:
            self.log_message_signal.emit("Worker: Error - Camera parameters not set before starting.")
            self.camera_stopped_signal.emit()
            return

        # Ensure camera is closed before attempting to open
        if self.zed.is_opened():
            self.zed.close()
            self.log_message_signal.emit("Worker: Closed existing camera session before opening new one.")

        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.log_message_signal.emit(f"Worker: Error opening ZED camera: {err}")
            if self.zed.is_opened(): # Should not be necessary if open failed, but good practice
                self.zed.close()
            self.running = False
            self.camera_stopped_signal.emit()
            return

        self.log_message_signal.emit("Worker: ZED Camera opened successfully.")
        self.camera_started_successfully_signal.emit() # Emit new signal

        image_sl = sl.Mat()
        depth_map_sl = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()

        # Custom runtime_parameters for confidence_threshold and texture_confidence_threshold removed.
        # SDK default values will be used.

        while self.running:
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve and log left image
                self.zed.retrieve_image(image_sl, sl.VIEW.LEFT)
                image_data_np = image_sl.get_data()
                self.log_message_signal.emit(f"Worker: image_data_np shape: {{image_data_np.shape if image_data_np is not None else 'None'}}, dtype: {{image_data_np.dtype if image_data_np is not None else 'None'}}")

                image_for_qt = None
                if image_data_np is not None and image_data_np.size > 0:
                    self.log_message_signal.emit(f"Worker: image_data_np is valid. Shape: {{image_data_np.shape}}, NumDims: {{image_data_np.ndim}}")
                    if image_data_np.ndim == 3 and image_data_np.shape[2] == 4:
                        image_for_qt = image_data_np[:, :, :3] # Assume RGBA, take RGB
                    elif image_data_np.ndim == 3 and image_data_np.shape[2] == 3:
                        image_for_qt = image_data_np # Assume RGB
                    else:
                        self.log_message_signal.emit(f"Worker: Retrieved image with unexpected shape/dims: {{image_data_np.shape}}")
                        continue
                else:
                    self.log_message_signal.emit("Worker: Retrieved empty or invalid image data from ZED.")
                    continue

                self.log_message_signal.emit(f"Worker: image_for_qt shape: {{image_for_qt.shape if image_for_qt is not None else 'None'}}")
                if image_for_qt is not None:
                    h, w, ch = image_for_qt.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(image_for_qt.data, w, h, bytes_per_line, QImage.Format_RGB888).copy() # .copy() is important!
                    self.log_message_signal.emit(f"Worker: Emitting new_image_signal. qt_image.isNull(): {{qt_image.isNull()}}, size: {{qt_image.size()}}")
                    self.new_image_signal.emit(QPixmap.fromImage(qt_image))

                # Retrieve and log depth map if depth mode is not NONE
                if self.init_params.depth_mode != sl.DEPTH_MODE.NONE:
                    self.zed.retrieve_measure(depth_map_sl, sl.MEASURE.DEPTH)
                    depth_map_ocv = depth_map_sl.get_data()
                    self.log_message_signal.emit(f"Worker: depth_map_ocv shape: {{depth_map_ocv.shape if depth_map_ocv is not None else 'None'}}, dtype: {{depth_map_ocv.dtype if depth_map_ocv is not None else 'None'}}")

                    if depth_map_ocv is not None and depth_map_ocv.size > 0:
                        depth_map_display = np.nan_to_num(depth_map_ocv, nan=0.0, posinf=0.0, neginf=0.0)
                        min_val, max_val = np.min(depth_map_display), np.max(depth_map_display)
                        if max_val > min_val:
                            depth_map_normalized = ((depth_map_display - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                        else:
                            depth_map_normalized = np.zeros(depth_map_display.shape, dtype=np.uint8)

                        # h_d, w_d were already defined from depth_map_normalized.shape
                        h_d, w_d = depth_map_normalized.shape
                        if not depth_map_normalized.flags.c_contiguous:
                             depth_map_normalized = np.ascontiguousarray(depth_map_normalized)
                        qt_depth_image = QImage(depth_map_normalized.data, w_d, h_d, w_d, QImage.Format_Grayscale8).copy()
                        self.log_message_signal.emit(f"Worker: Emitting new_depth_signal (grayscale). qt_depth_image.isNull(): {{qt_depth_image.isNull()}}, size: {{qt_depth_image.size() if qt_depth_image else 'None'}}")
                        self.new_depth_signal.emit(QPixmap.fromImage(qt_depth_image))
                    else:
                        self.log_message_signal.emit("Worker: Retrieved empty or invalid depth data.")
                else:
                    # Emit an empty pixmap if depth is NONE
                    dummy_res_w = self.init_params.camera_resolution.width // 2 if self.init_params else 320
                    dummy_res_h = self.init_params.camera_resolution.height // 2 if self.init_params else 180
                    empty_pixmap = QPixmap(dummy_res_w, dummy_res_h)
                    empty_pixmap.fill(Qt.black)
                    self.new_depth_signal.emit(empty_pixmap)
            else:
                self.msleep(10) # Wait briefly if grab failed

        if self.zed.is_opened():
            self.zed.close()
        self.log_message_signal.emit("Worker: ZED Camera closed.")
        self.camera_stopped_signal.emit()

    def stop(self):
        self.log_message_signal.emit("Worker: Attempting to stop...")
        self.running = False

class ZedAppGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZED Camera Viewer")
        self.setGeometry(100, 100, 1200, 700) # Adjusted default size

        # Initialize instance variables for current settings
        self.current_resolution_enum = sl.RESOLUTION.HD1200
        self.current_fps_val = 30
        self.current_depth_mode_enum = sl.DEPTH_MODE.NEURAL

        self.is_camera_running = False # Explicitly initialized
        self.pending_settings_change = False # For managing settings changes
        self.camera_worker = None

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- Left Panel (Controls) ---
        left_panel_frame = QFrame(self)
        left_panel_frame.setFrameShape(QFrame.StyledPanel)
        left_panel_frame.setFixedWidth(300)
        left_panel_layout = QVBoxLayout(left_panel_frame)

        settings_title_label = QLabel("Settings Panel", left_panel_frame)
        settings_title_label.setAlignment(Qt.AlignCenter)
        left_panel_layout.addWidget(settings_title_label)

        # Resolution ComboBox
        left_panel_layout.addWidget(QLabel("Resolution:", left_panel_frame))
        self.resolution_map = {
            "HD1200": sl.RESOLUTION.HD1200, "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720, "VGA": sl.RESOLUTION.VGA
        }
        self.resolution_combo = QComboBox(left_panel_frame)
        self.resolution_combo.addItems(self.resolution_map.keys())
        self.resolution_combo.setCurrentText("HD1200") # Default
        left_panel_layout.addWidget(self.resolution_combo)

        # FPS ComboBox
        left_panel_layout.addWidget(QLabel("FPS:", left_panel_frame))
        self.fps_map = {"15": 15, "30": 30, "60": 60, "100": 100}
        self.fps_combo = QComboBox(left_panel_frame)
        self.fps_combo.addItems(self.fps_map.keys())
        self.fps_combo.setCurrentText(str(self.current_fps_val)) # Default
        left_panel_layout.addWidget(self.fps_combo)

        # Depth Mode ComboBox
        left_panel_layout.addWidget(QLabel("Depth Mode:", left_panel_frame))
        self.depth_mode_map = {
            "NEURAL": sl.DEPTH_MODE.NEURAL, "ULTRA": sl.DEPTH_MODE.ULTRA,
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE, "QUALITY": sl.DEPTH_MODE.QUALITY,
            "NONE": sl.DEPTH_MODE.NONE
        }
        self.depth_mode_combo = QComboBox(left_panel_frame)
        self.depth_mode_combo.addItems(self.depth_mode_map.keys())
        # Find key for NEURAL to set as default
        neural_key = [k for k, v in self.depth_mode_map.items() if v == self.current_depth_mode_enum][0]
        self.depth_mode_combo.setCurrentText(neural_key)
        left_panel_layout.addWidget(self.depth_mode_combo)

        # Apply Settings Button
        self.apply_settings_button = QPushButton("Apply Settings", left_panel_frame)
        self.apply_settings_button.clicked.connect(self.apply_settings)
        left_panel_layout.addWidget(self.apply_settings_button)
        left_panel_layout.addStretch(1)

        # Log Area
        log_area_label = QLabel("Camera Logs:", left_panel_frame)
        left_panel_layout.addWidget(log_area_label)
        self.log_display = QLabel("Logs will appear here.", left_panel_frame)
        self.log_display.setWordWrap(True)
        self.log_display.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log_display.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px;")
        scroll_area = QScrollArea(left_panel_frame)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.log_display)
        scroll_area.setFixedHeight(200) # Adjust as needed
        left_panel_layout.addWidget(scroll_area)
        main_layout.addWidget(left_panel_frame, 1) # Relative width 1

        # --- Right Panel (Image and Depth Views) ---
        right_panel_frame = QFrame(self)
        right_panel_frame.setFrameShape(QFrame.StyledPanel)
        right_panel_layout = QVBoxLayout(right_panel_frame)

        self.image_label = QLabel("Camera Feed", right_panel_frame)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setFrameShape(QFrame.Box)
        img_palette = self.image_label.palette()
        img_palette.setColor(QPalette.Window, QColor("black"))
        self.image_label.setPalette(img_palette)
        self.image_label.setAutoFillBackground(True)
        right_panel_layout.addWidget(self.image_label, 2) # Relative height 2

        self.depth_label = QLabel("Depth Map", right_panel_frame)
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setMinimumSize(640, 180) # Adjusted for proportion
        self.depth_label.setFrameShape(QFrame.Box)
        depth_palette = self.depth_label.palette()
        depth_palette.setColor(QPalette.Window, QColor("black"))
        self.depth_label.setPalette(depth_palette)
        self.depth_label.setAutoFillBackground(True)
        right_panel_layout.addWidget(self.depth_label, 1) # Relative height 1
        main_layout.addWidget(right_panel_frame, 3) # Relative width 3

        self.log_message("GUI Initialized. Settings configured for autostart.")
        self.show()
        self.log_message("GUI: Autostarting camera...")
        self.start_camera_with_current_settings()

    def apply_settings(self):
        self.log_message("GUI: Apply Settings button clicked.")
        if self.is_camera_running and self.camera_worker:
            self.log_message("GUI: Camera is running. Stopping for settings change...")
            self.apply_settings_button.setEnabled(False) # Disable button during change
            self.pending_settings_change = True
            self.camera_worker.stop()
        else:
            self.log_message("GUI: Camera not running. Starting with new settings...")
            self.pending_settings_change = False # Ensure it's false if we start directly
            self.start_camera_with_current_settings()

    def start_camera_with_current_settings(self):
        self.log_message("GUI: Reading settings from UI and preparing to start camera...")
        self.apply_settings_button.setEnabled(False) # Disable while starting/changing

        # Read current settings from UI
        selected_res_str = self.resolution_combo.currentText()
        self.current_resolution_enum = self.resolution_map.get(selected_res_str, self.current_resolution_enum)

        selected_fps_str = self.fps_combo.currentText()
        self.current_fps_val = self.fps_map.get(selected_fps_str, self.current_fps_val)

        selected_depth_str = self.depth_mode_combo.currentText()
        self.current_depth_mode_enum = self.depth_mode_map.get(selected_depth_str, self.current_depth_mode_enum)

        self.log_message(f"GUI: Preparing ZED params: Res={selected_res_str}, FPS={self.current_fps_val}, Depth={selected_depth_str}")

        zed_sdk_params = sl.InitParameters()
        zed_sdk_params.camera_resolution = self.current_resolution_enum
        zed_sdk_params.camera_fps = self.current_fps_val
        zed_sdk_params.depth_mode = self.current_depth_mode_enum
        zed_sdk_params.coordinate_units = sl.UNIT.METER
        zed_sdk_params.sdk_verbose = 1 # Enable SDK verbose logging
        # zed_sdk_params.input = sl.InputType() # Not needed for live camera
        # zed_sdk_params.input.set_from_camera_id(0) # Default behavior

        # Clean up previous worker if it exists
        if self.camera_worker:
            if self.camera_worker.isRunning():
                self.log_message("GUI: Waiting for existing worker to stop...")
                self.camera_worker.stop()
                self.camera_worker.wait(2000) # Wait up to 2s
            # Disconnect signals from old worker to prevent multiple calls or errors
            try:
                self.camera_worker.new_image_signal.disconnect()
                self.camera_worker.new_depth_signal.disconnect()
                self.camera_worker.log_message_signal.disconnect()
                self.camera_worker.camera_stopped_signal.disconnect()
            except TypeError: # Thrown if signals were not connected or already disconnected
                self.log_message("GUI: Signals for old worker were already disconnected or not set.")
            except Exception as e:
                self.log_message(f"GUI: Error disconnecting signals: {e}")

            # Also try to disconnect the new signal
            if hasattr(self.camera_worker, 'camera_started_successfully_signal'):
                try:
                    self.camera_worker.camera_started_successfully_signal.disconnect(self.on_camera_worker_started_successfully)
                except TypeError:
                    pass # Already disconnected or never connected
                except Exception as e:
                    self.log_message(f"GUI: Error disconnecting camera_started_successfully_signal: {e}")


        self.camera_worker = ZedCameraWorker(self)
        self.camera_worker.new_image_signal.connect(self.update_image_label)
        self.camera_worker.new_depth_signal.connect(self.update_depth_label)
        self.camera_worker.log_message_signal.connect(self.log_message)
        self.camera_worker.camera_stopped_signal.connect(self.on_camera_worker_stopped)
        self.camera_worker.camera_started_successfully_signal.connect(self.on_camera_worker_started_successfully) # Connect new signal

        self.camera_worker.start_camera(zed_sdk_params)
        # self.is_camera_running = True # This will be set by on_camera_worker_started_successfully
        # Button will be re-enabled by on_camera_worker_started_successfully or on_camera_worker_stopped

    @pyqtSlot()
    def on_camera_worker_started_successfully(self):
        self.log_message("GUI: Camera worker confirmed successful start.")
        self.is_camera_running = True
        self.apply_settings_button.setEnabled(True)

    @pyqtSlot()
    def on_camera_worker_stopped(self):
        self.log_message("GUI: Camera worker confirmed stopped.")
        self.is_camera_running = False # Set state first
        self.apply_settings_button.setEnabled(True) # Enable button

        if self.pending_settings_change:
            self.log_message("GUI: Re-applying settings after stop...")
            self.pending_settings_change = False
            self.start_camera_with_current_settings() # Start again with new settings
        else:
            # If not due to pending settings change, it might be an error or normal stop
            # Ensure UI is ready for user to try again if it was an error
            self.log_message("GUI: Camera stopped. UI is active for new settings.")


    @pyqtSlot(QPixmap)
    def update_image_label(self, pixmap):
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.log_message("GUI: Received null QPixmap for image.")

    @pyqtSlot(QPixmap)
    def update_depth_label(self, pixmap):
        if not pixmap.isNull():
            self.depth_label.setPixmap(pixmap.scaled(self.depth_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.log_message("GUI: Received null QPixmap for depth.")

    @pyqtSlot(str)
    def log_message(self, message):
        # Prepend with "GUI:" or "Worker:" if not already done
        prefix = "GUI: "
        if "Worker:" in message or "GUI:" in message:
            prefix = ""

        current_text = self.log_display.text()
        if current_text == "Logs will appear here." or not current_text:
             self.log_display.setText(prefix + message)
        else:
             self.log_display.setText(f"{current_text}\n{prefix}{message}")

        # Auto-scroll to the bottom
        scrollbar = self.log_display.parent().parent().verticalScrollBar() # Accessing scrollbar of QScrollArea
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

        print(prefix + message) # Also print to console for redundancy

    def closeEvent(self, event):
        self.log_message("GUI: Closing application...")
        if self.camera_worker and self.camera_worker.isRunning():
            self.log_message("GUI: Stopping camera worker before exiting...")
            self.camera_worker.stop()
            self.camera_worker.wait(3000) # Wait up to 3 seconds
        event.accept()

def main():
    app = QApplication(sys.argv)
    # app.setAttribute(Qt.AA_EnableHighDpiScaling) # Optional: For HiDPI displays
    gui = ZedAppGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
