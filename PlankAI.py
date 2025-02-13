import sys
import cv2
import torch
import numpy as np
import pandas as pd  # For exporting to Excel
from pathlib import Path
from norfair import Detection, Tracker, draw_tracked_objects
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QSizePolicy, QSplitter, QSplashScreen, 
    QGroupBox, QPushButton, QScrollArea, QFrame, QComboBox, QFileDialog, QTableWidget, QTableWidgetItem, 
    QSlider, QHBoxLayout, QLineEdit
)
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtCore import QThread, Signal, Qt, QPoint, QTimer
from collections import defaultdict

"""
Process Overview:
This class, `YOLOv5Worker`, is a QThread-based implementation for real-time object detection,
tracking, and tallying of detected classes using YOLOv5 and Norfair tracking.

1. The YOLOv5 model is loaded from a specified weights file.
2. The thread accesses the webcam and continuously captures frames.
3. Each frame is processed by YOLOv5 to detect objects.
4. The detections are converted to a format compatible with Norfair for tracking.
5. The tracker updates and maintains object identities across frames.
6. The detected species are counted uniquely and visualized on the frame.
7. The processed frame is converted and sent as a signal for GUI display.
8. The thread can be stopped safely to release resources.
"""

class YOLOv5Worker(QThread):
    # Signals for communicating with the main application
    frame_processed = Signal(QImage)  # Signal to send the processed frame to the GUI
    class_tally_updated = Signal(dict)  # Signal to update class tallies in the GUI
    error_message = Signal(str)  # Signal to notify about errors

    def __init__(self, camera_index=0):
        super().__init__()
        self.running = False  # Flag to control thread execution
        self.capture = None  # Video capture object
        self.camera_index = camera_index  # Index of the camera to use
        self.yolo_model = None  # YOLOv5 model instance
        self.current_model_path = ""  # Track the currently loaded model to avoid reloading
        self.class_tallies = defaultdict(int)  # Dictionary to store class tallies
        self.tracked_object_ids = set()  # Set to store unique tracked object IDs
        self.conf_thresh = 0.5
        self.distance_threshold = 150
        
        # Assign unique colors for each detected species
        self.species_colors = defaultdict(lambda: tuple([int(x) for x in np.random.choice(range(256), size=3)]))
        
        # Initialize the Norfair tracker with Euclidean distance function
        self.tracker = Tracker(
            distance_function=self.euclidean_distance,
            distance_threshold=150  # Threshold for considering objects as the same
        )
        
        # Load the YOLOv5 model with default weights
        self.load_model("40x.pt")

        self.class_name_correction = {
            "Pyrodinium bahamense var compressum": "Pyrodinium bahamense var. compressum"
        }

    def set_conf_thresh(self, value):
            self.conf_thresh = value / 100  # Convert to decimal

    def set_distance_thresh(self, value):
        self.distance_thresh = value
        self.tracker = Tracker(
            distance_function=self.euclidean_distance,
            distance_threshold=self.distance_thresh
        )
    def load_model(self, weights_path):
        """
        Loads a YOLOv5 model from a specified weights file.
        Prevents reloading the same model to optimize performance.
        """
        if not weights_path or self.current_model_path == weights_path:
            return  # Avoid unnecessary reloading
        try:
            print(f"Loading YOLOv5 model from {weights_path}...")
            self.yolo_model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights_path))
            self.current_model_path = weights_path  # Store the currently loaded model path
            print("YOLOv5 model loaded.")
        except Exception as e:
            print(f"Error loading YOLOv5 model: {e}")
            self.yolo_model = None  # Reset model on failure

    def euclidean_distance(self, detection, tracked_object):
        """
        Computes the Euclidean distance between a detection and a tracked object.
        This is used by Norfair for object tracking.
        """
        detection_points = np.array(detection.points)
        tracked_object_points = np.array(tracked_object.estimate)
        return np.linalg.norm(detection_points - tracked_object_points)
    
    def update_class_tally(self, class_name):
        """
        Updates the class tally, ensuring that incorrect names are corrected.
        """
        corrected_name = self.class_name_correction.get(class_name, class_name)
        self.class_tallies[corrected_name] += 1

    def yolo_to_norfair_detections(self, yolo_results):
        """
        Converts YOLOv5 detection results to Norfair's detection format.
        Extracts bounding boxes and class labels for visualization.
        """
        detections = []
        self.bounding_boxes = []  # Store bounding box info for annotation

        for result in yolo_results.xyxy[0].cpu().numpy():
            x_min, y_min, x_max, y_max, confidence, class_id = result[:6]
                    
            if confidence < self.conf_thresh:
                continue

            center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
            scores = np.array([confidence])
            detections.append(Detection(points=center, scores=scores, label=int(class_id)))

            # Save bounding box and label info for visualization
            class_name = self.yolo_model.names[int(class_id)]
            color = self.species_colors[class_name]  # Assign unique color per class
            self.bounding_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max), class_name, color))

        return detections

    def run(self):

        if not self.yolo_model:
            print("Error: YOLOv5 model not loaded. Please load a valid model.")
            return
        
        self.running = True
        self.capture = cv2.VideoCapture(self.camera_index)  # Open the webcam
        if not self.capture.isOpened():
            error_message = "Error: Unable to access the webcam."
            print(error_message)
            self.error_message.emit(error_message)
            return

        while self.running:
            ret, frame = self.capture.read()
            # if not ret:
            #     error_message = "Error: Failed to read frame from webcam."
            #     print(error_message)
            #     self.error_message.emit(error_message)
            #     break

            # Perform YOLOv5 detection only if the model is loaded
            # try:
            results = self.yolo_model(frame)
            #     print("YOLOv5 detection successful.")
            # except Exception as e:
            #     error_message = f"Error during detection: {e}"
            #     print(error_message)
            #     self.error_message.emit(error_message)
            #     break

            # Convert YOLO detections to Norfair detections
            detections = self.yolo_to_norfair_detections(results)

            # Update Norfair tracker with detections
            tracked_objects = self.tracker.update(detections)

            # Update the tally only for new tracked object IDs
            # for obj in tracked_objects:
            #     if obj.id not in self.tracked_object_ids:
            #         self.tracked_object_ids.add(obj.id)
            #         class_name = self.yolo_model.names[obj.last_detection.label]
            #         self.class_tallies[class_name] += 1
            
            for obj in tracked_objects:
                if obj.id not in self.tracked_object_ids:
                    self.tracked_object_ids.add(obj.id)
                    class_name = self.yolo_model.names[obj.last_detection.label]
                    self.update_class_tally(class_name)

            # Emit updated tallies
            self.class_tally_updated.emit(dict(self.class_tallies))

            # Annotate the frame with bounding boxes and labels
            for (x_min, y_min, x_max, y_max, label, color) in self.bounding_boxes:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 1)
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Annotate the frame with Norfair tracked objects
            draw_tracked_objects(frame, tracked_objects)

            # Convert the frame to QImage for PySide6 display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Emit the processed frame
            self.frame_processed.emit(q_image)

        self.capture.release()

    def stop(self):
        self.running = False
        if self.capture:
            self.capture.release()
        self.quit()
        self.wait()

    def set_camera(self, index):
        self.camera_index = index

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plank AI")
        self.setGeometry(100, 100, 1400, 800)  # Adjusted for new right panel

        self.setWindowIcon(QIcon("plankai-icon.png"))

        self.zoom_level = 100  # Default zoom level percentage
        self.is_panning = False
        self.last_mouse_position = QPoint()
        self.current_offset = QPoint(1, 0)

        # Create main layout using QSplitter for side-by-side layout
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)

        # Left panel for settings
        self.settings_panel = self.create_settings_panel()
        self.splitter.addWidget(self.settings_panel)

        # Center panel for video feed
        self.video_feed_panel = self.create_video_feed_panel()
        self.splitter.addWidget(self.video_feed_panel)

        # Right panel for class tally display
        self.tally_panel = self.create_tally_panel()
        self.splitter.addWidget(self.tally_panel)
        self.species_tally = defaultdict(int)
        self.cell_density = defaultdict(float)


        self.splitter.setStretchFactor(0, 1)  # Settings panel (20%)
        self.splitter.setStretchFactor(1, 3)  # Video feed panel (60%)
        self.splitter.setStretchFactor(2, 1)  # Tally panel (20%)

        # YOLOv5 worker thread
        self.yolo_worker = YOLOv5Worker(camera_index=0)
        self.yolo_worker.frame_processed.connect(self.update_frame)
        self.yolo_worker.class_tally_updated.connect(self.update_tally)
        self.yolo_worker.load_model("40x.pt")
        self.yolo_worker.start()

    def init_ui(self):
        # Add a status bar for quick notifications
        self.statusBar().showMessage("Welcome to Real-Time Plankton Detection!")
    
    def show_notification(self, message):
        """Show a notification in the status bar."""
        self.statusBar().showMessage(message)  

    def create_settings_panel(self):
        # Create a scrollable settings panel
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)

        settings_layout.addWidget(self.create_collapsible_section("Camera List"))
        settings_layout.addWidget(self.create_collapsible_section("Capture & Resolution"))
        settings_layout.addWidget(self.create_collapsible_section("Select Lens"))
        settings_layout.addWidget(self.create_collapsible_section("Slider"))
        settings_layout.addStretch()
        scroll_area.setWidget(settings_widget)
        return scroll_area

    def create_collapsible_section(self, title):
        group_box = QGroupBox(title)
        group_box.setCheckable(True)
        group_box.setChecked(False)

        layout = QVBoxLayout()

        if title == "Camera List":
            camera_label = QLabel("Select Camera:")
            layout.addWidget(camera_label)

            self.camera_combo = QComboBox()
            self.camera_combo.addItems(self.get_camera_list())
            self.camera_combo.currentIndexChanged.connect(self.change_camera)
            layout.addWidget(self.camera_combo)

        elif title == "Capture & Resolution":
            # Button to capture a screenshot
            capture_button = QPushButton("Capture Screenshot")
            capture_button.clicked.connect(self.capture_screenshot)
            layout.addWidget(capture_button)

            # Resolution options
            resolution_label = QLabel("Set Resolution:")
            layout.addWidget(resolution_label)

            self.resolution_combo = QComboBox()
            self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
            self.resolution_combo.currentTextChanged.connect(self.change_resolution)
            layout.addWidget(self.resolution_combo)
        
        elif title == "Select Lens":
            lens_label = QLabel("Select Lens:")
            layout.addWidget(lens_label)

            self.lens_combo = QComboBox()
            self.lens_combo.addItems(["10x", "40x"])
            self.lens_combo.currentTextChanged.connect(self.change_lens_model)
            layout.addWidget(self.lens_combo)

        elif title == "Slider":
            conf_slider_label =QLabel("Confidence Slider:")
            layout.addWidget(conf_slider_label)
            
            self.conf_slider = QSlider(Qt.Horizontal)
            self.conf_slider.setRange(1, 99)
            self.conf_slider.setValue(50)
            self.conf_slider.valueChanged.connect(self.update_conf_thresh)
            layout.addWidget(self.conf_slider)

            dist_slider_label = QLabel("Distance Threshold:")
            layout.addWidget(dist_slider_label)
            
            self.dist_slider = QSlider(Qt.Horizontal)
            self.dist_slider.setRange(0, 300)
            self.dist_slider.setValue(150)
            self.dist_slider.valueChanged.connect(self.update_distance_thresh)
            layout.addWidget(self.dist_slider)

        group_box.setLayout(layout)
        return group_box

    def create_video_feed_panel(self):
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(100, 800)  # Zoom level from 100% to 800%
        self.zoom_slider.setValue(self.zoom_level)
        self.zoom_slider.valueChanged.connect(self.update_zoom_level)

        self.zoom_label = QLabel(f"Zoom: {self.zoom_level}%")

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)

        self.zoom_reset_button = QPushButton("Reset Zoom")
        self.zoom_reset_button.clicked.connect(self.reset_zoom)

        zoom_controls_layout = QHBoxLayout()
        zoom_controls_layout.addWidget(self.zoom_label)
        zoom_controls_layout.addWidget(self.zoom_in_button)
        zoom_controls_layout.addWidget(self.zoom_out_button)
        zoom_controls_layout.addWidget(self.zoom_reset_button)

        video_layout.addLayout(zoom_controls_layout)
        video_layout.addWidget(self.zoom_slider)

        self.video_label = QLabel("Initializing video stream...")
        self.video_label.setStyleSheet(
            """
            QLabel {
                background-color: black;
                color: white;
                font-size: 12px;
                border: 1px solid white;
            }
            """
        )
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        video_layout.addWidget(self.video_label)
        
        return video_widget

    def create_tally_panel(self):
        tally_widget = QWidget()
        tally_layout = QVBoxLayout(tally_widget)

        # Species composition tally
        self.tally_table = QTableWidget()
        self.tally_table.setColumnCount(2)
        self.tally_table.setHorizontalHeaderLabels(["Species", "Count"])
        self.tally_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tally_table.horizontalHeader().setStretchLastSection(True)

        self.export_button = QPushButton("Export to Excel")
        self.export_button.clicked.connect(self.export_to_excel)

        tally_layout.addWidget(QLabel("Species Composition:"))
        tally_layout.addWidget(self.tally_table)
        tally_layout.addWidget(self.export_button)

        # Cell density panel
        self.density_panel = QGroupBox("Cell Density")
        density_layout = QVBoxLayout()

        self.density_table = QTableWidget()
        self.density_table.setColumnCount(2)
        self.density_table.setHorizontalHeaderLabels(["Species", "Cell Density (cells/mL)"])
        self.density_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.density_table.horizontalHeader().setStretchLastSection(True)

        density_layout.addWidget(self.density_table)
        self.density_panel.setLayout(density_layout)

        tally_layout.addWidget(self.density_panel)

        # Input fields for parameters
        params_layout = QHBoxLayout()
        self.length_input = QLineEdit("50")
        self.width_input = QLineEdit("20")
        self.depth_input = QLineEdit("1")
        self.transects_input = QLineEdit("1")

        params_layout.addWidget(QLabel("L (mm):"))
        params_layout.addWidget(self.length_input)
        params_layout.addWidget(QLabel("W (mm):"))
        params_layout.addWidget(self.width_input)
        params_layout.addWidget(QLabel("D (mm):"))
        params_layout.addWidget(self.depth_input)
        params_layout.addWidget(QLabel("S:"))
        params_layout.addWidget(self.transects_input)

        tally_layout.addLayout(params_layout)

        # Calculate button
        calculate_button = QPushButton("Calculate Cell Density")
        calculate_button.clicked.connect(self.calculate_cell_density)
        tally_layout.addWidget(calculate_button)

        return tally_widget

    def get_camera_list(self):
        available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(f"Camera {i}")
                cap.release()
        if not available_cameras:
            available_cameras.append("No cameras available")
        return available_cameras

    def change_camera(self, index):
        if self.yolo_worker.running:
            self.yolo_worker.stop()
            self.yolo_worker.wait()
        
        self.yolo_worker.set_camera(index)
        self.yolo_worker.start()
        success_message = f"Switched to Camera {index}"
        print(success_message)
        self.show_notification(success_message)

    def capture_screenshot(self):
        if self.video_label.pixmap():
            pixmap = self.video_label.pixmap()
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if save_path:
                pixmap.save(save_path)
                success_message = f"Screenshot saved to {save_path}"
                print(success_message)
                self.show_notification(success_message)
        else:
            error_message = "No video frame available to capture."
            print(error_message)
            self.show_notification(error_message)

    def change_resolution(self, resolution):
        """Change the resolution of the video feed."""
        width, height = map(int, resolution.split('x'))
        self.yolo_worker.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.yolo_worker.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"Resolution set to {width}x{height}")

    def change_lens_model(self, lens):
        model_paths = {
            "10x": "10x.pt", 
            "40x": "40x.pt", 
        }
        selected_model = model_paths.get(lens, "40x.pt")
        self.yolo_worker.load_model(selected_model)
        success_message = f"Lens changed to {lens}, model loaded: {selected_model}"
        print(success_message)
        self.show_notification(success_message)

    def update_conf_thresh(self, value):
        self.yolo_worker.set_conf_thresh(value)

    def update_distance_thresh(self, value):
        self.yolo_worker.set_distance_thresh(value)

    def update_zoom_level(self):
        self.zoom_level = self.zoom_slider.value()
        self.zoom_label.setText(f"Zoom: {self.zoom_level}%")
        if self.video_label.pixmap():
            self.apply_zoom_transform()

    def apply_zoom_transform(self):
        pixmap = self.video_label.pixmap()
        # container_width = self.video_label.width()
        # container_height = self.video_label.height()
        scaled_width = pixmap.width() * self.zoom_level // 100
        scaled_height = pixmap.height() * self.zoom_level // 100

        # Constrain QLabel to the size of its container
        scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)
        self.update_pan()

    def update_pan(self):
        self.video_label.move(self.current_offset)

    def zoom_in(self):
        if self.zoom_level < 800:
            self.zoom_level += 10
            self.zoom_slider.setValue(self.zoom_level)

    def zoom_out(self):
        if self.zoom_level > 100:
            self.zoom_level -= 10
            self.zoom_slider.setValue(self.zoom_level)

    def reset_zoom(self):
        self.zoom_level = 100
        self.zoom_slider.setValue(self.zoom_level)
        self.current_offset = QPoint(0, 0)
        self.update_pan()
        self.zoom_label.setText(f"Zoom: {self.zoom_level}%")
        if self.video_label.pixmap():
            self.apply_zoom_transform()

    def update_frame(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)
        self.apply_zoom_transform()

    def eventFilter(self, source, event):
        if source == self.video_label:
            if event.type() == event.MouseButtonPress and self.zoom_level > 100:
                self.is_panning = True
                self.last_mouse_position = event.pos()
            elif event.type() == event.MouseMove and self.is_panning:
                delta = event.pos() - self.last_mouse_position
                self.current_offset += delta
                self.last_mouse_position = event.pos()
                self.update_pan()
            elif event.type() == event.MouseButtonRelease:
                self.is_panning = False
        return super().eventFilter(source, event)

    def update_tally(self, tallies):
        self.species_tally = tallies
        self.tally_table.setRowCount(len(tallies))
        for row, (species, count) in enumerate(tallies.items()):
            self.tally_table.setItem(row, 0, QTableWidgetItem(species))
            self.tally_table.setItem(row, 1, QTableWidgetItem(str(count)))

    def calculate_cell_density(self):
        try:
            L = float(self.length_input.text())
            W = float(self.width_input.text())
            D = float(self.depth_input.text())
            S = float(self.transects_input.text())

            self.cell_density.clear()
            for species, count in self.species_tally.items():
                density = (count * 1000) / (L * W * D * S)
                self.cell_density[species] = density

            self.density_table.setRowCount(len(self.cell_density))
            for row, (species, density) in enumerate(self.cell_density.items()):
                self.density_table.setItem(row, 0, QTableWidgetItem(species))
                self.density_table.setItem(row, 1, QTableWidgetItem(f"{density:.2f}"))

        except ValueError:
            print("Invalid input for parameters. Please ensure all inputs are numbers.")

    def export_to_excel(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Excel Files (*.xlsx)")
        if file_path:
            if not file_path.endswith(".xlsx"):
                file_path += ".xlsx"  # Ensure the file has the correct extension
            data = {"Class Name": [], "Total Tally": []}
            for row in range(self.tally_table.rowCount()):
                data["Class Name"].append(self.tally_table.item(row, 0).text())
                data["Total Tally"].append(int(self.tally_table.item(row, 1).text()))

            df = pd.DataFrame(data)
            try:
                df.to_excel(file_path, index=False)
                success_message = f"Tally exported to {file_path}"
                print(success_message)
                self.show_notification(success_message)
            except Exception as e:
                error_message = f"Error saving Excel file: {e}"
                print(error_message)
                self.show_notification(error_message)

    def closeEvent(self, event):
        self.yolo_worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

     # Load splash image
    splash_pixmap = QPixmap("plankai.png")  # Replace with your splash image path
    splash = QSplashScreen(splash_pixmap, Qt.WindowStaysOnTopHint)
    splash.show()

    # Simulate initialization delay (e.g., loading model, resources, etc.)
    QTimer.singleShot(3000, splash.close)  # Adjust the duration as needed

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
