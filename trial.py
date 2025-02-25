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
    QSlider, QLineEdit, QCheckBox, QMenuBar, QMessageBox, QFormLayout, QDialog, QHBoxLayout
)
from PySide6.QtGui import QImage, QPixmap, QIcon, QAction, QMouseEvent
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QSize, QTime, QDateTime, QPoint, QRect

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
        self.show_bboxes = True #toggles bounding boxes
        self.show_ids = True #toggles object ids
        self.show_labels = True #toggles object labels
        self.display_size_labels = True # Default display size labels
        self.display_rulers = False
        self.running = False  # Flag to control thread execution
        self.capture = None  # Video capture object
        self.camera_index = camera_index  # Index of the camera to use
        self.yolo_model = None  # YOLOv5 model instance
        self.current_model_path = ""  # Track the currently loaded model to avoid reloading
        self.class_tallies = defaultdict(int)  # Dictionary to store class tallies
        self.tracked_object_ids = set()  # Set to store unique tracked object IDs
        self.conf_thresh = 0.5 # Default confidence threshold for YOLOv5 detections
        self.distance_threshold = 150 # Default distance threshold for Norfair object tracking
        
        
        # Assign unique colors for each detected species
        self.species_colors = defaultdict(lambda: tuple([int(x) for x in np.random.choice(range(7), size=3)]))
        
        # Initialize the Norfair tracker with Euclidean distance function
        self.tracker = Tracker(
            distance_function=self.euclidean_distance,
            distance_threshold=150  # Threshold for considering objects as the same
        )
        
        # Load the YOLOv5 model with default weights
        self.load_model("models/10xv3.pt")

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

    def set_display_options(self, show_bboxes, show_ids, show_labels, display_size_labels, display_rulers):
        self.show_bboxes = show_bboxes
        self.show_ids = show_ids
        self.show_labels = show_labels
        self.display_size_labels = bool(display_size_labels)
        self.display_rulers = display_rulers

    # def toggle_size_labels(self, state):
    #     self.display_size_labels = bool(state)

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
            width_um = (x_max - x_min) * 0.25  # Convert pixels to micrometers
            height_um = (y_max - y_min) * 0.25 # Adjust for calibration
            self.bounding_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max), class_name, color, width_um, height_um))

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
            for obj in tracked_objects:
                if obj.id not in self.tracked_object_ids:
                    self.tracked_object_ids.add(obj.id)
                    class_name = self.yolo_model.names[obj.last_detection.label]
                    self.update_class_tally(class_name)

            # Emit updated tallies
            self.class_tally_updated.emit(dict(self.class_tallies))

            # Annotate the frame with bounding boxes, size, and labels
            for (x_min, y_min, x_max, y_max, label, color, width_um, height_um) in self.bounding_boxes:
                if self.show_bboxes:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 1)
                if self.display_size_labels:
                    size_label = f"{width_um:.2f} um x {height_um:.2f} um"
                    cv2.putText(frame, size_label, (x_min, y_max + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                if self.show_labels:
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Annotate the frame with Norfair tracked objects
            if self.show_ids: 
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
        # self.setGeometry(100, 100, 1400, 800)  # Adjusted for new right panel 

        self.setWindowIcon(QIcon("images/plankai-icon.ico"))  # Set the window icon

        self.video_save_path = "C:/Users/garay/Desktop/Plank AI"  # Default empty path
        self.is_recording = False
        self.video_writer = None
        self.start_time = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer_display)
        self.elapsed_time = 0

        self.show_bboxes = True
        self.show_ids = True
        self.show_labels = True
        self.display_size_labels = True
        self.display_rulers = False

        self.zoom_factor = 1.0

        self.pan_offset = QPoint(0, 0)  # Initial panning offset
        self.is_panning = False  # Track whether panning is active
        self.last_mouse_pos = QPoint()  # Store last mouse position

        self.setMenuBar(self.create_menu_bar())

        # Initialize parameters as attributes
        self.length_value = "50"  
        self.width_value = "20"
        self.depth_value = "1"
        self.transects_value = "1"

        self.toolbar = self.addToolBar("Main Toolbar")
        self.toolbar.setIconSize(QSize(24, 24))  # Set icon size

        # Create main layout using QSplitter layout
        main_layout = QVBoxLayout()
        self.top_panel = self.toolbar_actions()
        self.add_zoom_controls()
        # main_layout.addWidget(self.top_panel)

        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

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
        self.yolo_worker.load_model("models/10xv3.pt")
        self.yolo_worker.set_display_options(self.show_bboxes, self.show_ids, self.show_labels, self.display_size_labels, self.display_rulers)
        self.yolo_worker.start()

    def init_ui(self):
        # Add a status bar for quick notifications
        self.statusBar().showMessage("Welcome to Real-Time Plankton Detection!")
    
    def show_notification(self, message):
        """Show a notification in the status bar."""
        self.statusBar().showMessage(message)  

    def create_menu_bar(self):
        menu_bar = QMenuBar(self)

        # File Menu
        file_menu = menu_bar.addMenu("File")
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        load_tally_action = QAction("Load Tally File", self)
        load_tally_action.triggered.connect(self.load_tally_file)
        file_menu.addAction(load_tally_action)

        # About Menu
        about_menu = menu_bar.addMenu("About")
        about_action = about_menu.addAction("About Plank.AI")
        about_action.triggered.connect(self.show_about_dialog)

        # Adjust Cell Density params
        preferences_menu = menu_bar.addMenu("Preferences")
        cell_density_calculator_action = preferences_menu.addAction("Cell Density Calculator")
        video_directory_action = preferences_menu.addAction("Video Directory")
        cell_density_calculator_action.triggered.connect(self.show_preference_dialog)
        video_directory_action.triggered.connect(self.set_save_path)
        
        # Help Menu
        help_menu = menu_bar.addMenu("Help")
        user_guide_action = help_menu.addAction("User Guide")
        feedback_action = help_menu.addAction("Feedback")
        user_guide_action.triggered.connect(self.show_user_guide)
        feedback_action.triggered.connect(self.show_feedback)


        return menu_bar

    def show_about_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("About Plank.AI")
        # dialog.setFixedSize(400, 300)  # Set fixed size for consistency

        layout = QVBoxLayout(dialog)

        # Add Logo
        logo_label = QLabel()
        pixmap = QPixmap("images/plankai-icon.png")
        logo_label.setPixmap(pixmap.scaled(64, 64, Qt.KeepAspectRatio))
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # About Text
        about_text = """
        <h2 style='color: #3498db; text-align: center;'>Plank.AI</h2>
        <p style='text-align: center;'><b>An AI-based plankton detection and counting system.</b></p>
        <br>
        <p style='text-align: center;'><b>Developed by:</b></p>
        <p style='text-align: center;'>Amor Lea T. Palatolon</p>
        <p style='text-align: center;'>Daniel A. Papaya</p>
        <p style='text-align: center;'>Peter Ville C. Carmen</p>
        <br>
        <p style='text-align: center;'><b>Version:</b> 1.0</p>
        """

        about_label = QLabel(about_text)
        about_label.setWordWrap(True)
        about_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(about_label)

        # OK Button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        ok_button.setStyleSheet("padding: 5px; font-size: 12px;")
        layout.addWidget(ok_button, alignment=Qt.AlignCenter)

        dialog.setLayout(layout)
        dialog.exec()

    def show_preference_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Cell Density Calculator Parameters")
        layout = QFormLayout(dialog)

        # Create input fields using stored values
        length_input = QLineEdit(self.length_value)
        width_input = QLineEdit(self.width_value)
        depth_input = QLineEdit(self.depth_value)
        transects_input = QLineEdit(self.transects_value)

        layout.addRow("L (mm):", length_input)
        layout.addRow("W (mm):", width_input)
        layout.addRow("D (mm):", depth_input)
        layout.addRow("S:", transects_input)

        # Save button
        save_button = QPushButton("Save")
        save_button.clicked.connect(lambda: self.update_input_params(dialog, length_input, width_input, depth_input, transects_input))
        layout.addRow(save_button)

        dialog.setLayout(layout)
        dialog.exec()

    def set_save_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if directory:
            self.video_save_path = directory
            QMessageBox.information(self, "Save Path Set", f"Videos will be saved to: {directory}")

    def show_user_guide(self):
        user_guide_box = QMessageBox(self)
        user_guide_box.setWindowTitle("User Guide")
        user_guide_text = """
        <h3>Plank.AI User Guide</h3>
        <p>To use Plank.AI:</p>
        <ol>
            <li>Select a camera.</li>
            <li>Click Start to begin detection.</li>
            <li>Adjust settings as needed.</li>
            <li>Check results in the tally panel.</li>
            <li>Export data if necessary.</li>
        </ol>
        """
        user_guide_box.setText(user_guide_text)
        user_guide_box.setStandardButtons(QMessageBox.Ok)
        user_guide_box.exec()   

    def show_feedback(self):
        QMessageBox.information(self, "Feedback", 
                                "For feedback and inquiries, please contact:\n\n"
                                "Email: peter.carmen0101@gmail.com")
 
    def toolbar_actions(self):
        
        start_action = QAction(QIcon("icons/play.png"), "Start", self)
        start_action.triggered.connect(self.start_detection)
        self.toolbar.addAction(start_action)

        pause_action = QAction(QIcon("icons/pause-button.png"), "Pause", self)
        pause_action.triggered.connect(self.pause_detection)
        self.toolbar.addAction(pause_action)

        stop_action = QAction(QIcon("icons/stop-button.png"), "Stop", self)
        stop_action.triggered.connect(self.stop_detection)
        self.toolbar.addAction(stop_action)

    def add_zoom_controls(self):
        zoom_layout = QHBoxLayout()

        zoom_out_btn = QPushButton()
        zoom_out_btn.setIcon(QIcon("icons/zoom-out.png"))
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_btn)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        zoom_layout.addWidget(self.zoom_slider)

        self.zoom_label = QLabel("100%")
        zoom_layout.addWidget(self.zoom_label)

        zoom_in_btn = QPushButton()
        zoom_in_btn.setIcon(QIcon("icons/zoom-in.png"))
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_btn)

        reset_zoom_btn = QPushButton("Reset")
        reset_zoom_btn.clicked.connect(self.reset_zoom)
        zoom_layout.addWidget(reset_zoom_btn)

        zoom_container = QWidget()
        zoom_container.setLayout(zoom_layout)
        self.toolbar.addWidget(zoom_container)

    def zoom_in(self):
        self.zoom_slider.setValue(min(self.zoom_slider.value() + 20, 400))

    def zoom_out(self):
        self.zoom_slider.setValue(max(self.zoom_slider.value() - 20, 60))

    def reset_zoom(self):
        self.zoom_slider.setValue(100)
        self.pan_offset = QPoint(0, 0)  # Reset panning
        self.update_frame(self.current_qimage)

    def update_zoom(self, value):
        self.zoom_factor = value / 100.0
        self.zoom_label.setText(f"{value}%")
        self.update_frame(self.current_qimage)
    
    def create_settings_panel(self):
        # Create a scrollable settings panel
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(200)
        scroll_area.setFrameShape(QFrame.NoFrame)

        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)

        # Group box for display options
        display_options_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()

        self.bbox_checkbox = QCheckBox("Show Bounding Boxes")
        self.bbox_checkbox.setChecked(True)
        self.bbox_checkbox.setToolTip("Toggle to show or hide bounding boxes around detected objects.")
        self.bbox_checkbox.stateChanged.connect(self.toggle_display_options)
        display_layout.addWidget(self.bbox_checkbox)

        self.id_checkbox = QCheckBox("Show Object IDs")
        self.id_checkbox.setChecked(True)
        self.id_checkbox.setToolTip("Toggle to show or hide unique tracking IDs for each detected objects.")
        self.id_checkbox.stateChanged.connect(self.toggle_display_options)
        display_layout.addWidget(self.id_checkbox)

        self.label_checkbox = QCheckBox("Show Labels")
        self.label_checkbox.setChecked(True)
        self.label_checkbox.setToolTip("Toggle to show or hide species name on detected objects.")
        self.label_checkbox.stateChanged.connect(self.toggle_display_options)
        display_layout.addWidget(self.label_checkbox)

        self.toggle_size_checkbox = QCheckBox("Show Box Sizes")
        self.toggle_size_checkbox.setChecked(True)
        self.toggle_size_checkbox.setToolTip("Toggle to display the width and height of detected objects in micrometers.")
        self.toggle_size_checkbox.stateChanged.connect(self.toggle_display_options)
        display_layout.addWidget(self.toggle_size_checkbox)

        self.rulers_checkbox = QCheckBox("Show Rulers")
        self.rulers_checkbox.setChecked(False)
        self.rulers_checkbox.setToolTip("Toggle to display rulers on the video feed")
        self.rulers_checkbox.stateChanged.connect(self.toggle_display_options)
        display_layout.addWidget(self.rulers_checkbox)

        display_options_group.setLayout(display_layout)
        settings_layout.addWidget(display_options_group)

        settings_layout.addWidget(self.create_collapsible_section("Camera List"))
        settings_layout.addWidget(self.create_collapsible_section("Capture & Record"))
        settings_layout.addWidget(self.create_collapsible_section("Select Lens"))
        settings_layout.addWidget(self.create_collapsible_section("Slider"))

        
        settings_layout.addStretch()
        scroll_area.setWidget(settings_widget)
        return scroll_area

    def create_collapsible_section(self, title):
        group_box = QGroupBox(title)
        group_box.setCheckable(True)
        group_box.setChecked(True)

        layout = QVBoxLayout()

        if title == "Camera List":
            camera_label = QLabel("Select Camera:")
            layout.addWidget(camera_label)

            self.camera_combo = QComboBox()
            self.camera_combo.setToolTip("Select an available camera to use for detection.")
            self.camera_combo.addItems(self.get_camera_list())
            self.camera_combo.currentIndexChanged.connect(self.change_camera)
            layout.addWidget(self.camera_combo)

        elif title == "Capture & Record":
            # Button to capture a screenshot
            capture_button = QPushButton("Capture Screenshot")
            capture_button.setToolTip("Save the current video frame as a screenshot.")
            capture_button.clicked.connect(self.capture_screenshot)
            layout.addWidget(capture_button)

            # Button to start/stop recording
            self.record_button = QPushButton("Start Recording")
            self.record_button.setToolTip("Start or stop recording the video feed.")
            self.record_button.clicked.connect(self.toggle_recording)
            layout.addWidget(self.record_button)
        
        elif title == "Select Lens":
            lens_label = QLabel("Select Lens:")
            layout.addWidget(lens_label)

            self.lens_combo = QComboBox()
            self.lens_combo.addItems(["10x", "40x"])
            self.lens_combo.setToolTip("Choose the magnification level of the microscope lens.")
            self.lens_combo.currentTextChanged.connect(self.change_lens_model)
            layout.addWidget(self.lens_combo)

        elif title == "Slider":
            conf_slider_label =QLabel("Confidence Slider:")
            layout.addWidget(conf_slider_label)
            
            self.conf_slider = QSlider(Qt.Horizontal)
            self.conf_slider.setRange(1, 99)
            self.conf_slider.setValue(50)
            self.conf_slider.setToolTip("Adjust the confidence threshold for object detection (higher values show fewer but more accurate detections).")
            self.conf_slider.valueChanged.connect(self.update_conf_thresh)
            layout.addWidget(self.conf_slider)

            dist_slider_label = QLabel("Distance Threshold:")
            layout.addWidget(dist_slider_label)
            
            self.dist_slider = QSlider(Qt.Horizontal)
            self.dist_slider.setRange(0, 300)
            self.dist_slider.setValue(150)
            self.dist_slider.setToolTip("Adjust the tracking distance threshold (higher values track objects over longer distances).")
            self.dist_slider.valueChanged.connect(self.update_distance_thresh)
            layout.addWidget(self.dist_slider)

        group_box.setLayout(layout)
        return group_box

    def create_video_feed_panel(self):
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)

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
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setScaledContents(False)
        self.video_label.setMouseTracking(True)

        self.video_label.mousePressEvent = self.start_pan
        self.video_label.mouseMoveEvent = self.pan_video
        self.video_label.mouseReleaseEvent = self.end_pan

        video_layout.addWidget(self.video_label)
        
        return video_widget

    def create_tally_panel(self):
        tally_widget = QWidget()
        tally_widget.setFixedWidth(270)
        tally_layout = QVBoxLayout(tally_widget)

        # Species composition tally
        self.tally_table = QTableWidget()
        self.tally_table.setColumnCount(3)
        self.tally_table.setHorizontalHeaderLabels(["Species", "Count", "Cells per mL"])
        self.tally_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tally_table.horizontalHeader().setStretchLastSection(True)

        calculate_button = QPushButton("Calculate Cell Density")
        calculate_button.clicked.connect(self.calculate_cell_density)

        self.export_button = QPushButton("Export to Excel")
        self.export_button.clicked.connect(self.export_to_excel)

        tally_layout.addWidget(QLabel("Species Composition:"))
        tally_layout.addWidget(self.tally_table)
        tally_layout.addWidget(calculate_button)
        tally_layout.addWidget(self.export_button)

        return tally_widget

    def start_detection(self):
        if not self.yolo_worker.isRunning():
            self.yolo_worker.start()
            self.statusBar().showMessage("Detection started.")

    def pause_detection(self):
        if self.yolo_worker.running:
            self.yolo_worker.running = False
            self.statusBar().showMessage("Detection paused.")

    def stop_detection(self):
        if self.yolo_worker.isRunning():
            self.yolo_worker.stop()
            self.statusBar().showMessage("Detection stopped.")

    def toggle_display_options(self):
        self.show_bboxes = self.bbox_checkbox.isChecked()
        self.show_ids = self.id_checkbox.isChecked()
        self.show_labels = self.label_checkbox.isChecked()
        self.display_size_labels = self.toggle_size_checkbox.isChecked()
        self.display_rulers = self.rulers_checkbox.isChecked()
        self.yolo_worker.set_display_options(self.show_bboxes, self.show_ids, self.show_labels, self.display_size_labels, self.display_rulers)

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

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if not self.video_save_path:
            QMessageBox.critical(self, "Error", "No save path set. Please set a save path in Preferences.")
            return
        
        # Ensure directory exists
        save_dir = Path(self.video_save_path)
        if not save_dir.exists():
            QMessageBox.critical(self, "Error", "Save directory does not exist. Please select a valid path.")
            return
        
        filename = save_dir / f"recording_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = int(self.yolo_worker.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.yolo_worker.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.yolo_worker.capture.get(cv2.CAP_PROP_FPS)) or 11  # Default to 30 if 0

        self.video_writer = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))
        self.is_recording = True
        self.start_time = QTime.currentTime()
        self.elapsed_time = 0
        self.timer.start(10)  # Update timer every 10ms
        self.statusBar().showMessage("Recording started...")
        self.record_button.setText("Stop Recording")
        self.current_video_filename = str(filename)

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        self.timer.stop()
        self.statusBar().showMessage(f"Recording saved: {self.current_video_filename}")
        QMessageBox.information(self, "Recording Saved", f"Video saved to: {self.current_video_filename}")
        self.record_button.setText("Start Recording")

    def update_timer_display(self):
        if self.start_time:
            elapsed = self.start_time.msecsTo(QTime.currentTime())
            minutes = elapsed // 60000
            seconds = (elapsed % 60000) // 1000
            milliseconds = elapsed % 1000
            self.statusBar().showMessage(f"Recording: {minutes:02}:{seconds:02}.{milliseconds:03}")

    def change_lens_model(self, lens):
        model_paths = {
            "10x": "models/10xv3.pt", 
            "40x": "models/40x.pt", 
        }
        selected_model = model_paths.get(lens, "models/10xv3.pt")
        self.yolo_worker.load_model(selected_model)
        self.update_frame(self.current_qimage)
        success_message = f"Lens changed to {lens}, model loaded: {selected_model}"
        print(success_message)
        self.show_notification(success_message)

    def update_conf_thresh(self, value):
        self.yolo_worker.set_conf_thresh(value)

    def update_distance_thresh(self, value):
        self.yolo_worker.set_distance_thresh(value)

    def draw_rulers(self, frame):
        if not self.display_rulers:
            return frame
        height, width, _ = frame.shape
        scale_factor = self.get_scale_factor()

        # Define ruler parameters
        num_divisions = 10
        spacing = width // num_divisions  # Equally spaced divisions
        micrometer_spacing = spacing * scale_factor  # Convert pixels to µm

        # Draw horizontal ruler
        for i in range(num_divisions + 1):
            x_pos = i * spacing
            cv2.line(frame, (x_pos, height - 30), (x_pos, height - 10), (255, 255, 255), 1)
            cv2.putText(frame, f"{i * micrometer_spacing:.1f}", (x_pos + 2, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw vertical ruler
        for i in range(num_divisions + 1):
            y_pos = i * spacing
            cv2.line(frame, (10, y_pos), (30, y_pos), (255, 255, 255), 1)
            cv2.putText(frame, f"{i * micrometer_spacing:.1f}", (35, y_pos + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def get_scale_factor(self):
        """Get the scale factor based on selected lens and zoom level."""
        base_factors = {"10x": 1.5, "40x": 0.375}  # µm per pixel for each lens
        lens = self.lens_combo.currentText()
        zoom_adjustment = self.zoom_factor

        return base_factors.get(lens, 1.5) / zoom_adjustment  # Adjust for zoom level

    def update_frame(self, q_image):
        self.current_qimage = q_image  # Store current frame
        
        # Convert QImage to OpenCV format
        frame = q_image.bits().tobytes()
        frame = np.frombuffer(frame, dtype=np.uint8).reshape((q_image.height(), q_image.width(), 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Draw rulers before converting back
        frame = self.draw_rulers(frame)
        
        # Convert back to QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0],
                        rgb_frame.shape[1] * 3, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        new_size = pixmap.size() * self.zoom_factor
        scaled_pixmap = pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Constrain pan offset to prevent black areas
        max_x_offset = max(0, scaled_pixmap.width() - self.video_label.width())
        max_y_offset = max(0, scaled_pixmap.height() - self.video_label.height())
        self.pan_offset.setX(max(0, min(self.pan_offset.x(), max_x_offset)))
        self.pan_offset.setY(max(0, min(self.pan_offset.y(), max_y_offset)))
        
        # Apply panning offset
        target_rect = QRect(self.pan_offset.x(), self.pan_offset.y(), self.video_label.width(), self.video_label.height())
        cropped_pixmap = scaled_pixmap.copy(target_rect.intersected(scaled_pixmap.rect()))
        self.video_label.setPixmap(cropped_pixmap)
        
        if self.is_recording and self.video_writer:
            frame = q_image.bits().tobytes()
            frame = np.frombuffer(frame, dtype=np.uint8).reshape((q_image.height(), q_image.width(), 3))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame)

    def start_pan(self, event: QMouseEvent):
        if self.zoom_factor > 1.0:
            self.is_panning = True
            self.last_mouse_pos = event.position().toPoint()

    def pan_video(self, event: QMouseEvent):
        if self.is_panning:
            delta = event.position().toPoint() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.position().toPoint()
            self.update_frame(self.current_qimage)

    def end_pan(self, event: QMouseEvent):
        self.is_panning = False


    def load_tally_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Tally File", "", "Excel Files (*.xlsx)")
        if file_path:
            try:
                df = pd.read_excel(file_path)
                if not all(col in df.columns for col in ["Species", "Count", "Cell Density (cells/mL)"]):
                    QMessageBox.warning(self, "Invalid File", "The selected file does not have the required columns.")
                    return

                self.tally_table.setRowCount(len(df))
                for row, (species, count, density) in enumerate(zip(df["Species"], df["Count"], df["Cell Density (cells/mL)"])):
                    self.tally_table.setItem(row, 0, QTableWidgetItem(species))
                    self.tally_table.setItem(row, 1, QTableWidgetItem(str(count)))
                    self.tally_table.setItem(row, 2, QTableWidgetItem(f"{density:.2f}"))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def update_tally(self, tallies):
        self.species_tally = tallies
        self.tally_table.setRowCount(len(tallies))
        for row, (species, count) in enumerate(tallies.items()):
            self.tally_table.setItem(row, 0, QTableWidgetItem(species))
            self.tally_table.setItem(row, 1, QTableWidgetItem(str(count)))

    def calculate_cell_density(self):
        try:
            L = float(self.length_value)  
            W = float(self.width_value)  
            D = float(self.depth_value)  
            S = float(self.transects_value)  

            self.cell_density.clear()
            for species, count in self.species_tally.items():
                density = (count * 1000) / (L * W * D * S)
                self.cell_density[species] = density

            self.tally_table.setRowCount(len(self.cell_density))
            for row, (species, density) in enumerate(self.cell_density.items()):
                self.tally_table.setItem(row, 0, QTableWidgetItem(species))
                self.tally_table.setItem(row, 1, QTableWidgetItem(str(self.species_tally[species])))
                self.tally_table.setItem(row, 2, QTableWidgetItem(f"{density:.2f}"))

        except ValueError:
            print("Invalid input for parameters. Please ensure all inputs are numbers.")

    def update_input_params(self, dialog, length_input, width_input, depth_input, transects_input):
        """Update stored values for parameters from the preferences dialog."""
        self.length_value = length_input.text()
        self.width_value = width_input.text()
        self.depth_value = depth_input.text()
        self.transects_value = transects_input.text()
        dialog.accept()

    def export_to_excel(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Excel Files (*.xlsx)")
        if file_path:
            if not file_path.endswith(".xlsx"):
                file_path += ".xlsx"  # Ensure the file has the correct extension

            data = {"Species": [], "Count": [], "Cell Density (cells/mL)": []}
            for row in range(self.tally_table.rowCount()):
                data["Species"].append(self.tally_table.item(row, 0).text())
                data["Count"].append(int(self.tally_table.item(row, 1).text()))
                data["Cell Density (cells/mL)"].append(float(self.tally_table.item(row, 2).text()))
        
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
    splash_pixmap = QPixmap("images/plankai.png")  # Replace with your splash image path
    splash = QSplashScreen(splash_pixmap, Qt.WindowStaysOnTopHint)
    splash.show()

    # Simulate initialization delay (e.g., loading model, resources, etc.)
    QTimer.singleShot(3000, splash.close)  # Adjust the duration as needed

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
