import sys
import cv2
import torch
import numpy as np
import pandas as pd
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
        self.running = False  # Flag to control thread execution
        self.capture = None  # Video capture object
        self.camera_index = camera_index  # Index of the camera to use
        self.yolo_model = None  # YOLOv5 model instance
        self.current_model_path = ""  # Track the currently loaded model to avoid reloading
        self.class_tallies = defaultdict(int)  # Dictionary to store class tallies
        self.tracked_object_ids = set()  # Set to store unique tracked object IDs
        
        
        # Assign unique colors for each detected species
        self.species_colors = defaultdict(lambda: tuple([int(x) for x in np.random.choice(range(7), size=3)]))
        
        # Initialize the Norfair tracker with Euclidean distance function
        self.tracker = Tracker(
            distance_function=self.euclidean_distance,
            distance_threshold=150  # Threshold for considering objects as the same
        )
        
        # Load the YOLOv5 model with default weights
        self.load_model("C:/Users/garay/Desktop/Thesis data/models/10xv3.pt")

        self.class_name_correction = {
            "Pyrodinium bahamense var compressum": "Pyrodinium bahamense var. compressum"
        }

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
            results = self.yolo_model(frame)
           
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

        self.tally_panel = self.create_tally_panel()

	# YOLOv5 worker thread
        self.yolo_worker = YOLOv5Worker(camera_index=0)
        self.yolo_worker.frame_processed.connect(self.update_frame)
        self.yolo_worker.class_tally_updated.connect(self.update_tally)
        self.yolo_worker.load_model("C:/Users/garay/Desktop/Thesis data/models/10xv3.pt")
        self.yolo_worker.set_display_options(self.show_bboxes, self.show_ids, self.show_labels, 	self.display_size_labels, self.display_rulers)
        self.yolo_worker.start()

    def init_ui(self):
        # Add a status bar for quick notifications
        self.statusBar().showMessage("Welcome to Real-Time Plankton Detection!")
    
    def show_notification(self, message):
        """Show a notification in the status bar."""
        self.statusBar().showMessage(message) 

    def create_tally_panel(self):
        tally_widget = QWidget()
        tally_widget.setFixedWidth(270)
        tally_layout = QVBoxLayout(tally_widget)

        # Input fields for metadata
        metadata_layout = QFormLayout()
        self.date_input = QLineEdit()
        self.location_input = QLineEdit()
        self.technician_input = QLineEdit()
        metadata_layout.addRow("Date:", self.date_input)
        metadata_layout.addRow("Location:", self.location_input)
        metadata_layout.addRow("Lab Technician:", self.technician_input)
        # layout.addLayout(metadata_layout)

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

        tally_layout.addWidget(QLabel("Metadata:"))
        tally_layout.addLayout(metadata_layout)
        tally_layout.addWidget(QLabel("Species Composition:"))
        tally_layout.addWidget(self.tally_table)
        tally_layout.addWidget(calculate_button)
        tally_layout.addWidget(self.export_button)

        return tally_widget

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
                file_path += ".xlsx"

            # Capture metadata values
            self.sample_date = self.date_input.text()
            self.sample_location = self.location_input.text()
            self.lab_technician = self.technician_input.text()

            data = {
                "Species": [], "Count": [], "Cell Density (cells/mL)": []
            }
            for row in range(self.tally_table.rowCount()):
                data["Species"].append(self.tally_table.item(row, 0).text())
                data["Count"].append(int(self.tally_table.item(row, 1).text()))
                data["Cell Density (cells/mL)"].append(float(self.tally_table.item(row, 2).text()))

            df = pd.DataFrame(data)
            metadata_df = pd.DataFrame({
                "Date": [self.sample_date],
                "Location": [self.sample_location],
                "Lab Technician": [self.lab_technician]
            })

            with pd.ExcelWriter(file_path) as writer:
                metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
                df.to_excel(writer, sheet_name="Tally Data", index=False)

            QMessageBox.information(self, "Export Successful", f"Data exported to {file_path}")

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