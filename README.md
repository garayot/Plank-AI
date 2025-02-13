# Plank.AI - Real-Time Plankton Detection and Tracking

## 🌊 Introduction

**Plank.AI** is an AI-powered application for real-time plankton detection, tracking, and tallying. It utilizes **YOLOv5** for object detection and **Norfair** for object tracking, integrating seamlessly into a user-friendly **PySide6** GUI for visualization and interaction.

---

## 🚀 Features

- **Real-time plankton detection** using YOLOv5
- **Object tracking** with Norfair
- **Customizable display options** (Bounding Boxes, Labels, Object IDs)
- **Camera selection and recording capabilities**
- **Export detection results** to Excel for analysis
- **Plankton density calculation** based on configurable parameters
- **User-friendly GUI** built with PySide6

---

## 🛠️ Installation

### Prerequisites

Ensure you have **Python 3.8+** installed along with the required dependencies.

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Run the Application

```sh
python trial.py
```

---

## 📜 How It Works

1. The **YOLOv5 model** loads pre-trained weights for plankton detection.
2. The **camera captures live frames**, which are processed by the model.
3. **Norfair tracking** maintains object identities across frames.
4. The GUI displays **real-time bounding boxes**, **object labels**, and **track IDs**.
5. The **species tally is updated**, and density calculations are available.
6. Users can **export results to Excel** for further analysis.

---

## 🎛️ User Guide

### Main Features

- **Start, Pause, Stop Detection**: Control the object detection process.
- **Change Camera Source**: Select from available webcam sources.
- **Toggle Display Options**: Show/hide bounding boxes, object labels, and track IDs.
- **Adjust Confidence & Distance Thresholds**: Customize detection sensitivity.
- **Capture Screenshots & Record Videos**: Save detection results.
- **Export Data**: Save species tally to an Excel file.
- **Plankton Density Calculation**: Automatically compute cell density in mL.

---

## 📊 Exporting Data

Results can be exported as an **Excel file** (.xlsx) containing:

- **Species Name**
- **Detection Count**
- **Cell Density per mL**

---

## ⚙️ Technologies Used

- **Python 3.8+**
- **PySide6** (for GUI)
- **OpenCV** (for image processing)
- **YOLOv5** (for object detection)
- **Norfair** (for tracking)
- **Pandas** (for data export)
- **NumPy** (for numerical processing)

---

## 👥 Contributors

- **Amor Lea T. Palatolon**
- **Daniel A. Papaya**
- **Peter Ville C. Carmen**

---

## 📌 Future Improvements

- ✅ Integration with **cloud storage** for dataset sharing.
- ✅ Support for **custom YOLO models**.
- ✅ **Performance optimizations** for real-time tracking.
- ✅ Enhanced **post-processing features**.

---

## 📧 Contact

For inquiries and feedback, contact:
📩 **peter.carmen0101@gmail.com**
