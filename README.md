# Plank.AI - Real-Time Plankton Detection and Tracking

## 🌊 Introduction

**Plank.AI** is an AI-powered application for real-time plankton detection, tracking, and tallying. It utilizes **YOLOv5** for object detection and **Norfair** for object tracking, integrating seamlessly into a user-friendly **PySide6** GUI for visualization and interaction.

Now enhanced with an **AI-powered chat assistant**, Plank.AI can provide insightful answers and context-aware suggestions based on the **plankton species tally metadata**.

![Screenshot](images/screenshot.png)

---

## 🚀 Features

- **Real-time plankton detection** using YOLOv5
- **Object tracking** with Norfair
- **AI Chat Assistant** integrated into the GUI for data-driven conversations
- **Customizable display options** (Bounding Boxes, Labels, Object IDs, Plankton Size)
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
4. The GUI displays **real-time bounding boxes**, **object labels**, **track IDs**, and **plankton size**.
5. The **species tally is updated**, and density calculations are available.
6. The **AI Assistant** responds to user queries based on the tally's metadata.
7. Users can **export results to Excel** for further analysis.

---

## 🎛️ User Guide

### Main Features

- **Start, Pause, Stop Detection**: Control the object detection process.
- **Change Camera Source**: Select from available webcam sources.
- **Toggle Display Options**: Show/hide bounding boxes, object labels, and track IDs.
- **Adjust Confidence & Distance Thresholds**: Customize detection sensitivity.
- **Capture Screenshots & Record Videos**: Save detection results.
- **Switch Models for different Objectives**: supports only 10x and 40x objectives.
- **Export Data**: Save species tally to an Excel file.
- **Plankton Density Calculation**: Automatically compute cell density in mL.
- **AI Chat Assistant**: Ask questions and get real-time answers based on detection data.

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
- **Google Generative AI API** (for AI assistant)

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
- ✅ Web Support **Website App**.
- 🔄 Enhanced natural language understanding for the **AI Assistant**.
- 🔄 Real-time communication between tally updates and chat responses.

---

## 📧 Contact

For inquiries and feedback, contact:
📩 **peter.carmen0101@gmail.com**
