# 🎾 Tennis Match Analysis Using Deep Learning and Computer Vision

This project is a deep learning-powered system for the automatic analysis of tennis match videos. It detects and tracks the tennis ball, extracts the key points of the playing court, and provides visual and statistical insights through a graphical interface.

## 🔍 Overview

The system combines **object detection**, **court keypoint detection**, and **video annotation** to help users analyze gameplay, visualize ball trajectories, and extract performance metrics — all in a user-friendly desktop application.

### Key Features
- ⚙️ **YOLOv5 & YOLOv8**: Custom-trained for accurate tennis ball detection.
- 📍 **Court Keypoint Detection**: Uses a ResNet-50 CNN to extract 14 court reference points.
- 🧠 **Ball Tracking**: Interpolates ball trajectories across frames.
- 📊 **Match Statistics**: Computes ball speed, number of shots, and trajectory data.
- 🌡 **Heatmap Generation**: Highlights frequently hit zones on the court.
- 🖼 **Mini-Court Projection**: Maps ball hits onto a scaled-down court for visual summaries.
- 🖥 **Tkinter GUI**: Allows users to upload videos, run analysis, and view/download results.

## 🛠 Technologies Used
- **Python 3.x**
- **OpenCV**
- **Ultralytics YOLOv5 & YOLOv8**
- **PyTorch**
- **ResNet-50 (for court keypoint detection)**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **PIL**
- **yt-dlp**
- **Pickle**
- **Tkinter**

## 🚀 How It Works

1. **Upload a Video** via the GUI.
2. **YOLO Model** detects the ball in each frame.
3. **Ball Trajectory** is interpolated to fill in missed detections.
4. **ResNet50** model extracts court keypoints to define court geometry.
5. **Court Projection** places ball hits onto a mini-court layout.
6. **Statistics & Heatmaps** are generated and displayed when pressing the "See Advanced Results" button.
7. **Output Video** is annotated and saved with detection overlays.

## 📷 Example Outputs

<img width="1918" height="1079" alt="Screenshot 2025-07-27 143959" src="https://github.com/user-attachments/assets/dd0e146d-8fac-4647-9d41-f856c8ea3c97" />
<img width="1919" height="1079" alt="Screenshot 2025-07-27 144006" src="https://github.com/user-attachments/assets/b0642c11-a14f-4c00-8770-9738bb2c23d2" />
<img width="1919" height="1079" alt="Screenshot 2025-07-27 144057" src="https://github.com/user-attachments/assets/828dbcef-138e-47bf-a39e-7b1b5b8206bd" />
<img width="1919" height="1078" alt="Screenshot 2025-07-27 144106" src="https://github.com/user-attachments/assets/27cf344c-8e26-4897-9e0f-1495bd4b1801" />



