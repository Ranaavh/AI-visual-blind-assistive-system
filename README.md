# AI-visual-blind-assistive-system

# Webcam Application with Object Detection and Distance Estimation


![Screenshot 2024-04-20 221124](https://github.com/Ranaavh/AI-visual-blind-assistive-system-/assets/166323572/4698c82d-adde-424c-a719-047f92eedc9b)

## Description

This project implements a real-time webcam application using Python, OpenCV, YOLOv5, and other libraries for object detection and distance estimation. The application detects persons in the webcam feed, calculates their distance from the camera, and provides real-time feedback through graphical and auditory interfaces.

## Features

- Real-time object detection using YOLOv5
- Distance estimation to detected persons based on perspective geometry
- Graphical user interface (GUI) for webcam control and feedback
- Video recording and compression capabilities
- Auditory feedback using text-to-speech (pyttsx3)

## Dependencies

- Python 3.x
- OpenCV
- PyTorch
- YOLOv5
- Pyttsx3
- Tkinter

## Installation

1. Clone the repository:
    https://github.com/Ranaavh/AI-visual-blind-assistive-system-

    also clone yolov5 github repo:https://github.com/ultralytics/yolov5
      

  

2.Setup Anaconda Environment

   conda create --name <env_name> python=<python_version>

   conda activate <env_name>

   pip install package_name


3.Install dependencies:

   pip install -r requirements.txt
   
   python detect.py --weights yolov5s.pt

  



## Usage

1. Run the main script:

   python webcam_app.py

3. Use the GUI buttons to start/stop the webcam, record videos, and control other functionalities.
