import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
import pyttsx3
import time
import numpy as np
import datetime
import subprocess

class WebcamApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.attributes("-fullscreen", True)  # Set window to fullscreen
        self.cap = None  # VideoCapture object
        self.webcam_active = False
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device).eval()
        self.engine = pyttsx3.init()
        self.average_height_cm = 170
        self.window_size = 5
        self.distance_measurements = []
        self.first_detection = True
        self.last_detection_time = 0
        self.out = None  # VideoWriter object
        self.video_filename = None
        self.frame_width = None
        self.frame_height = None
        self.codec = cv2.VideoWriter_fourcc(*'XVID')  # Define MPEG-4 codec

        # Set background image for the window
        try:
            bg_image = Image.open("C:/Users/ranah/yolojupi/yolov5/img1.jpg")
            window_width = self.window.winfo_screenwidth()
            window_height = self.window.winfo_screenheight()
            bg_image = bg_image.resize((window_width, window_height))
            self.background_image = ImageTk.PhotoImage(bg_image)
            self.background_label = tk.Label(window, image=self.background_image)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load background image: {e}")

        # Create buttons
        self.stop_button = tk.Button(window, text="Stop Webcam", width=20, command=self.stop_webcam, bg="white", fg="black", state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=20, pady=10)

        self.start_button = tk.Button(window, text="Start Webcam", width=20, command=self.start_webcam, bg="blue", fg="white")
        self.start_button.pack(side=tk.RIGHT, padx=20, pady=10)

        # OpenCV video display
        self.video_label = tk.Label(window)
        self.video_label.pack()

        self.window.mainloop()

    def start_webcam(self):
        if not self.webcam_active:
            self.cap = cv2.VideoCapture(0)
            self.webcam_active = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.create_new_video_file()
            self.show_webcam()
        else:
            messagebox.showwarning("Warning", "Webcam is already active!")

    def create_new_video_file(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.video_filename = f"output_{timestamp}.avi"
        self.out = cv2.VideoWriter(self.video_filename, self.codec, 20.0, (self.frame_width, self.frame_height))

    def stop_webcam(self):
        if self.webcam_active:
            self.cap.release()
            if self.out is not None:
                self.out.release()
                self.compress_video()  # Compress the video before releasing resources
            self.webcam_active = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.video_label.config(image="")
        else:
            messagebox.showwarning("Warning", "Webcam is not active!")

    def calculate_distance(self, h):
        focal_length_cm = 250
        object_height_cm = self.average_height_cm
        return focal_length_cm * object_height_cm / h

    def update_moving_average(self, distance):
        self.distance_measurements.append(distance)
        if len(self.distance_measurements) > self.window_size:
            self.distance_measurements.pop(0)

    def moving_average(self):
        if self.distance_measurements:
            return sum(self.distance_measurements) / len(self.distance_measurements)
        else:
            return None

    def compress_video(self):
        if self.video_filename:
            compressed_filename = f"compressed_{self.video_filename}"
            ffmpeg_command = f"ffmpeg -i {self.video_filename} -vf scale=640:480 -c:v libx264 -crf 28 {compressed_filename}"
            try:
                subprocess.run(ffmpeg_command, shell=True, check=True)
                messagebox.showinfo("Success", f"Video compressed and saved as {compressed_filename}")
            except subprocess.CalledProcessError as e:
                messagebox.showerror("Error", f"Failed to compress video: {e}")

    def show_webcam(self):
        _, frame = self.cap.read()
        
        # Apply color correction to reduce violet shade
        corrected_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        corrected_frame = cv2.convertScaleAbs(corrected_frame, alpha=1.2, beta=0)
        
        results = self.model(corrected_frame)
        person_results = [detection for detection in results.pandas().xyxy[0].to_dict(orient='records') if detection['name'] == 'person']

        if person_results:
            person = person_results[0]  # Only consider the first detected person
            x1, y1, x2, y2, conf = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax']), person['confidence']
            cv2.rectangle(corrected_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
            object_height = y2 - y1
            distance = self.calculate_distance(object_height)
            self.update_moving_average(distance)
            smoothed_distance = self.moving_average()
            if smoothed_distance is not None:
                # Convert distance to integer
                integer_distance = int(smoothed_distance)
                cv2.putText(corrected_frame, f"Distance: {integer_distance} cm", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(corrected_frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Speak the distance
                current_time = time.time()
                if self.first_detection or current_time - self.last_detection_time >= 10:
                    self.engine.say(f"Distance to person: {integer_distance} centimeters")
                    if integer_distance > 150:
                        self.engine.say("Person is far")
                    else:
                        self.engine.say("Person is near")
                    self.engine.runAndWait()
                    self.last_detection_time = current_time  # Update last announcement time
                
                if self.first_detection or current_time - self.last_detection_time >= 20:
                    self.engine.say("Person detected")
                    self.engine.runAndWait()
                    self.first_detection = False
                    self.last_detection_time = current_time

        else:
            self.first_detection = True

        # Write the frame into the video file
        if self.out is not None:
            # Compress the frame before writing to the video file
            compressed_frame = cv2.resize(corrected_frame, (self.frame_width, self.frame_height))
            self.out.write(compressed_frame)

        img = Image.fromarray(corrected_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        if self.webcam_active:
            self.video_label.after(10, self.show_webcam)

# Create a Tkinter window
root = tk.Tk()
app = WebcamApp(root, "Webcam Application")
