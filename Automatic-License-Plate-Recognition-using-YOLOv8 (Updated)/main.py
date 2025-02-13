import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk  # For adding an image/logo
import sys
from ultralytics import YOLO
import cv2
import numpy as np
import os  # For checking if files exist

sys.path.append('C:/Users/ah442/Automatic-License-Plate-Recognition-using-YOLOv8/sort')
from sort import Sort
from util import get_car, read_license_plate, write_csv

# Function to handle video selection
def select_option(event=None):
    option = mode_combobox.get()
    if option == "Process Video":
        video_path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[("Video Files", "*.mp4 *.avi")]
        )
        if video_path:
            root.destroy()  # Close the GUI after selection
            process_video(video_path)
        else:
            messagebox.showerror("Error", "No video file selected. Please try again.")
    elif option == "Start Live Monitoring":
        root.destroy()  # Close the GUI after selection
        start_live_monitoring()

# Function to generate a unique output filename for video
def generate_output_filename():
    base_name = './tracked_output'
    file_extension = '.mp4'
    counter = 0
    output_filename = base_name + file_extension
    
    # Check if the file already exists and increment the counter if it does
    while os.path.exists(output_filename):
        counter += 1
        output_filename = f"{base_name}_{counter}{file_extension}"
    
    return output_filename

# Function to generate a unique output filename for CSV
def generate_csv_filename():
    base_name = './tracked_output'
    file_extension = '.csv'
    counter = 0
    output_filename = base_name + file_extension
    
    # Check if the file already exists and increment the counter if it does
    while os.path.exists(output_filename):
        counter += 1
        output_filename = f"{base_name}_{counter}{file_extension}"
    
    return output_filename

# Function to process the video (simplified for demonstration)
def process_video(video_path):
    # Generate a unique filename for the output video
    output_filename = generate_output_filename()
    
    # Generate a unique filename for the CSV
    csv_filename = generate_csv_filename()
    
    # Initialize results and tracker
    results = {}
    mot_tracker = Sort()

    # Load models
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')

    # Load the selected video
    cap = cv2.VideoCapture(video_path)

    # Set up output video writer with the generated filename
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Vehicle class IDs (e.g., cars, trucks)
    vehicles = [2, 3, 5, 7]

    # Process video frames
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}

            # Detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # Track vehicles using SORT
            track_ids = mot_tracker.update(np.asarray(detections_))

            # Detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Assign license plate to a tracked car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # Crop and process the license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(
                        license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
                    )

                    # Read the license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(
                        license_plate_crop_thresh
                    )

                    # Store results if a license plate is detected
                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score,
                            },
                        }

                    # Draw bounding boxes and annotations
                    cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Car ID: {int(car_id)}",
                        (int(xcar1), int(ycar1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"Plate: {license_plate_text}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

            # Write the processed frame to the output video
            out.write(frame)

    # Save results to a unique CSV and release resources
    write_csv(results, csv_filename)
    cap.release()
    out.release()

    messagebox.showinfo("Processing Complete", f"Results saved to '{csv_filename}' and video saved to '{output_filename}'.")

# Function to start live monitoring
def start_live_monitoring():
    # Initialize models
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')

    # Open camera (default webcam)
    cap = cv2.VideoCapture(0)

    # Initialize tracker
    mot_tracker = Sort()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in [2, 3, 5, 7]:  # vehicles
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles using SORT
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Draw bounding boxes around vehicles and plates
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"Plate Detected", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the result
        cv2.imshow("Live Monitoring", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create the main GUI window
root = tk.Tk()
root.title("Automatic License Plate Recognition")
root.geometry("800x500")
root.configure(bg="#F5F5F5")

# Add a modern header with gradient
header_frame = tk.Frame(root, bg="#4A90E2", height=100, relief="ridge")
header_frame.pack(fill="x")

header_label = tk.Label(
    header_frame,
    text="Automatic License Plate Recognition",
    bg="#4A90E2",
    fg="white",
    font=("Helvetica", 24, "bold"),
)
header_label.place(relx=0.5, rely=0.5, anchor="center")

# Add a card-style container for main content
card_frame = tk.Frame(root, bg="white", bd=0, relief="raised")
card_frame.place(relx=0.5, rely=0.5, anchor="center", width=650, height=350)

# Add a title inside the card
card_title_label = tk.Label(
    card_frame,
    text="Analyze Vehicle License Plates",
    bg="white",
    fg="#333333",
    font=("Helvetica", 18, "bold"),
)
card_title_label.pack(pady=10)

# Add a description label
description_label = tk.Label(
    card_frame,
    text="Upload a video or select live monitoring to analyze vehicle license plates.\nResults include detected plates and tracking data.",
    bg="white",
    fg="#666666",
    font=("Helvetica", 12),
    wraplength=600,
    justify="center",
)
description_label.pack(pady=10)

# Add a drop-down menu for selecting mode
mode_label = tk.Label(
    card_frame,
    text="Select Mode",
    bg="white",
    fg="#666666",
    font=("Helvetica", 12),
)
mode_label.pack(pady=10)

mode_combobox = ttk.Combobox(
    card_frame,
    values=["Process Video", "Start Live Monitoring"],
    state="readonly",
    width=30
)
mode_combobox.pack(pady=10)

# Add a button to confirm selection
select_button = ttk.Button(
    card_frame,
    text="Confirm Selection",
    command=select_option
)
select_button.pack(pady=20)

# Add a footer for branding or credits
footer_label = tk.Label(
    root,
    text="© 2025 License Plate Recognition | Powered by YOLOv8",
    bg="#F5F5F5",
    fg="#999999",
    font=("Helvetica", 10),
)
footer_label.pack(side="bottom", pady=10)

# Run the GUI event loop
root.mainloop()