import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import deque

# Define keypoint indices
RIGHT_SHOULDER = 6
RIGHT_ELBOW = 8
RIGHT_WRIST = 10

# Configuration
TRAJECTORY_LENGTH = 30  # Number of frames to track
PLOT_EVERY = 10         # Update plot every N frames
MIN_BOWING_DISTANCE = 50  # Minimum pixel movement to consider as bowing

def run_bowing_analysis(model_name, input_video, output_video):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Trajectory tracking
    trajectory = deque(maxlen=TRAJECTORY_LENGTH)
    bowing_directions = []
    current_direction = None
    frame_count = 0
    
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        
        if results and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            
            for kp in keypoints:
                shoulder = kp[RIGHT_SHOULDER]
                elbow = kp[RIGHT_ELBOW]
                wrist = kp[RIGHT_WRIST]
                
                # Store wrist position
                trajectory.append(wrist)
                
                # Draw arm connections
                cv2.line(annotated_frame, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), (186, 110, 210), 2)
                cv2.line(annotated_frame, tuple(elbow.astype(int)), tuple(wrist.astype(int)), (186, 110, 210), 2)
                
                # Draw trajectory
                for i in range(1, len(trajectory)):
                    cv2.line(annotated_frame, 
                            tuple(trajectory[i-1].astype(int)), 
                            tuple(trajectory[i].astype(int)), 
                            (0, 0, 255), 3)
                
                # Detect bowing direction changes
                if len(trajectory) > 1:
                    movement = trajectory[-1] - trajectory[-2]
                    distance = np.linalg.norm(movement)
                    
                    if distance > MIN_BOWING_DISTANCE:
                        direction = "down" if movement[0] > 0 else "up"
                        
                        if current_direction != direction:
                            current_direction = direction
                            bowing_directions.append((frame_count/fps, direction))
                            
                            # Visual feedback for direction change
                            cv2.putText(annotated_frame, f"BOWING {direction.upper()}",
                                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                       (0, 0, 255), 2)
                
                # Update trajectory plot periodically
                if frame_count % PLOT_EVERY == 0 and len(trajectory) > 10:
                    plt.clf()
                    trajectory_array = np.array(trajectory)
                    plt.plot(trajectory_array[:,0], -trajectory_array[:,1], 'b-')
                    plt.title(f"Wrist Trajectory (Frame {frame_count})")
                    plt.pause(0.01)
        
        out.write(annotated_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    

# Run the analysis
run_bowing_analysis("yolo11n-pose.pt", "violonist.mp4", "violinist_motion_track_output.mp4")