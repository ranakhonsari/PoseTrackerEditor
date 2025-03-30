import cv2
import numpy as np
from ultralytics import YOLO

# Define keypoint indices
right_shoulder_idx = 6
right_elbow_idx = 8

def run_tracker(model_name, input_video, output_video):
    """
    Run YOLO pose tracking on a video and save the output to a file.
    """
    model = YOLO(model_name)
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define video writer to save output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO pose tracking
        results = model.track(frame, persist=True)

        if results and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()  # Extract keypoints
            
            for kp in keypoints:
                right_shoulder = kp[right_shoulder_idx]
                right_elbow = kp[right_elbow_idx]

                if right_elbow[1] < right_shoulder[1]:  # If elbow is above the shoulder
                    # Draw a red line from shoulder to elbow
                    cv2.line(frame, tuple(right_shoulder.astype(int)), tuple(right_elbow.astype(int)), (0, 0, 255), 3)

        # Write the frame to the output video
        out.write(results[0].plot())

    # Release resources
    cap.release()
    out.release()
    print(f"Processing finished. Video saved as {output_video}")

# Run the tracker on a video file and save output
run_tracker("yolo11n-pose.pt", "violin.mp4", "violin_output_video.mp4")
