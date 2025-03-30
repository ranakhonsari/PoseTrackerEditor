import cv2
import numpy as np
from ultralytics import YOLO

# Define keypoint indices
right_shoulder_idx = 6
right_elbow_idx = 8
def run_tracker(model_name, source):
    """
    Run YOLO pose tracking on a video source (webcam in this case).
    """
    model = YOLO(model_name)
    cap = cv2.VideoCapture(source)
    
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
                    # Draw a line from shoulder to elbow
                    cv2.line(frame, tuple(right_shoulder.astype(int)), tuple(right_elbow.astype(int)), (0, 0, 255), 3)
        # Display the frame
        cv2.imshow("Webcam Pose Tracking", results[0].plot())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam tracking stopped.")

# Run the tracker on webcam
run_tracker("yolo11n-pose.pt", 0)
