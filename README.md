# Violin Pose Analysis 

A computer vision system for analyzing violinists' posture and bowing technique using 2D pose estimation.

## Features

### 1. Basic Pose Tracking
- **`pose_webcam.py`** & **`pose_video.py`**:
  - Real-time 2D keypoint tracking from webcam or video input
  - Highlights right arm in red when elbow rises above shoulder

### 2. Advanced Bowing Analysis
- **`motion_track.py`**:
  - Tracks right wrist trajectory (plotted in red)
  - Visualizes arm position (purple lines)
  - Detects bowing directions:
    - **Down-bow**: Rightward wrist movement (X-coordinate increases)
    - **Up-bow**: Leftward wrist movement (X-coordinate decreases)
  - 50-pixel movement threshold to ignore minor movements

## Technical Details
- **Pose Estimation**: YOLO11 model (`yolo11n-pose.pt`)
- **Key Features**:
  - Frame-by-frame trajectory visualization
  - Direction change detection
  - Configurable sensitivity thresholds
- **Output**: Annotated video with:
  - Real-time bow direction labels
  - Wrist trajectory overlay
  - Posture highlights

