# Fall Detection System

A real-time fall detection system using YOLOv12 and MediaPipe for elderly care facilities. This system can detect and classify four types of falls: step and fall, slip and fall, trip and fall, and stump and fall.

## Features

- **Real-time Detection**: Process video feeds from webcams or video files in real-time
- **Fall Type Classification**: Distinguishes between four types of falls
- **Dashboard Interface**: User-friendly dashboard for monitoring and statistics
- **Multi-person Detection**: Can track and analyze multiple people simultaneously
- **Critical Point Analysis**: Monitors key body points (shoulders, hips, feet) to accurately assess falls
- **Visualization**: Shows pose estimation, body tracking, and alerts
- **Fall Statistics**: Tracks and analyzes falls with detailed statistics
- **Command Line Mode**: Also supports a headless command-line mode for deployments

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- MediaPipe
- Ultralytics (YOLOv12)
- Tkinter (for dashboard mode)
- Matplotlib
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/fall-detection-system.git
   cd fall-detection-system
   ```

2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

3. Download YOLOv12 model (automatic on first run, or manually place in project root)

## Usage

### Dashboard Mode (GUI)

```
python fall_detection_system.py --mode dashboard --model yolov12n1.pt --source 0
```

- `--model`: Path to YOLOv12 model file (default: yolov12n1.pt)
- `--source`: Video source (0 for webcam, or path to video file)
- `--conf`: Detection confidence threshold (0-1, default: 0.5)

### Command Line Mode

```
python fall_detection_system.py --mode cli --model yolov12n1.pt --source 0 --save-falls
```

Additional CLI options:
- `--fall-threshold`: Threshold for fall detection sensitivity (0-1, default: 0.4)
- `--angle-threshold`: Threshold for body angle in degrees (0-90, default: 45)
- `--save-falls`: Save frames when falls are detected
- `--output-dir`: Directory to save fall snapshots

## Project Structure

```
fall-detection-system/
├── models/
│   └── fall_detector.py        # Core fall detection model
├── dashboard/
│   └── dashboard_app.py        # Dashboard UI application
├── utils/
│   └── utils.py                # Utility functions
├── fall_detection_system.py    # Main application script
├── README.md                   # Project documentation
└── requirements.txt            # Project dependencies
```

## How It Works

The Fall Detection System uses a two-stage detection process:

1. **Person Detection**: YOLOv12 model identifies and locates people in the video frame
2. **Pose Estimation**: MediaPipe analyzes body posture to identify falls
3. **Fall Classification**: Algorithm classifies falls into four categories based on movement patterns
4. **Alert Generation**: Visual alerts when falls are detected

### Fall Classification Criteria

- **Step and Fall**: Gradual angle change with moderate velocity
- **Slip and Fall**: Rapid angle change with high velocity
- **Trip and Fall**: Forward momentum with moderate angle change
- **Stump and Fall**: Minimal horizontal movement with significant vertical drop

## Future Enhancements

- Email notification system for alerting caregivers
- Integration with mobile applications
- Support for distributed camera networks
- Fall risk prediction based on movement patterns
- Cloud integration for remote monitoring

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLO by Ultralytics (https://github.com/ultralytics/ultralytics)
- MediaPipe by Google (https://mediapipe.dev/) 