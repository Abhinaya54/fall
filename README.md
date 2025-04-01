# Human Fall Detection Using YOLOv12 and MediaPipe

![Program Preview](https://imgur.com/a/ClCcBmN)

A real-time fall detection system using computer vision with YOLOv12 for person detection and MediaPipe for pose estimation. The system features an intuitive dashboard for monitoring and analyzing falls.

## Features

- **Real-time Detection**: Process video feeds from webcams or video files in real-time
- **Fall Type Classification**: Distinguishes between four types of falls: step and fall, slip and fall, trip and fall, and stump and fall
- **Modern PyQt5 Dashboard**: User-friendly interface with statistics, charts, and history tracking
- **Multi-person Detection**: Can track and analyze multiple people simultaneously
- **Advanced Pose Analysis**: Monitors key body points (shoulders, hips, feet) to accurately assess falls
- **Adjustable Sensitivity**: Fine-tune detection thresholds for different environments
- **Intelligent Fall Counting**: Prevents duplicate counts with person tracking and cooldown periods
- **Automatic Fall Snapshot**: Captures and stores images when falls are detected
- **Sound Alerts**: Audio notifications when falls are detected
- **Command Line Mode**: Also supports a headless command-line mode for deployments

## Requirements

- Python 3.7-3.10 (Python 3.11+ may have compatibility issues with some dependencies)
- PyTorch
- OpenCV
- MediaPipe
- Ultralytics (YOLOv12)
- PyQt5
- Matplotlib
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/itSkyyledc/Human-Fall-Detection-Yolov12-Mediapipe.git
   cd Human-Fall-Detection-Yolov12-Mediapipe
   ```

2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

3. Install PyQt5 (if not included in requirements):
   ```
   pip install PyQt5==5.15.10 PyQt5-Qt5==5.15.2 PyQt5-sip==12.13.0 matplotlib
   ```

4. Download YOLOv12 model (automatic on first run, or manually place in project root)

## Usage

### Dashboard Mode (Recommended)

```
python fall_detection_system.py --mode dashboard
```

### Command Line Mode

```
python fall_detection_system.py --mode cli --source 0
```

### Options

- `--mode`: Application mode (`dashboard` or `cli`)
- `--model`: Path to YOLOv12 model file (default: `yolov12n1.pt`)
- `--conf`: Detection confidence threshold (0-1)
- `--source`: Video source (0 for webcam, or path to video file)
- `--fall-threshold`: Threshold for fall detection sensitivity (0-1)
- `--angle-threshold`: Threshold for body angle in degrees (0-90)
- `--save-falls`: Save frames when falls are detected
- `--output-dir`: Directory to save fall snapshots

## System Architecture

The system consists of several key components:

1. **YOLOv12 Person Detection**: Identifies people in the video frames
2. **MediaPipe Pose Estimation**: Extracts skeletal data from detected persons
3. **Fall Detection Algorithm**: Analyzes pose data to detect falls with these criteria:
   - Body angle relative to vertical
   - Body aspect ratio (horizontal vs. vertical orientation)
   - Sudden vertical position changes
   - Motion pattern analysis
4. **Person Tracking**: Follows individuals across frames to maintain identity
5. **PyQt5 Dashboard**: Provides real-time monitoring and analytics

## Fall Classification Criteria

- **Step and Fall**: Gradual angle change with moderate velocity
- **Slip and Fall**: Rapid angle change with high velocity
- **Trip and Fall**: Forward momentum with moderate angle change
- **Stump and Fall**: Minimal horizontal movement with significant vertical drop

## Project Structure

```
fall-detection-system/
├── models/
│   └── fall_detector.py        # Core fall detection model
├── dashboard/
│   └── dashboard_app.py        # PyQt5 Dashboard UI application
├── utils/
│   └── utils.py                # Utility functions
├── fall_snapshots/             # Directory for fall images
├── fall_detection_system.py    # Main application script
├── README.md                   # Project documentation
└── requirements.txt            # Project dependencies
```

## Recent Improvements

- Enhanced fall detection algorithm with stricter criteria
- Added cooldown periods to prevent duplicate fall counts
- Improved sensitivity controls for different environments
- Added comprehensive fall type classification
- Fixed critical point detection for more accurate pose analysis
- Improved error handling for camera access and processing
- Optimized snapshot saving to reduce disk usage
- Migrated from Tkinter to PyQt5 for a more modern interface

## Future Enhancements

- Email notification system for alerting caregivers
- Integration with mobile applications
- Support for distributed camera networks
- Fall risk prediction based on movement patterns
- Cloud integration for remote monitoring
- AI-assisted fall risk prediction

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv12 by Ultralytics (https://github.com/ultralytics/ultralytics)
- MediaPipe by Google (https://mediapipe.dev/)
- PyQt5 Dashboard inspiration from modern monitoring applications 
