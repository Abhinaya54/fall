import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import math
import time
import argparse
import os
import torch
import pkg_resources

class SimpleFallDetection:
    def __init__(self, model_path='yolov12n1.pt', confidence=0.5):
        """Initialize the fall detection system.
        
        Args:
            model_path (str): Path to the YOLOv12 model file (default: 'yolov12.pt')
            confidence (float): Detection confidence threshold (0-1)
        """
        # First verify CUDA is working
        try:
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                self.device = "cuda"
                
                # Force CUDA initialization
                torch.cuda.init()
                x = torch.tensor([1.0], device="cuda")
                print(f"Test tensor device: {x.device}")
            else:
                self.device = "cpu"
                print("WARNING: CUDA not detected despite having CUDA-enabled PyTorch")
                print("Check NVIDIA drivers and CUDA installation")
            
            # Initialize YOLOv12 model with explicit task
            print(f"Loading YOLOv12 model: {model_path}")
            
            # Download the model if needed, but DON'T overwrite model_path
            try:
                from ultralytics.utils.downloads import attempt_download
                downloaded_path = attempt_download(model_path)
                if downloaded_path:
                    print(f"Downloaded model to: {downloaded_path}")
                    model_path = str(downloaded_path)  # Only update if successful
                else:
                    print("Download returned None, using original path")
            except Exception as download_error:
                print(f"Model download error (continuing with original path): {download_error}")
            
            # Load model with explicit task type
            print(f"Using model path: {model_path}")
            self.model = YOLO(model_path, task='detect')
            self.model.to(self.device)
            print(f"Successfully loaded YOLOv12 model on {self.device}")
            
            # Verify model is on correct device
            model_device = next(self.model.parameters()).device
            print(f"Model confirmed on device: {model_device}")
            
            # Person class ID for COCO dataset
            self.person_class_id = 0
            self.confidence = confidence
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Fall detection parameters
        self.fall_threshold = 0.4  # Threshold for fall detection
        self.angle_threshold = 45  # Threshold for body angle
        self.prev_poses = []  # Store previous poses for motion analysis
        self.fall_detected = False
        self.fall_start_time = None
        self.fall_cooldown = 3  # Cooldown in seconds after fall detection
        
    def detect_person(self, frame):
        """Detect persons in the frame using YOLO."""
        # Process frame with explicit device
        try:
            # Convert frame to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if frame.dtype != 'uint8':
                    frame = cv2.convertScaleAbs(frame)
                
            # Run inference with explicit device
            results = self.model(frame, 
                                verbose=False, 
                                conf=self.confidence,
                                device=self.device)
            
            person_boxes = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    if cls == self.person_class_id and conf > self.confidence:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        person_boxes.append((x1, y1, x2, y2))
            
            return person_boxes
        except Exception as e:
            print(f"Error in detect_person: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def analyze_pose(self, frame, person_box):
        """Analyze pose for a detected person using MediaPipe.
        
        Args:
            frame: Input frame
            person_box: Person bounding box (x1, y1, x2, y2)
            
        Returns:
            tuple: (landmarks, pose_features) or (None, None) if pose detection fails
        """
        x1, y1, x2, y2 = person_box
        
        # Extract the person from the frame
        person_img = frame[y1:y2, x1:x2]
        
        if person_img.size == 0:
            return None, None
        
        # Convert to RGB for MediaPipe
        rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_img)
        
        if not results.pose_landmarks:
            return None, None
        
        # Extract landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z, landmark.visibility))
        
        # Calculate pose features
        pose_features = self.calculate_pose_features(landmarks)
        
        return landmarks, pose_features
    
    def calculate_pose_features(self, landmarks):
        """Calculate pose features from landmarks.
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            dict: Dictionary of pose features or None if calculation fails
        """
        if not landmarks:
            return None
        
        # Extract key landmarks (indices may vary based on MediaPipe version)
        # Head, shoulders, hips, knees, ankles
        key_points = [0, 11, 12, 23, 24, 25, 26, 27, 28]
        
        # Extract x, y coordinates of key points
        key_landmarks = [landmarks[i][:2] for i in key_points if i < len(landmarks)]
        
        if len(key_landmarks) < len(key_points):
            return None
        
        # Calculate height of the person (vertical distance between head and feet)
        head_y = landmarks[0][1]
        left_ankle_y = landmarks[27][1]
        right_ankle_y = landmarks[28][1]
        ankle_y = (left_ankle_y + right_ankle_y) / 2
        height = abs(ankle_y - head_y)
        
        # Calculate orientation (angle with the vertical)
        # Using the spine (mid-point of shoulders to mid-point of hips)
        mid_shoulder_x = (landmarks[11][0] + landmarks[12][0]) / 2
        mid_shoulder_y = (landmarks[11][1] + landmarks[12][1]) / 2
        mid_hip_x = (landmarks[23][0] + landmarks[24][0]) / 2
        mid_hip_y = (landmarks[23][1] + landmarks[24][1]) / 2
        
        dx = mid_hip_x - mid_shoulder_x
        dy = mid_hip_y - mid_shoulder_y
        
        angle = math.degrees(math.atan2(dx, dy))  # Angle with vertical axis
        
        # Calculate velocity features if we have previous poses
        velocity = 0
        if self.prev_poses:
            prev_mid_shoulder_y = self.prev_poses[-1]["mid_shoulder_y"]
            time_diff = 1  # Assuming consistent frame rate for simplicity
            velocity = (mid_shoulder_y - prev_mid_shoulder_y) / time_diff
        
        features = {
            "height": height,
            "angle": angle,
            "velocity": velocity,
            "mid_shoulder_y": mid_shoulder_y,
            "mid_shoulder_x": mid_shoulder_x,
            "mid_hip_y": mid_hip_y,
            "mid_hip_x": mid_hip_x
        }
        
        # Keep track of previous poses, limit to 10 for memory efficiency
        self.prev_poses.append(features)
        if len(self.prev_poses) > 10:
            self.prev_poses.pop(0)
        
        return features
    
    def detect_fall(self, pose_features):
        """Detect if a person has fallen based on pose features.
        
        Args:
            pose_features: Dictionary of pose features
            
        Returns:
            bool: True if fall is detected, False otherwise
        """
        if not pose_features:
            return False
        
        # Fall detection logic:
        # 1. Large angle with vertical axis (person not upright)
        # 2. Sudden change in vertical position
        # 3. Change in body proportions
        
        is_fall = False
        angle = abs(pose_features["angle"])
        
        # Check if angle with vertical is large (person not upright)
        if angle > self.angle_threshold:
            # If we have enough previous poses to detect sudden movements
            if len(self.prev_poses) > 5:
                # Calculate velocity over last few frames
                current_y = pose_features["mid_shoulder_y"]
                prev_y = self.prev_poses[-6]["mid_shoulder_y"]
                
                # If there's been a significant vertical movement downward
                if current_y - prev_y > self.fall_threshold:
                    is_fall = True
        
        current_time = time.time()
        
        # Handle fall detection state and cooldown
        if is_fall and not self.fall_detected:
            self.fall_detected = True
            self.fall_start_time = current_time
            return True
        elif self.fall_detected:
            # Check if cooldown period has passed
            if current_time - self.fall_start_time > self.fall_cooldown:
                self.fall_detected = False
            return self.fall_detected
        
        return False
    
    def process_frame(self, frame):
        """Process a single frame for fall detection.
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (output_frame, fall_detected)
        """
        # Make a copy to avoid modifying the original
        output_frame = frame.copy()
        
        # Detect persons in the frame
        person_boxes = self.detect_person(frame)
        
        falls_detected = False
        
        # Process each detected person
        for box in person_boxes:
            x1, y1, x2, y2 = box
            
            # Draw the person bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Analyze pose
            landmarks, pose_features = self.analyze_pose(frame, box)
            
            if landmarks and pose_features:
                # Draw the pose on the cropped image
                h, w = y2 - y1, x2 - x1
                annotated_img = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Draw landmarks - updated for compatibility with newer MediaPipe versions
                person_img = frame[y1:y2, x1:x2]
                if person_img.size > 0:
                    # Convert landmarks to pixel coordinates for the ROI
                    pixel_landmarks = []
                    for lm in landmarks:
                        x, y = int(lm[0] * w), int(lm[1] * h)
                        pixel_landmarks.append((x, y))
                    
                    # Draw connections between landmarks
                    connections = self.mp_pose.POSE_CONNECTIONS
                    for connection in connections:
                        start_idx, end_idx = connection
                        if (start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks) and 
                            0 <= pixel_landmarks[start_idx][0] < w and 
                            0 <= pixel_landmarks[start_idx][1] < h and
                            0 <= pixel_landmarks[end_idx][0] < w and 
                            0 <= pixel_landmarks[end_idx][1] < h):
                            
                            cv2.line(annotated_img, 
                                    pixel_landmarks[start_idx], 
                                    pixel_landmarks[end_idx], 
                                    (245, 66, 230), 2)
                    
                    # Draw landmark points
                    for point in pixel_landmarks:
                        if 0 <= point[0] < w and 0 <= point[1] < h:
                            cv2.circle(annotated_img, point, 4, (245, 117, 66), -1)
                    
                    # Overlay pose drawing on the region of interest in the output frame
                    mask = annotated_img > 0
                    roi = output_frame[y1:y2, x1:x2]
                    if roi.shape[:2] == mask[:,:,0].shape:  # Ensure shapes match
                        roi[mask[:,:,0]] = annotated_img[mask[:,:,0]]
                
                # Check for fall
                if self.detect_fall(pose_features):
                    cv2.putText(
                        output_frame, 
                        "FALL DETECTED", 
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (0, 0, 255), 
                        2
                    )
                    falls_detected = True
        
        return output_frame, falls_detected

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple Fall Detection System')
    parser.add_argument('--model', type=str, default='yolov12n1.pt', help='Path to YOLOv12 model (default: yolov12n.pt)')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--conf', type=float, default=0.5, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found in current directory")
        print("Available model files:")
        for file in os.listdir('.'):
            if file.endswith('.pt'):
                print(f"- {file}")
        return
    
    # Initialize the fall detection system
    try:
        fall_detector = SimpleFallDetection(model_path=args.model, confidence=args.conf)
        print(f"Successfully initialized fall detector with model: {args.model}")
    except Exception as e:
        print(f"Error initializing fall detector: {e}")
        return
    
    # Open video capture
    if args.source.isdigit():
        source_id = int(args.source)
        print(f"Opening webcam with ID: {source_id}")
        cap = cv2.VideoCapture(source_id, cv2.CAP_DSHOW)  # Try DSHOW backend on Windows
    else:
        print(f"Opening video file: {args.source}")
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        
        # List available cameras
        print("Attempting to find available cameras:")
        for i in range(5):  # Check first 5 camera indices
            test_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if test_cap.isOpened():
                print(f"Camera {i} is available")
                test_cap.release()
            else:
                print(f"Camera {i} is not available")
        
        return
    
    # Print camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened with resolution: {width}x{height}, FPS: {fps}")
    
    # Initialize FPS calculation
    frame_count = 0
    start_time = time.time()
    
    print("Starting video processing. Press 'q' to quit.")
    
    # Try to read the first frame with a retry mechanism
    max_retries = 5
    retries = 0
    has_frame = False
    
    while retries < max_retries and not has_frame:
        ret, frame = cap.read()
        if ret:
            has_frame = True
            print("Successfully read first frame from camera")
        else:
            retries += 1
            print(f"Failed to read frame, retry {retries}/{max_retries}")
            time.sleep(1)  # Wait a bit between retries
    
    if not has_frame:
        print("Could not read any frames from the camera after multiple retries.")
        cap.release()
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive frame. Exiting...")
            break
        
        frame_count += 1
        
        # Process the frame
        output_frame, fall_detected = fall_detector.process_frame(frame)
        
        # Display status
        status = "Status: FALL DETECTED" if fall_detected else "Status: Normal"
        cv2.putText(
            output_frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if fall_detected else (0, 255, 0),
            2
        )
        
        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            current_fps = frame_count / elapsed_time
            cv2.putText(
                output_frame,
                f"FPS: {current_fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        
        # Show the frame
        cv2.imshow('Fall Detection', output_frame)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    print("Program ended.")

if __name__ == "__main__":
    main() 
