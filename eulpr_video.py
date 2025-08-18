import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image
from huggingface_hub import hf_hub_download
import os
import warnings
from collections import defaultdict, deque
import time
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class PlateDetection:
    """Data class for storing plate detection information"""
    text: str
    ocr_confidence: float
    detection_confidence: float
    bbox: List[int]
    frame_number: int
    timestamp: float

class PlateTracker:
    """Track licence plates across video frames to improve accuracy"""
    
    def __init__(self, max_history: int = 10, similarity_threshold: float = 0.8):
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold
        self.plate_history = defaultdict(lambda: deque(maxlen=max_history))
        
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple character matching"""
        if not text1 or not text2:
            return 0.0
        
        # Remove spaces and convert to uppercase
        text1 = text1.replace(" ", "").upper()
        text2 = text2.replace(" ", "").upper()
        
        if text1 == text2:
            return 1.0
        
        # Calculate character-level similarity
        matches = sum(c1 == c2 for c1, c2 in zip(text1, text2))
        max_len = max(len(text1), len(text2))
        return matches / max_len if max_len > 0 else 0.0
    
    def find_best_match(self, new_text: str, bbox: List[int]) -> Optional[str]:
        """Find the best matching plate ID based on text similarity and position"""
        best_match = None
        best_similarity = 0.0
        
        for plate_id, history in self.plate_history.items():
            if not history:
                continue
                
            # Get the most recent detection for this plate
            recent_detection = history[-1]
            
            # Calculate text similarity
            text_similarity = self.calculate_similarity(new_text, recent_detection.text)
            
            # Calculate position similarity (simple overlap check)
            position_similarity = self.calculate_position_similarity(bbox, recent_detection.bbox)
            
            # Combined similarity score
            combined_similarity = (text_similarity * 0.7) + (position_similarity * 0.3)
            
            if combined_similarity > best_similarity and combined_similarity > self.similarity_threshold:
                best_similarity = combined_similarity
                best_match = plate_id
        
        return best_match
    
    def calculate_position_similarity(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate position similarity based on bounding box overlap"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def add_detection(self, detection: PlateDetection) -> str:
        """Add a detection and return the plate ID"""
        # Find best matching existing plate
        plate_id = self.find_best_match(detection.text, detection.bbox)
        
        if plate_id is None:
            # Create new plate ID
            plate_id = f"plate_{len(self.plate_history) + 1}_{detection.frame_number}"
        
        self.plate_history[plate_id].append(detection)
        return plate_id
    
    def get_best_reading(self, plate_id: str) -> Optional[PlateDetection]:
        """Get the most confident reading for a plate"""
        if plate_id not in self.plate_history or not self.plate_history[plate_id]:
            return None
        
        # Return detection with highest combined confidence
        best_detection = max(
            self.plate_history[plate_id],
            key=lambda d: (d.ocr_confidence * d.detection_confidence)
        )
        return best_detection

class VideoLicensePlateRecognizer:
    """Main class for video licence plate recognition"""
    
    def __init__(self, model_confidence: float = 0.5, ocr_languages: List[str] = None):
        self.model_confidence = model_confidence
        self.ocr_languages = ocr_languages or ['en', 'de', 'fr', 'es', 'it', 'nl']
        
        # Download and load models
        self._load_models()
        
        # Initialize tracker
        self.tracker = PlateTracker()
        
    def _load_models(self):
        """Load EULPR model"""
        print("Downloading EULPR model from HuggingFace...")
        model_path = hf_hub_download(repo_id="0xnu/european-license-plate-recognition", filename="model.onnx")
        config_path = hf_hub_download(repo_id="0xnu/european-license-plate-recognition", filename="config.json")
        
        # Load EULPR model
        self.yolo_model = YOLO(model_path, task='detect')
        self.ocr_reader = easyocr.Reader(self.ocr_languages, gpu=False, verbose=False)
        print("Models loaded successfully")
    
    def enhance_plate_image(self, plate_crop: np.ndarray) -> np.ndarray:
        """Enhance plate image quality for improved OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        enhanced = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced_rgb
    
    def detect_plates_in_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> List[PlateDetection]:
        """Detect licence plates in a single frame"""
        # Convert colour space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect licence plates using EULPR
        results = self.yolo_model(frame_rgb, conf=self.model_confidence, iou=0.4, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Crop plate with bounds checking
                    h, w = frame_rgb.shape[:2]
                    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
                    
                    if x2 > x1 and y2 > y1:  # Valid crop dimensions
                        plate_crop = frame_rgb[y1:y2, x1:x2]
                        
                        # Extract text only if crop is valid
                        if plate_crop.size > 0:
                            # Enhance image quality for better OCR results
                            plate_crop_enhanced = self.enhance_plate_image(plate_crop)
                            
                            ocr_results = self.ocr_reader.readtext(plate_crop_enhanced)
                            if ocr_results:
                                text = ocr_results[0][1]
                                ocr_confidence = float(ocr_results[0][2])
                                detection_confidence = float(box.conf[0])
                                
                                detection = PlateDetection(
                                    text=text,
                                    ocr_confidence=ocr_confidence,
                                    detection_confidence=detection_confidence,
                                    bbox=[x1, y1, x2, y2],
                                    frame_number=frame_number,
                                    timestamp=timestamp
                                )
                                detections.append(detection)
        
        return detections
    
    def process_video_file(self, video_path: str, output_path: str = None, 
                          frame_skip: int = 1, max_frames: int = None) -> Dict[str, PlateDetection]:
        """
        Process a video file for licence plate recognition
        
        Args:
            video_path (str): Path to input video file
            output_path (str): Path for output video with annotations (optional)
            frame_skip (int): Process every nth frame (default: 1)
            max_frames (int): Maximum number of frames to process (optional)
            
        Returns:
            dict: Dictionary of detected plates with their best readings
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path}")
        print(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        
        # Setup video writer if output path is specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if specified
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Check max frames limit
                if max_frames and processed_frames >= max_frames:
                    break
                
                # Calculate timestamp
                timestamp = frame_count / fps
                
                # Detect plates in current frame
                detections = self.detect_plates_in_frame(frame, frame_count, timestamp)
                
                # Add detections to tracker
                for detection in detections:
                    plate_id = self.tracker.add_detection(detection)
                    print(f"Frame {frame_count}: Detected '{detection.text}' (ID: {plate_id})")
                
                # Draw annotations if output video is requested
                if output_path:
                    annotated_frame = self._annotate_frame(frame, detections)
                    out.write(annotated_frame)
                
                frame_count += 1
                processed_frames += 1
                
                # Progress update
                if processed_frames % 30 == 0:
                    progress = (processed_frames * frame_skip / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({processed_frames} frames processed)")
        
        finally:
            cap.release()
            if output_path:
                out.release()
        
        print(f"Video processing complete. Processed {processed_frames} frames.")
        
        # Return best readings for each detected plate
        results = {}
        for plate_id in self.tracker.plate_history:
            best_reading = self.tracker.get_best_reading(plate_id)
            if best_reading:
                results[plate_id] = best_reading
        
        return results
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[PlateDetection]) -> np.ndarray:
        """Annotate frame with detection results"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw text
            label = f"{detection.text} ({detection.ocr_confidence:.2f})"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame
    
    def process_live_camera(self, camera_index: int = 0, display: bool = True):
        """
        Process live camera feed for real-time licence plate recognition
        
        Args:
            camera_index (int): Camera index (default: 0)
            display (bool): Whether to display the video feed
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_index}")
        
        frame_count = 0
        fps_counter = time.time()
        
        print("Starting live camera feed. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = time.time()
                
                # Process every 3rd frame to maintain performance
                if frame_count % 3 == 0:
                    detections = self.detect_plates_in_frame(frame, frame_count, timestamp)
                    
                    for detection in detections:
                        plate_id = self.tracker.add_detection(detection)
                        print(f"Live detection: '{detection.text}' (Confidence: {detection.ocr_confidence:.2f})")
                
                if display:
                    # Draw current detections
                    if frame_count % 3 == 0:
                        frame = self._annotate_frame(frame, detections)
                    
                    # Display FPS
                    if time.time() - fps_counter > 1.0:
                        fps = frame_count / (time.time() - fps_counter + 1e-6)
                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    cv2.imshow('Live Licence Plate Recognition', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

def main():
    """Example usage of the video licence plate recognizer"""
    recognizer = VideoLicensePlateRecognizer()
    
    # Example 1: Process a video file
    video_path = './examples/paris_la_grande_arche.mp4'
    if os.path.exists(video_path):
        results = recognizer.process_video_file(
            video_path, 
            output_path='./examples/output_video.mp4',
            frame_skip=2  # Process every 2nd frame for performance
        )
        
        print("\nFinal Results:")
        for plate_id, detection in results.items():
            print(f"{plate_id}: '{detection.text}' "
                  f"(OCR: {detection.ocr_confidence:.2f}, "
                  f"Detection: {detection.detection_confidence:.2f}, "
                  f"Frame: {detection.frame_number})")
    
    # Example 2: Process live camera (uncomment to use)
    # try:
    #     recognizer.process_live_camera(camera_index=0, display=True)
    # except ValueError as e:
    #     print(f"Camera error: {e}")

if __name__ == "__main__":
    main()