import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image
from huggingface_hub import hf_hub_download
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Download EULPR model from HuggingFace
print("Downloading EULPR model from HuggingFace...")
model_path = hf_hub_download(repo_id="0xnu/european-license-plate-recognition", filename="model.onnx")
config_path = hf_hub_download(repo_id="0xnu/european-license-plate-recognition", filename="config.json")

# Load EULPR model with explicit task specification
yolo_model = YOLO(model_path, task='detect')
ocr_reader = easyocr.Reader(['en', 'de', 'fr', 'es', 'it', 'nl'], gpu=False, verbose=False)

def recognize_license_plate(image_path):
    """
    Recognise European licence plates using EULPR detection and EasyOCR text extraction.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        list: List of dictionaries containing detected plate text and confidence scores
    """
    # Validate file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load and validate image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image file: {image_path}")
    
    # Convert colour space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect licence plates using EULPR
    results = yolo_model(image_rgb, conf=0.5, iou=0.4, verbose=False)
    
    plates = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Crop plate with bounds checking
                h, w = image_rgb.shape[:2]
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
                
                if x2 > x1 and y2 > y1:  # Valid crop dimensions
                    plate_crop = image_rgb[y1:y2, x1:x2]
                    
                    # Extract text only if crop is valid
                    if plate_crop.size > 0:
                        # Enhance image quality for better OCR results
                        plate_crop_enhanced = enhance_plate_image(plate_crop)
                        
                        ocr_results = ocr_reader.readtext(plate_crop_enhanced)
                        if ocr_results:
                            text = ocr_results[0][1]
                            confidence = float(ocr_results[0][2])
                            detection_confidence = float(box.conf[0])
                            
                            plates.append({
                                'text': text,
                                'ocr_confidence': confidence,
                                'detection_confidence': detection_confidence,
                                'bbox': [x1, y1, x2, y2]
                            })
    
    return plates

def enhance_plate_image(plate_crop):
    """
    Enhance plate image quality for improved OCR accuracy.
    
    Args:
        plate_crop (np.ndarray): Cropped plate image
        
    Returns:
        np.ndarray: Enhanced plate image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding
    enhanced = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced_rgb

def process_multiple_images(image_directory):
    """
    Process multiple images in a directory for licence plate recognition.
    
    Args:
        image_directory (str): Path to directory containing images
        
    Returns:
        dict: Results for each processed image
    """
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    results_dict = {}
    
    if not os.path.exists(image_directory):
        print(f"Directory not found: {image_directory}")
        return results_dict
    
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(supported_formats)]
    
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        try:
            results = recognize_license_plate(image_path)
            results_dict[image_file] = results
            print(f"Processed {image_file}: {len(results)} plates detected")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            results_dict[image_file] = []
    
    return results_dict

# Create examples directory if it doesn't exist
os.makedirs('./examples', exist_ok=True)

# Process single image
image_path = './examples/poland_car.jpeg'
if os.path.exists(image_path):
    try:
        results = recognize_license_plate(image_path)
        print("Detection Results:")
        for i, plate in enumerate(results):
            print(f"Plate {i+1}: {plate['text']} (OCR: {plate['ocr_confidence']:.2f}, Detection: {plate['detection_confidence']:.2f})")
    except Exception as e:
        print(f"Error processing image: {e}")
else:
    print(f"Please ensure the image file exists at: {image_path}")
    print("Current working directory:", os.getcwd())
    print("Contents of examples directory:", os.listdir('./examples') if os.path.exists('./examples') else "Directory doesn't exist")

# Optional: Process all images in examples directory
# batch_results = process_multiple_images('./examples')
# print("\nBatch Processing Results:")
# for filename, plates in batch_results.items():
#     print(f"{filename}: {len(plates)} plates detected")