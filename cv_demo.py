"""
Smart Parking POC - Computer Vision Feasibility Demo

This script demonstrates a feasibility proof for computer vision-based
vehicle detection and license plate OCR. It is designed to work on
generic or synthetic images and does NOT use any proprietary Megaworld data.

Modes of operation:
- REAL mode: If ultralytics (YOLOv8) and pytesseract are installed, it runs
  actual detection and OCR models.
- SIMULATED mode: If dependencies are missing, it falls back to generating
  realistic mock results to demonstrate the data flow and matching concepts.

Output:
- Generates synthetic parking lot images.
- Saves annotated images with bounding boxes to `cv-demo/annotated/`.
- Saves structured results to `cv-demo/matching_results.json`.
"""

import os
import json
import random
import sys

# Add current directory to path so we can import matcher
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import matcher
except ImportError:
    print("Warning: matcher.py not found. Please ensure it exists in the project root.")
    matcher = None

# Attempt to import dependencies
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV (cv2) and numpy are required to run this demo. Please install them.")
    sys.exit(1)

try:
    from ultralytics import YOLO
    import pytesseract
    REAL_MODE_AVAILABLE = True
except ImportError:
    REAL_MODE_AVAILABLE = False

# Ensure directories exist
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cv-demo")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")
os.makedirs(ANNOTATED_DIR, exist_ok=True)


def generate_synthetic_image(scenario_index):
    """
    Generates a synthetic parking lot image using OpenCV.
    Returns the image and ground truth bounding boxes.
    """
    img = np.ones((600, 800, 3), dtype=np.uint8) * 200  # Gray background
    
    # Draw parking lines
    cv2.line(img, (200, 100), (200, 500), (255, 255, 255), 3)
    cv2.line(img, (400, 100), (400, 500), (255, 255, 255), 3)
    cv2.line(img, (600, 100), (600, 500), (255, 255, 255), 3)
    
    vehicles = []
    if scenario_index == 1:
        # Scenario 1: One clear vehicle
        cv2.rectangle(img, (220, 200), (380, 400), (0, 0, 255), -1)
        vehicles.append({"bbox": [220, 200, 380, 400], "plate_gt": "ABC1234", "scenario": "confident match"})
    elif scenario_index == 2:
        # Scenario 2: Two vehicles, one clear, one blurry/partially obscured
        cv2.rectangle(img, (220, 250), (380, 450), (255, 0, 0), -1)
        cv2.rectangle(img, (420, 150), (580, 350), (0, 255, 0), -1)
        vehicles.append({"bbox": [220, 250, 380, 450], "plate_gt": "XYZ9876", "scenario": "below-threshold non-match"})
        vehicles.append({"bbox": [420, 150, 580, 350], "plate_gt": "LMN456", "scenario": "confident match"})
    else:
        # Scenario 3: Near-tie match scenario
        cv2.rectangle(img, (420, 200), (580, 400), (0, 255, 255), -1)
        vehicles.append({"bbox": [420, 200, 580, 400], "plate_gt": "DEF5678", "scenario": "near-tie"})

    # Add some text to simulate plates on the drawn vehicles
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        cv2.putText(img, v["plate_gt"], (x1 + 20, y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    return img, vehicles


def run_real_mode(image, vehicles, candidate_pool):
    """
    Runs actual YOLOv8 and Tesseract.
    """
    results = []
    # Load model (will download if not present)
    model = YOLO("yolov8n.pt")
    
    # Run inference
    detections = model(image)[0]
    
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        # Class 2 is car in COCO
        if cls == 2 and conf > 0.5:
            # Crop region
            cropped = image[y1:y2, x1:x2]
            
            # Simple thresholding for OCR
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Run OCR
            ocr_text = pytesseract.image_to_string(thresh, config='--psm 8').strip()
            
            # Simulate char confidences (Tesseract doesn't give them easily without complex parsing)
            char_confidences = [random.uniform(0.7, 0.99) for _ in ocr_text] if ocr_text else []
            confidence_avg = sum(char_confidences)/len(char_confidences) if char_confidences else 0.0
            
            # Match
            match_res = None
            if matcher:
                match_res = matcher.match_plate(ocr_text, char_confidences, candidate_pool)
            
            results.append({
                "ocr_text": ocr_text,
                "confidence_avg": confidence_avg,
                "char_confidences": char_confidences,
                "region_description": f"bbox: {x1},{y1},{x2},{y2}",
                "matching_result": match_res
            })
            
            # Annotate
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, ocr_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    return results, image


def run_simulated_mode(image, vehicles, candidate_pool):
    """
    Simulates CV operations with realistic mock outputs based on the scenarios.
    """
    results = []
    
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        gt_plate = v["plate_gt"]
        scenario = v["scenario"]
        
        # Simulate OCR read with some noise based on scenario
        if scenario == "confident match":
            ocr_text = gt_plate
            char_confidences = [random.uniform(0.85, 0.99) for _ in ocr_text]
        elif scenario == "below-threshold non-match":
            # Very noisy read
            ocr_text = gt_plate[:2] + "8" + gt_plate[3:] # minor error
            char_confidences = [random.uniform(0.4, 0.7) for _ in ocr_text]
        elif scenario == "near-tie":
            ocr_text = gt_plate[:3] + "5" + gt_plate[4:] # single character error
            char_confidences = [random.uniform(0.75, 0.95) for _ in ocr_text]
        else:
            ocr_text = gt_plate
            char_confidences = [0.9] * len(ocr_text)
            
        confidence_avg = sum(char_confidences) / len(char_confidences) if char_confidences else 0.0
        
        match_res = None
        if matcher:
            match_res = matcher.match_plate(ocr_text, char_confidences, candidate_pool)
            
        results.append({
            "ocr_text": ocr_text,
            "confidence_avg": confidence_avg,
            "char_confidences": char_confidences,
            "region_description": f"bbox: {x1},{y1},{x2},{y2}",
            "matching_result": match_res
        })
        
        # Annotate
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, ocr_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    return results, image

def main():
    print("--- Smart Parking CV Feasibility Demo ---")
    
    # Define a candidate pool for matcher
    candidate_pool = [
        {"ticket_id": "T001", "plate": "ABC1234"},
        {"ticket_id": "T002", "plate": "XYZ1111"},
        {"ticket_id": "T003", "plate": "DEF5679"}, # near-tie candidate
        {"ticket_id": "T004", "plate": "DEF5670"}, # near-tie candidate 2
        {"ticket_id": "T005", "plate": "LMN456"}
    ]
    
    mode = "real" if REAL_MODE_AVAILABLE else "simulated"
    print(f"Running in {mode.upper()} mode...")
    
    all_results = []
    
    for i in range(1, 4):
        filename = f"scenario_{i}.jpg"
        filepath = os.path.join(ANNOTATED_DIR, filename)
        
        # Generate image
        img, vehicles = generate_synthetic_image(i)
        
        if mode == "real":
            plates_read, annotated_img = run_real_mode(img, vehicles, candidate_pool)
            notes = "Real YOLOv8 detection and Tesseract OCR."
        else:
            plates_read, annotated_img = run_simulated_mode(img, vehicles, candidate_pool)
            notes = [v["scenario"] for v in vehicles]
            notes_str = ", ".join(notes) if notes else "No vehicles"
            notes = f"Simulated mode scenarios: {notes_str}"
            
        # Save image
        cv2.imwrite(filepath, annotated_img)
        print(f"Saved annotated image to {filepath}")
        
        # Record results
        all_results.append({
            "image_file": filename,
            "detection_count": len(plates_read),
            "plates_read": plates_read,
            "matching_result": plates_read[0]["matching_result"] if plates_read else None, # The prompt requested matching_result at the top level of the array entry, but since there could be multiple detections, let's keep it here or aggregate it.
            "mode": mode,
            "notes": f"Scenario {i}: {notes}"
        })
        
    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "matching_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Results saved to {json_path}")
    print("Demo complete.")

if __name__ == "__main__":
    main()
