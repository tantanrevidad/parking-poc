"""
parking_detector.py — Smart Parking POC
-----------------------------------------
Enterprise Computer Vision Parking Space Occupancy & Vehicle Detection Engine.
Follows the 5-Phase Occupancy Detection Algorithm Architecture:

  Phase 1: Data Preparation & ROI Calibration (slots_config.json + Dual Native Calibration)
  Phase 2: Model Setup & Inference Pipeline (YOLOv8n + COCO Vehicle Filter)
  Phase 3: Spatial Logic & Occupancy Calculation (Intersection over Area - IoA + Ground Contact Reinforcement)
  Phase 4: Temporal Filtering & Edge-Case Handling (Sliding-Window State Debouncing)
  Phase 5: Output Structuring (Standardized JSON Payload with Low-Confidence Flags)
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
from collections import deque

import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPoly, Point as ShapelyPoint, box as ShapelyBox
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "car_dataset")
CONFIG_PATH = os.path.join(PROJECT_DIR, "slots_config.json")

_yolo_model: Optional[YOLO] = None
_slots_config_cache: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Phase 1: Data Preparation & ROI Calibration
# ---------------------------------------------------------------------------

def load_slots_config(force_reload: bool = False) -> Dict[str, Any]:
    """Loads and caches the static parking slots configuration from slots_config.json."""
    global _slots_config_cache
    if _slots_config_cache is None or force_reload:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
                _slots_config_cache = json.load(f)
        else:
            _slots_config_cache = {"camera_angles": {}}
    return _slots_config_cache


def get_slot_rois_for_camera(
    image_name: str,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves calibrated slot ROI polygons for the specified camera perspective.
    Applies resolution scaling automatically if target dimensions differ from calibrated dimensions.
    """
    config = load_slots_config()
    angles = config.get("camera_angles", {})

    target_key = None
    if image_name in angles:
        target_key = image_name
    elif "empty" in image_name.lower() and "empty lot.jpg" in angles:
        target_key = "empty lot.jpg"
    elif "142441" in image_name and "Screenshot 2026-08-17 142441.png" in angles:
        target_key = "Screenshot 2026-08-17 142441.png"
    elif "142454" in image_name and "Screenshot 2026-08-17 142454.png" in angles:
        target_key = "Screenshot 2026-08-17 142454.png"
    elif "142506" in image_name and "Screenshot 2026-08-17 142506.png" in angles:
        target_key = "Screenshot 2026-08-17 142506.png"
    elif "row_sequence" in angles:
        target_key = "row_sequence"

    if not target_key or target_key not in angles:
        return []

    cam_entry = angles[target_key]
    slots = cam_entry.get("slots", [])
    calib_res = cam_entry.get("resolution", [1372, 768])
    calib_w, calib_h = calib_res[0], calib_res[1]

    # If target resolution matches calibration or not provided, return as-is
    if target_width is None or target_height is None:
        return slots
    if target_width == calib_w and target_height == calib_h:
        return slots

    # Scale polygon coordinates proportionally
    scale_x = target_width / float(calib_w)
    scale_y = target_height / float(calib_h)

    scaled_slots = []
    for s in slots:
        scaled_poly = [
            [int(round(pt[0] * scale_x)), int(round(pt[1] * scale_y))]
            for pt in s["polygon"]
        ]
        scaled_s = dict(s)
        scaled_s["polygon"] = scaled_poly
        scaled_slots.append(scaled_s)

    return scaled_slots


# ---------------------------------------------------------------------------
# Phase 2: Model Setup & Inference Pipeline
# ---------------------------------------------------------------------------

def get_detector() -> YOLO:
    """Loads and caches the YOLOv8n object detection model."""
    global _yolo_model
    if _yolo_model is None:
        model_path = os.path.join(PROJECT_DIR, "yolov8n.pt")
        if not os.path.exists(model_path):
            model_path = "yolov8n.pt"
        _yolo_model = YOLO(model_path)
    return _yolo_model


def enhance_low_light(img_bgr: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel
    in LAB color space to illuminate dark vehicles (black SUVs, pickups) and shadow regions.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)


def _nms_boxes(boxes_list: List[Dict[str, Any]], iou_thresh: float = 0.50) -> List[Dict[str, Any]]:
    """Merges overlapping bounding boxes from dual-exposure inference via Non-Maximum Suppression."""
    if not boxes_list:
        return []
    boxes = np.array([b["bbox"] for b in boxes_list], dtype=np.float32)
    scores = np.array([b["confidence"] for b in boxes_list], dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(boxes_list[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def run_vehicle_inference(
    img_bgr: np.ndarray,
    conf_threshold: float = 0.25,
    enable_low_light_boost: bool = True,
) -> List[Dict[str, Any]]:
    """
    Runs YOLOv8n object detection with adaptive dual-pass low-light enhancement.
    COCO Vehicle Classes: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck).
    """
    detector = get_detector()
    raw_results = detector(img_bgr, verbose=False, conf=conf_threshold)[0]
    candidates = []

    for box in raw_results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in (2, 3, 5, 7):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            candidates.append({
                "bbox": [x1, y1, x2, y2],
                "centroid": (cx, cy),
                "bottom_contact_point": (cx, float(y2)),
                "confidence": round(conf, 3),
                "class_id": cls_id,
                "class_name": detector.names.get(cls_id, "vehicle"),
                "shapely_box": ShapelyBox(x1, y1, x2, y2),
            })

    # Second pass with CLAHE contrast enhancement for dark vehicles / low lighting
    if enable_low_light_boost:
        enhanced_bgr = enhance_low_light(img_bgr, clip_limit=2.5)
        enh_results = detector(enhanced_bgr, verbose=False, conf=max(0.18, conf_threshold * 0.85))[0]

        for box in enh_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in (2, 3, 5, 7):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                candidates.append({
                    "bbox": [x1, y1, x2, y2],
                    "centroid": (cx, cy),
                    "bottom_contact_point": (cx, float(y2)),
                    "confidence": round(conf, 3),
                    "class_id": cls_id,
                    "class_name": detector.names.get(cls_id, "vehicle"),
                    "shapely_box": ShapelyBox(x1, y1, x2, y2),
                })

        return _nms_boxes(candidates, iou_thresh=0.55)

    return candidates


# ---------------------------------------------------------------------------
# Phase 3: Spatial Logic & Occupancy Calculation (IoA + Ground Contact)
# ---------------------------------------------------------------------------

def calculate_ioa_occupancy(
    slot_poly_coords: List[List[int]],
    vehicles: List[Dict[str, Any]],
    ioa_threshold: float = 0.30,
) -> Tuple[bool, float, Optional[Dict[str, Any]], bool]:
    """
    Computes the Intersection over Area (IoA) metric with ground-contact reinforcement:
      IoA = Area(Slot Polygon ∩ Vehicle Box) / Area(Slot Polygon)

    Ground-Contact Rule:
      If bottom contact point (cx, y2) falls inside slot polygon and raw IoA >= 0.12:
        effective_ioa = max(effective_ioa, 0.50)

    Returns:
        (is_occupied, effective_ioa, best_matched_vehicle, is_borderline)
    """
    if len(slot_poly_coords) < 3:
        return False, 0.0, None, False

    slot_poly = ShapelyPoly(slot_poly_coords)
    slot_area = slot_poly.area
    if slot_area <= 0:
        return False, 0.0, None, False

    max_ioa = 0.0
    best_veh = None

    for veh in vehicles:
        v_box = veh["shapely_box"]
        if not slot_poly.intersects(v_box):
            continue

        inter_geom = slot_poly.intersection(v_box)
        inter_area = inter_geom.area
        raw_ioa = inter_area / slot_area
        veh_coverage = inter_area / v_box.area if v_box.area > 0 else 0.0

        effective_ioa = raw_ioa

        # Spatial Reinforcement Rules:
        cx, cy = veh["centroid"]
        bc_x, bc_y = veh["bottom_contact_point"]
        centroid_point = ShapelyPoint(cx, cy)
        bc_point = ShapelyPoint(bc_x, bc_y)

        # 1. Centroid containment (vehicle center is inside slot)
        if slot_poly.contains(centroid_point):
            effective_ioa = max(effective_ioa, 0.60)

        # 2. High vehicle coverage (>= 35% of the vehicle is inside the slot)
        elif veh_coverage >= 0.35:
            effective_ioa = max(effective_ioa, 0.55)

        # 3. Ground-contact reinforcement (tire contact line is inside slot)
        elif slot_poly.contains(bc_point) and raw_ioa >= 0.10:
            effective_ioa = max(effective_ioa, 0.50)

        if effective_ioa > max_ioa:
            max_ioa = effective_ioa
            best_veh = veh

    is_occupied = max_ioa >= ioa_threshold
    # Flag borderline calls within +/- 0.05 of threshold
    is_borderline = abs(max_ioa - ioa_threshold) <= 0.05

    return is_occupied, round(min(1.0, max_ioa), 3), best_veh, is_borderline


# ---------------------------------------------------------------------------
# Phase 4: Temporal Filtering & Edge-Case Handling
# ---------------------------------------------------------------------------

class TemporalStateDebouncer:
    """
    Maintains a sliding window buffer of slot states across consecutive frames
    to prevent state flickering from moving vehicles or transient shadows.
    """
    def __init__(self, window_size: int = 5, min_confirmation_ratio: float = 0.60):
        self.window_size = window_size
        self.min_confirmation_ratio = min_confirmation_ratio
        self.history: Dict[str, deque] = {}

    def update_and_smooth(self, slot_id: str, current_occupied: bool) -> bool:
        if slot_id not in self.history:
            self.history[slot_id] = deque(maxlen=self.window_size)
        self.history[slot_id].append(current_occupied)

        occupied_votes = sum(1 for v in self.history[slot_id] if v)
        ratio = occupied_votes / len(self.history[slot_id])
        return ratio >= self.min_confirmation_ratio

    def reset(self):
        self.history.clear()


_global_debouncer = TemporalStateDebouncer(window_size=5)


# ---------------------------------------------------------------------------
# Phase 5: Output Structuring (Standard JSON Payload)
# ---------------------------------------------------------------------------

def generate_standard_json_payload(
    camera_feed: str,
    bay_results: List[Dict[str, Any]],
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates a structured JSON payload conforming to the Phase 5 enterprise schema,
    including low-confidence edge-case flags.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    total_spaces = len(bay_results)
    occupied_count = sum(1 for b in bay_results if b["status"] == "occupied")
    vacant_count = total_spaces - occupied_count
    occ_rate = round(occupied_count / total_spaces, 3) if total_spaces > 0 else 0.0
    borderline_count = sum(1 for b in bay_results if b.get("low_confidence_flag", False))

    slots_payload = []
    for b in bay_results:
        slots_payload.append({
            "id": b["slot_id"],
            "name": b["slot_name"],
            "zone": b["zone"],
            "status": b["status"],
            "occupancy_ratio": b["occupancy_ratio"],
            "confidence": b["confidence"],
            "vehicle_class": b["matched_vehicle_class"],
            "low_confidence_flag": b.get("low_confidence_flag", False),
        })

    return {
        "timestamp": timestamp,
        "camera_feed": camera_feed,
        "total_spaces": total_spaces,
        "occupied_count": occupied_count,
        "vacant_count": vacant_count,
        "occupancy_rate": occ_rate,
        "borderline_count": borderline_count,
        "slots": slots_payload,
    }


# ---------------------------------------------------------------------------
# End-to-End Detection Pipeline
# ---------------------------------------------------------------------------

def detect_parking_spaces(
    img_bgr: np.ndarray,
    image_name: str,
    conf_threshold: float = 0.25,
    ioa_threshold: float = 0.30,
    enable_temporal_smoothing: bool = False,
    enable_low_light_boost: bool = True,
) -> Dict[str, Any]:
    """
    Executes the 5-phase parking space occupancy detection pipeline:
      Phase 1: Retrieve calibrated slot ROI polygons (with automatic resolution scaling).
      Phase 2: Run YOLOv8 vehicle detection (with adaptive low-light dual-exposure boost).
      Phase 3: Compute IoA occupancy ratio per slot with ground-contact reinforcement.
      Phase 4: Optional sliding-window temporal smoothing.
      Phase 5: Format structured JSON output & render visual overlay.
    """
    h, w = img_bgr.shape[:2]

    # Phase 1: Calibrated Slot ROIs (scaled to image resolution)
    slot_defs = get_slot_rois_for_camera(image_name, target_width=w, target_height=h)

    # Phase 2: YOLOv8 Vehicle Detection with Low-Light Boost
    vehicles = run_vehicle_inference(
        img_bgr,
        conf_threshold=conf_threshold,
        enable_low_light_boost=enable_low_light_boost,
    )

    # Phase 3 & 4: Spatial IoA & Temporal Smoothing
    bay_results = []
    occupied_count = 0
    vacant_count = 0

    for s in slot_defs:
        slot_id = s["id"]
        slot_name = s.get("name", slot_id)
        zone = s.get("zone", "General")
        poly_coords = s["polygon"]

        is_occ, ioa_score, matched_veh, is_borderline = calculate_ioa_occupancy(
            poly_coords,
            vehicles,
            ioa_threshold=ioa_threshold,
        )

        if enable_temporal_smoothing:
            is_occ = _global_debouncer.update_and_smooth(slot_id, is_occ)

        status_str = "occupied" if is_occ else "vacant"
        if is_occ:
            occupied_count += 1
        else:
            vacant_count += 1

        # Determine confidence score
        if matched_veh:
            conf = matched_veh["confidence"]
        else:
            conf = round(max(0.75, 1.0 - ioa_score), 3)

        poly_np = np.array(poly_coords, dtype=np.int32)
        bx, by, bw, bh = cv2.boundingRect(poly_np)

        bay_results.append({
            "slot_id": slot_id,
            "slot_name": slot_name,
            "zone": zone,
            "polygon": poly_coords,
            "bbox": [bx, by, bx + bw, by + bh],
            "status": status_str,
            "is_occupied": is_occ,
            "occupancy_ratio": ioa_score,
            "confidence": conf,
            "matched_vehicle_class": matched_veh["class_name"] if matched_veh else None,
            "matched_vehicle_conf": matched_veh["confidence"] if matched_veh else None,
            "low_confidence_flag": is_borderline,
        })

    total_bays = len(bay_results)
    occupancy_rate = (occupied_count / total_bays) if total_bays > 0 else 0.0

    # Phase 5: Structured JSON Output
    json_payload = generate_standard_json_payload(image_name, bay_results)

    # Visual Overlay Rendering
    annotated = img_bgr.copy()
    overlay = img_bgr.copy()

    for b in bay_results:
        poly_np = np.array(b["polygon"], dtype=np.int32)
        if b["is_occupied"]:
            # Translucent Rose Red for Occupied: BGR = (50, 50, 225)
            cv2.fillPoly(overlay, [poly_np], (50, 50, 225))
        else:
            # Translucent Emerald Green for Vacant: BGR = (50, 200, 50)
            cv2.fillPoly(overlay, [poly_np], (50, 200, 50))

    cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

    font_scale = max(0.28, min(0.55, w / 1600.0))
    thickness = 1 if w < 600 else 2

    # Draw slot polygons and clean status badges
    for idx, b in enumerate(bay_results):
        poly_np = np.array(b["polygon"], dtype=np.int32)
        if b["is_occupied"]:
            color = (60, 60, 240)
            badge_text = f"{b['slot_id']} OCC ({int(b['occupancy_ratio']*100)}%)"
        else:
            color = (30, 220, 30)
            badge_text = f"{b['slot_id']} FREE"

        cv2.polylines(annotated, [poly_np], isClosed=True, color=color, thickness=thickness)

        # Smart badge positioning (vertical offset to avoid overlap)
        bx, by = b["bbox"][0], b["bbox"][1]
        y_offset = (idx % 2) * (14 if w < 600 else 22) if "Back" in b["zone"] else 0
        badge_y = max(10, by + y_offset)

        t_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        cv2.rectangle(annotated, (bx, badge_y), (bx + t_size[0] + 6, badge_y + t_size[1] + 6), (15, 23, 42), -1)
        cv2.putText(
            annotated,
            badge_text,
            (bx + 3, badge_y + t_size[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

    # Draw detected vehicle boxes
    for veh in vehicles:
        vx1, vy1, vx2, vy2 = veh["bbox"]
        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (248, 189, 56), 1)
        v_label = f"{veh['class_name']} {int(veh['confidence']*100)}%"
        cv2.putText(
            annotated,
            v_label,
            (vx1 + 2, vy1 - 3 if vy1 > 15 else vy1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.85,
            (248, 189, 56),
            1,
            cv2.LINE_AA,
        )

    summary = {
        "image_name": image_name,
        "total_bays": total_bays,
        "vacant_count": vacant_count,
        "occupied_count": occupied_count,
        "occupancy_rate": round(occupancy_rate, 3),
        "detected_vehicles": len(vehicles),
        "confidence_threshold": conf_threshold,
        "ioa_threshold": ioa_threshold,
        "detection_method": "ROI_IoA_DualCalibrated_Pipeline",
        "borderline_count": json_payload.get("borderline_count", 0),
    }

    return {
        "summary": summary,
        "bays": bay_results,
        "vehicles": vehicles,
        "annotated_image": annotated,
        "json_payload": json_payload,
    }


def list_available_camera_angles() -> List[Dict[str, str]]:
    """Lists all available parking camera perspectives and frames from car_dataset/."""
    available = []
    if not os.path.exists(DATASET_DIR):
        return available

    config = load_slots_config()
    angles = config.get("camera_angles", {})

    # 1. Master Empty Lot Reference
    empty_path = os.path.join(DATASET_DIR, "empty lot.jpg")
    if os.path.exists(empty_path):
        meta = angles.get("empty lot.jpg", {
            "name": "Master Reference Feed — Empty Lot Calibration",
            "description": "Full-resolution un-occluded parking lot baseline with true perspective trapezoids (12 bays)"
        })
        available.append({
            "filename": "empty lot.jpg",
            "display_name": meta["name"],
            "type": "Master Reference Baseline",
            "description": meta["description"],
            "path": empty_path,
        })

    # 2. CCTV Camera Angles
    for fname, meta in angles.items():
        if fname in ("row_sequence", "empty lot.jpg"):
            continue
        fpath = os.path.join(DATASET_DIR, fname)
        if os.path.exists(fpath):
            available.append({
                "filename": fname,
                "display_name": meta["name"],
                "type": "CCTV Surveillance Angle",
                "description": meta["description"],
                "path": fpath,
            })

    # 3. Time-Series Row Frames
    for i in range(1, 13):
        fname = f"image_{i}.png"
        fpath = os.path.join(DATASET_DIR, fname)
        if os.path.exists(fpath):
            available.append({
                "filename": fname,
                "display_name": f"Time-Series Row Feed Frame #{i:02d}",
                "type": "Time-Series Row Feed",
                "description": f"Fixed surveillance camera monitoring bay row transitions #{i:02d} (12 bays)",
                "path": fpath,
            })

    return available
