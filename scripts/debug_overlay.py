"""
scripts/debug_overlay.py — Smart Parking POC
---------------------------------------------
Diagnostic and Regression Test Framework for Parking Space Occupancy Detection.
Renders 3 independent visual layers:
  1. Layer A: Raw YOLO detections (bounding boxes, class, confidence)
  2. Layer B: Calibrated slot polygons (geometry verification)
  3. Layer C: Combined IoA overlap, vehicle base points, and occupancy classifications

Usage:
  python scripts/debug_overlay.py --all
  python scripts/debug_overlay.py --image "empty lot.jpg"
  python scripts/debug_overlay.py --image "image_5.png" --conf 0.35 --ioa 0.30
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import parking_detector as pd_engine

DATASET_DIR = os.path.join(PROJECT_ROOT, "car_dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "debug_output")


def render_layer_detections(img_bgr: np.ndarray, vehicles: List[Dict[str, Any]]) -> np.ndarray:
    """Layer A: Raw YOLO vehicle bounding boxes, class names, and confidences."""
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]
    font_scale = max(0.35, min(0.60, w / 1600.0))

    for veh in vehicles:
        x1, y1, x2, y2 = veh["bbox"]
        conf = veh["confidence"]
        cname = veh["class_name"]
        cx, cy = veh["centroid"]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.circle(canvas, (int(cx), int(cy)), 4, (0, 255, 255), -1)
        cv2.circle(canvas, (int(cx), int(y2)), 5, (0, 0, 255), -1)

        label = f"{cname} {int(conf*100)}%"
        t_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        cv2.rectangle(canvas, (x1, y1 - t_sz[1] - 4), (x1 + t_sz[0] + 4, y1), (0, 215, 255), -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def render_layer_polygons(img_bgr: np.ndarray, slot_defs: List[Dict[str, Any]]) -> np.ndarray:
    """Layer B: Calibrated slot polygon boundaries and slot IDs."""
    canvas = img_bgr.copy()
    overlay = canvas.copy()
    h, w = canvas.shape[:2]
    font_scale = max(0.30, min(0.55, w / 1600.0))

    for slot in slot_defs:
        poly_pts = np.array(slot["polygon"], dtype=np.int32)
        cv2.fillPoly(overlay, [poly_pts], (255, 200, 100))
        cv2.polylines(canvas, [poly_pts], isClosed=True, color=(255, 120, 0), thickness=2)

    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    for slot in slot_defs:
        poly_pts = np.array(slot["polygon"], dtype=np.int32)
        bx, by, bw, bh = cv2.boundingRect(poly_pts)
        label = slot["id"]
        t_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        cv2.rectangle(canvas, (bx, by), (bx + t_sz[0] + 4, by + t_sz[1] + 4), (0, 0, 0), -1)
        cv2.putText(canvas, label, (bx + 2, by + t_sz[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas


def run_debug_frame(
    image_name: str,
    conf_threshold: float = 0.35,
    ioa_threshold: float = 0.30,
) -> Dict[str, Any]:
    """Runs complete debug pipeline for a single frame, saving layers and telemetry."""
    img_path = os.path.join(DATASET_DIR, image_name)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]

    result = pd_engine.detect_parking_spaces(
        img_bgr,
        image_name,
        conf_threshold=conf_threshold,
        ioa_threshold=ioa_threshold,
        enable_temporal_smoothing=False,
    )

    slot_defs = pd_engine.get_slot_rois_for_camera(image_name)
    vehicles = result["vehicles"]

    layer_a = render_layer_detections(img_bgr, vehicles)
    layer_b = render_layer_polygons(img_bgr, slot_defs)
    layer_c = result["annotated_image"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(image_name)[0].replace(" ", "_")

    path_a = os.path.join(OUTPUT_DIR, f"{base_name}_layerA_detections.png")
    path_b = os.path.join(OUTPUT_DIR, f"{base_name}_layerB_polygons.png")
    path_c = os.path.join(OUTPUT_DIR, f"{base_name}_layerC_combined.png")
    path_json = os.path.join(OUTPUT_DIR, f"{base_name}_telemetry.json")

    cv2.imwrite(path_a, layer_a)
    cv2.imwrite(path_b, layer_b)
    cv2.imwrite(path_c, layer_c)

    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(result["json_payload"], f, indent=2)

    return {
        "image_name": image_name,
        "shape": [h, w],
        "vehicles_detected": len(vehicles),
        "total_slots": len(slot_defs),
        "occupied_count": result["summary"]["occupied_count"],
        "vacant_count": result["summary"]["vacant_count"],
        "occupancy_rate": result["summary"]["occupancy_rate"],
        "layer_a_path": path_a,
        "layer_b_path": path_b,
        "layer_c_path": path_c,
    }


def main():
    parser = argparse.ArgumentParser(description="Debug Overlay and Regression Framework")
    parser.add_argument("--image", type=str, help="Single image filename in car_dataset/")
    parser.add_argument("--all", action="store_true", help="Run across all regression frames")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--ioa", type=float, default=0.30, help="IoA overlap threshold")

    args = parser.parse_args()

    if args.all:
        frames = ["empty lot.jpg"] + [f"image_{i}.png" for i in range(1, 13)]
        print(f"Running debug overlay across {len(frames)} regression frames...")
        print(f"Settings: conf_threshold={args.conf}, ioa_threshold={args.ioa}\n")
        print(f"{'Frame':<16} | {'Shape':<12} | {'Slots':<6} | {'Vehicles':<8} | {'Occ':<5} | {'Free':<5} | {'Occ %':<6}")
        print("-" * 75)

        for fname in frames:
            fpath = os.path.join(DATASET_DIR, fname)
            if not os.path.exists(fpath):
                continue
            res = run_debug_frame(fname, conf_threshold=args.conf, ioa_threshold=args.ioa)
            h, w = res["shape"]
            print(f"{fname:<16} | {f'{w}x{h}':<12} | {res['total_slots']:<6} | {res['vehicles_detected']:<8} | {res['occupied_count']:<5} | {res['vacant_count']:<5} | {res['occupancy_rate']*100:>5.1f}%")

        print(f"\nAll debug outputs and JSON telemetry saved to: {OUTPUT_DIR}")

    elif args.image:
        res = run_debug_frame(args.image, conf_threshold=args.conf, ioa_threshold=args.ioa)
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
