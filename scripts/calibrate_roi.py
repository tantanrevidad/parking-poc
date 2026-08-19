"""
scripts/calibrate_roi.py — Smart Parking POC
---------------------------------------------
Interactive GUI Calibration Tool to click and define parking slot polygon coordinates.

Controls:
  - Left Click: Add polygon vertex (x, y)
  - 'c': Close current polygon (connects back to first vertex) and save slot
  - 'u': Undo last clicked vertex
  - 'r': Reset / clear current in-progress polygon
  - 's': Save all slots to slots_config.json
  - 'q' or ESC: Quit

Usage:
  python scripts/calibrate_roi.py --image "empty lot.jpg"
  python scripts/calibrate_roi.py --image "image_1.png"
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "slots_config.json")
DATASET_DIR = os.path.join(PROJECT_ROOT, "car_dataset")

current_poly = []
all_slots = []
slot_counter = 1
img_display = None
img_base = None


def mouse_callback(event, x, y, flags, param):
    global current_poly, img_display, img_base

    if event == cv2.EVENT_LBUTTONDOWN:
        current_poly.append([int(x), int(y)])
        redraw()


def redraw():
    global img_display, img_base, all_slots, current_poly
    img_display = img_base.copy()

    # Draw saved slots in green
    for s in all_slots:
        pts = np.array(s["polygon"], dtype=np.int32)
        cv2.polylines(img_display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        bx, by, bw, bh = cv2.boundingRect(pts)
        cv2.putText(
            img_display,
            s["id"],
            (bx + 5, by + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw in-progress polygon in yellow / cyan
    if len(current_poly) > 0:
        for idx, pt in enumerate(current_poly):
            cv2.circle(img_display, tuple(pt), 4, (0, 255, 255), -1)
            cv2.putText(
                img_display,
                str(idx + 1),
                (pt[0] + 6, pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

        if len(current_poly) > 1:
            pts = np.array(current_poly, dtype=np.int32)
            cv2.polylines(img_display, [pts], isClosed=False, color=(255, 255, 0), thickness=1)

    cv2.imshow("Parking Slot ROI Calibrator", img_display)


def main():
    global img_base, img_display, all_slots, current_poly, slot_counter

    parser = argparse.ArgumentParser(description="Interactive Parking Slot Polygon Calibrator")
    parser.add_argument("--image", type=str, default="empty lot.jpg", help="Image filename in car_dataset/")
    args = parser.parse_args()

    img_path = os.path.join(DATASET_DIR, args.image)
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    img_base = cv2.imread(img_path)
    h, w = img_base.shape[:2]

    # Load existing slots if available
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
            config = json.load(f)

    target_key = args.image
    if target_key not in config.get("camera_angles", {}):
        if "empty" in target_key.lower():
            target_key = "empty lot.jpg"
        elif "image_" in target_key.lower():
            target_key = "row_sequence"

    existing_slots = config.get("camera_angles", {}).get(target_key, {}).get("slots", [])
    all_slots = [dict(s) for s in existing_slots]
    slot_counter = len(all_slots) + 1

    print("\n" + "=" * 60)
    print(f"  PARKING SLOT ROI CALIBRATOR: {args.image} ({w}x{h})")
    print("=" * 60)
    print("  • Left Click : Add polygon vertex (x, y)")
    print("  • 'c'        : Close polygon & save slot")
    print("  • 'u'        : Undo last point")
    print("  • 'r'        : Reset in-progress polygon")
    print("  • 'd'        : Delete last saved slot")
    print("  • 's'        : Save all slots to slots_config.json")
    print("  • 'q' / ESC  : Quit")
    print("=" * 60 + "\n")

    cv2.namedWindow("Parking Slot ROI Calibrator", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Parking Slot ROI Calibrator", mouse_callback)

    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):  # ESC or q
            break

        elif key == ord("c"):  # Close & add slot
            if len(current_poly) >= 3:
                slot_id = f"F-{slot_counter:02d}"
                all_slots.append({
                    "id": slot_id,
                    "name": f"Bay {slot_id}",
                    "zone": "Front Row",
                    "polygon": [list(pt) for pt in current_poly],
                })
                print(f"Added Slot [{slot_id}] with {len(current_poly)} vertices: {current_poly}")
                slot_counter += 1
                current_poly = []
                redraw()
            else:
                print("A polygon requires at least 3 points before closing.")

        elif key == ord("u"):  # Undo point
            if current_poly:
                removed = current_poly.pop()
                print(f"Undid point: {removed}")
                redraw()

        elif key == ord("r"):  # Reset in-progress
            current_poly = []
            print("Cleared in-progress polygon.")
            redraw()

        elif key == ord("d"):  # Delete last saved slot
            if all_slots:
                removed_slot = all_slots.pop()
                slot_counter = max(1, slot_counter - 1)
                print(f"Removed Slot: {removed_slot['id']}")
                redraw()

        elif key == ord("s"):  # Save to slots_config.json
            if "camera_angles" not in config:
                config["camera_angles"] = {}

            if target_key not in config["camera_angles"]:
                config["camera_angles"][target_key] = {
                    "name": f"📷 Calibrated Feed — {args.image}",
                    "resolution": [w, h],
                    "description": f"Calibrated slot definitions ({len(all_slots)} bays)",
                    "slots": [],
                }

            config["camera_angles"][target_key]["resolution"] = [w, h]
            config["camera_angles"][target_key]["slots"] = all_slots

            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            print(f"\n Successfully saved {len(all_slots)} slots to {CONFIG_PATH} under '{target_key}'!\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
